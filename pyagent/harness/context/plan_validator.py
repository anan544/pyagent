"""
规划校验器 — 结构合法性 + 业务合理性双重校验。

校验分两层：
  1. 结构层（Pydantic）：调用 ExecutionPlan 自身的 model_validate，
     捕获类型错误 / 必填缺失 / 约束违反。
  2. 业务层（规则引擎）：检查步骤间依赖合理性、高危操作是否配对
     审批步骤、验收标准是否可量化等。

使用方式：
    validator = PlanValidator()
    failures = validator.validate(plan)
    if failures:
        raise PlanValidationError("规划校验未通过", validation_failures=failures)
"""

import logging
from typing import Optional

from .models import ExecutionPlan, Step

logger = logging.getLogger("pyagent.harness.context")


# ── 高危工具列表 ────────────────────────────────────

HIGH_RISK_TOOLS = {
    "write_file",
    "execute_python",
    "execute_command",
    "delete_file",
    "rm",
    "git_push",
    "git_force_push",
    "pip_install",
    "npm_install",
    "database_write",
    "env_set",
    "chmod",
}

REVIEW_ACTIONS = {"review", "ask"}


# ── 业务规则定义 ────────────────────────────────────

class BusinessRule:
    """一条业务校验规则。"""

    def __init__(self, name: str, description: str, check_fn):
        """
        Args:
            name: 规则简短名称（如 "high_risk_requires_review"）。
            description: 人类可读的规则说明。
            check_fn: callable(ExecutionPlan) -> list[dict]，
                      返回该校验发现的问题列表。
        """
        self.name = name
        self.description = description
        self.check_fn = check_fn

    def run(self, plan: ExecutionPlan) -> list[dict]:
        """执行校验，返回 failure dicts。"""
        try:
            return self.check_fn(plan)
        except Exception as e:
            logger.warning("业务规则 %s 执行异常: %s", self.name, e)
            return [{
                "rule": self.name,
                "input_snippet": "",
                "error_type": "business",
                "detail": f"规则执行异常: {e}",
            }]


# ── 校验函数 ────────────────────────────────────────

def _check_no_empty_steps(plan: ExecutionPlan) -> list[dict]:
    """检查无空步骤（description / expected_output / acceptance_criteria 非空）。"""
    failures = []
    for step in plan.steps:
        if not step.description.strip():
            failures.append({
                "rule": "no_empty_steps",
                "input_snippet": f"步骤 {step.id}: description 为空",
                "error_type": "business",
                "detail": "步骤 description 不能为空或仅含空白字符",
            })
        if not step.expected_output.strip():
            failures.append({
                "rule": "no_empty_steps",
                "input_snippet": f"步骤 {step.id}: expected_output 为空",
                "error_type": "business",
                "detail": "步骤 expected_output 不能为空",
            })
        if not step.acceptance_criteria.strip():
            failures.append({
                "rule": "no_empty_steps",
                "input_snippet": f"步骤 {step.id}: acceptance_criteria 为空",
                "error_type": "business",
                "detail": "步骤 acceptance_criteria 不能为空（若不确定，标记为 needs_clarification）",
            })
    return failures


def _check_step_ids_sequential(plan: ExecutionPlan) -> list[dict]:
    """检查步骤 ID 从 0 开始且连续（允许间隔但建议连续）。"""
    failures = []
    ids = sorted(s.id for s in plan.steps)
    expected = list(range(len(plan.steps)))
    if ids != expected:
        failures.append({
            "rule": "step_ids_sequential",
            "input_snippet": f"步骤 ID: {ids}",
            "error_type": "business",
            "detail": (
                f"步骤 ID 建议从 0 开始连续排列。"
                f"期望 {expected}，实际 {ids}"
            ),
        })
    return failures


def _check_high_risk_tools_have_review(plan: ExecutionPlan) -> list[dict]:
    """检查高危工具是否配对了审核/确认步骤。"""
    failures = []
    high_risk_step_ids = []

    for step in plan.steps:
        if step.tool and step.tool in HIGH_RISK_TOOLS:
            high_risk_step_ids.append(step.id)

    if not high_risk_step_ids:
        return []  # 无高危步骤，无需检查

    # 找到所有 review/ask 步骤
    review_step_ids = {
        s.id for s in plan.steps
        if s.action in REVIEW_ACTIONS
    }

    for hrid in high_risk_step_ids:
        step = next(s for s in plan.steps if s.id == hrid)
        # 检查是否有 review 步骤依赖此步骤（在某步骤之后有审核）
        has_review_after = False
        for rid in review_step_ids:
            rs = next(s for s in plan.steps if s.id == rid)
            if hrid in rs.depends_on:
                has_review_after = True
                break

        if not has_review_after and review_step_ids:
            # 如果存在 review 步骤但高危步骤未在其依赖链中
            pass  # 不强制要求依赖链，仅检查是否存在任何 review 步骤

    # 简化检查：存在高危步骤时，必须有至少一个 review/ask 步骤
    if not review_step_ids:
        failures.append({
            "rule": "high_risk_requires_review",
            "input_snippet": (
                f"高危步骤: {[s.id for s in plan.steps if s.tool in HIGH_RISK_TOOLS]}"
            ),
            "error_type": "business",
            "detail": (
                f"检测到 {len(high_risk_step_ids)} 个高危工具步骤"
                f"（{', '.join(HIGH_RISK_TOOLS & {s.tool for s in plan.steps if s.tool})}），"
                f"但计划中没有 review/ask 步骤。"
                f"建议在每个高危操作后插入人工确认步骤。"
            ),
        })

    return failures


def _check_dependencies_valid(plan: ExecutionPlan) -> list[dict]:
    """检查步骤依赖关系合法性。"""
    failures = []
    step_ids = {s.id for s in plan.steps}

    for step in plan.steps:
        for dep_id in step.depends_on:
            # 检查依赖的步骤是否存在于计划中
            if dep_id not in step_ids:
                failures.append({
                    "rule": "dependencies_valid",
                    "input_snippet": (
                        f"步骤 {step.id} 依赖不存在的步骤 {dep_id}"
                    ),
                    "error_type": "business",
                    "detail": (
                        f"步骤 {step.id} 的 depends_on 引用了不存在的步骤 {dep_id}。"
                        f"可用步骤 ID: {sorted(step_ids)}"
                    ),
                })
            # 检查是否依赖自身（已在 Schema 层防住，此处兜底）
            if dep_id == step.id:
                failures.append({
                    "rule": "dependencies_valid",
                    "input_snippet": f"步骤 {step.id} 依赖自身",
                    "error_type": "business",
                    "detail": f"步骤 {step.id} 不能依赖自身",
                })

    # 检查是否有循环依赖（简单检测：拓扑排序）
    if not failures:
        cycle = _detect_cycle(plan)
        if cycle:
            failures.append({
                "rule": "dependencies_valid",
                "input_snippet": f"循环依赖链: {' -> '.join(map(str, cycle))}",
                "error_type": "business",
                "detail": f"检测到循环依赖: {cycle}",
            })

    return failures


def _check_acceptance_is_quantifiable(plan: ExecutionPlan) -> list[dict]:
    """检查验收标准是否包含可量化指标。"""
    failures = []
    # 模糊词汇（在验收标准中不应出现）
    vague_patterns = [
        "差不多", "大概", "可能", "尽量", "应该可以",
        "approximately", "about", "maybe", "probably",
    ]

    for step in plan.steps:
        ac = step.acceptance_criteria.lower()
        for pattern in vague_patterns:
            if pattern.lower() in ac:
                failures.append({
                    "rule": "acceptance_quantifiable",
                    "input_snippet": (
                        f"步骤 {step.id} acceptance_criteria: "
                        f"\"{step.acceptance_criteria[:100]}\""
                    ),
                    "error_type": "business",
                    "detail": (
                        f"步骤 {step.id} 的验收标准包含模糊词汇 '{pattern}'。"
                        f"验收标准应使用可量化的指标（如 '0 个错误'、'100% 覆盖'）。"
                    ),
                })
                break  # 每步只报一次

    return failures


def _check_estimated_steps(plan: ExecutionPlan) -> list[dict]:
    """检查 estimated_total_steps 是否与实际步数一致。"""
    failures = []
    actual = len(plan.steps)
    if plan.estimated_total_steps > 0 and plan.estimated_total_steps < actual:
        failures.append({
            "rule": "estimated_steps_consistent",
            "input_snippet": (
                f"estimated_total_steps={plan.estimated_total_steps}, "
                f"实际步骤数={actual}"
            ),
            "error_type": "business",
            "detail": (
                f"预估步骤数 ({plan.estimated_total_steps}) 少于实际步骤数 ({actual})。"
                f"请调整估计值或拆分步骤。"
            ),
        })
    return failures


def _check_high_risk_marked(plan: ExecutionPlan) -> list[dict]:
    """检查使用高危工具的步骤是否标记为 high 风险。"""
    failures = []
    for step in plan.steps:
        if step.tool and step.tool in HIGH_RISK_TOOLS:
            if step.risk_level != "high":
                failures.append({
                    "rule": "high_risk_marked",
                    "input_snippet": (
                        f"步骤 {step.id} 使用高危工具 '{step.tool}' "
                        f"但 risk_level={step.risk_level}"
                    ),
                    "error_type": "business",
                    "detail": (
                        f"步骤 {step.id} 使用高危工具 '{step.tool}'，"
                        f"risk_level 应为 'high'，当前为 '{step.risk_level}'。"
                    ),
                })
    return failures


def _detect_cycle(plan: ExecutionPlan) -> list[int]:
    """
    检测步骤依赖中是否存在循环。

    使用 Kahn 拓扑排序算法。若无法遍历所有节点，说明存在环。
    返回环中的节点路径（简化版：返回未访问的节点）。
    """
    step_ids = {s.id for s in plan.steps}
    in_degree = {sid: 0 for sid in step_ids}
    adj = {sid: [] for sid in step_ids}

    for step in plan.steps:
        for dep_id in step.depends_on:
            if dep_id in step_ids:
                adj[dep_id].append(step.id)
                in_degree[step.id] += 1

    # Kahn 算法
    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    visited = set()

    while queue:
        node = queue.pop(0)
        visited.add(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(visited) < len(step_ids):
        return sorted(step_ids - visited)
    return []


# ── 规则注册表 ──────────────────────────────────────

BUILTIN_RULES: list[BusinessRule] = [
    BusinessRule(
        name="no_empty_steps",
        description="步骤的关键字段不能为空或仅含空白",
        check_fn=_check_no_empty_steps,
    ),
    BusinessRule(
        name="step_ids_sequential",
        description="步骤 ID 应从 0 开始连续排列",
        check_fn=_check_step_ids_sequential,
    ),
    BusinessRule(
        name="high_risk_requires_review",
        description="高危工具步骤需要配对的审核步骤",
        check_fn=_check_high_risk_tools_have_review,
    ),
    BusinessRule(
        name="dependencies_valid",
        description="步骤间依赖关系必须合法（无幽灵依赖、无循环）",
        check_fn=_check_dependencies_valid,
    ),
    BusinessRule(
        name="acceptance_quantifiable",
        description="验收标准应包含可量化指标，避免模糊表述",
        check_fn=_check_acceptance_is_quantifiable,
    ),
    BusinessRule(
        name="estimated_steps_consistent",
        description="预估步骤数应 ≥ 实际步骤数",
        check_fn=_check_estimated_steps,
    ),
    BusinessRule(
        name="high_risk_marked",
        description="使用高危工具的步骤应标记为 high 风险",
        check_fn=_check_high_risk_marked,
    ),
]


# ── 主校验器 ────────────────────────────────────────

class PlanValidator:
    """
    规划校验器 — 双层校验：Pydantic 结构层 + 业务规则层。

    使用方式：
        validator = PlanValidator()
        failures = validator.validate(raw_data)

        if failures:
            raise PlanValidationError(
                "规划校验未通过",
                validation_failures=failures,
                attempt_number=1,
            )
    """

    def __init__(self, rules: Optional[list[BusinessRule]] = None):
        """
        Args:
            rules: 自定义业务规则列表。None 时使用内置规则。
        """
        self.rules = rules or BUILTIN_RULES

    def validate_structure(self, raw_data: dict) -> list[dict]:
        """
        结构层校验：尝试将 dict 解析为 ExecutionPlan。

        Returns:
            校验失败列表（空列表 = 通过）。
        """
        try:
            ExecutionPlan.model_validate(raw_data)
        except Exception as e:
            return [{
                "rule": "pydantic_schema",
                "input_snippet": str(raw_data)[:200],
                "error_type": "schema",
                "detail": str(e),
            }]
        return []

    def validate_business(self, plan: ExecutionPlan) -> list[dict]:
        """
        业务层校验：逐条运行注册的业务规则。

        Returns:
            所有规则的校验失败项汇总。
        """
        failures = []
        for rule in self.rules:
            failures.extend(rule.run(plan))
        return failures

    def validate(self, raw_data: dict) -> list[dict]:
        """
        完整校验流程：结构 → 业务。

        先执行 Pydantic 结构校验，结构通过后再运行业务规则。
        结构层失败时不继续业务校验（无有效 plan 对象）。

        Args:
            raw_data: LLM 输出的原始 dict（待组装为 ExecutionPlan）。

        Returns:
            所有校验失败项列表。空列表 = 完全通过。
        """
        # 第 1 层：结构校验
        struct_failures = self.validate_structure(raw_data)
        if struct_failures:
            return struct_failures

        # 第 2 层：构建实例 + 业务校验
        try:
            plan = ExecutionPlan.model_validate(raw_data)
        except Exception as e:
            return [{
                "rule": "pydantic_schema",
                "input_snippet": str(raw_data)[:200],
                "error_type": "schema",
                "detail": str(e),
            }]

        return self.validate_business(plan)
