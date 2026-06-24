"""
三级权限门控 — 计划级预审 / 步骤级执行检查 / 修补特权降级。

职责：
    - 计划级预审：PLANNING→EXECUTING 转换时扫描高危工具组合
    - 步骤级检查：EXECUTING 内工具调用前比对参数与 Plan 快照
    - 修补特权降级：REPAIRING 阶段动态限制可用工具集

设计原则：
    - 纯逻辑模块，不依赖 Agent/LLM/IO
    - 所有校验基准锚定 ExecutionPlan 快照（唯一可信源）
    - ScopedToolRegistry 不修改原 ToolRegistry 任何状态
"""

from typing import Any, Optional
from ...core.message import ToolMessage
from ...tools.registry import ToolRegistry
from .observability import compute_step_fingerprint, compute_risk_score


# ── 高危工具组合规则 ──────────────────────────────────

# 任一组合中的所有工具同时出现在 Plan.steps 中时触发告警
HIGH_RISK_COMBOS: list[set[str]] = [
    {"write_file", "execute_python"},
    {"write_file", "execute_command"},
    {"delete_file"},
    {"execute_python", "execute_command"},
    {"git_push", "git_force_push"},
    {"pip_install", "execute_python"},
    {"npm_install", "execute_python"},
]


# ── ScopedToolRegistry ──────────────────────────────────

class ScopedToolRegistry:
    """
    工具注册表装饰器 — 仅暴露白名单内的工具。

    不修改原 ToolRegistry 的任何状态（不调用 register/unregister），
    在 REPAIRING 阶段创建，退出时自动销毁。

    使用方式：
        original = agent.tool_registry
        scoped = ScopedToolRegistry(original, {"read_file", "search_content"})
        agent.tool_registry = scoped  # 临时替换
        # ... REPAIRING 阶段 ...
        agent.tool_registry = original  # 恢复
    """

    def __init__(self, parent: ToolRegistry, allowed: set[str]):
        """
        Args:
            parent: 原始 ToolRegistry 实例。
            allowed: 允许使用的工具名称集合。
        """
        self._parent = parent
        self._allowed = allowed

    @property
    def allowed_tools(self) -> set[str]:
        """返回当前白名单（只读）。"""
        return set(self._allowed)

    def get_all_schemas(self) -> list[dict]:
        """
        仅返回白名单内工具的 schema。

        这是传给 LLM 的 tools 参数来源——LLM 只能"看到"白名单工具。
        """
        return [
            schema for schema in self._parent.get_all_schemas()
            if schema["function"]["name"] in self._allowed
        ]

    def list_names(self) -> list[str]:
        """仅返回白名单内的工具名称。"""
        return [name for name in self._parent.list_names()
                if name in self._allowed]

    async def execute(
        self, name: str, call_id: str, arguments: dict[str, Any]
    ) -> ToolMessage:
        """
        执行工具调用——非白名单工具返回"未授权"错误而非实际执行。

        Args:
            name: 工具名称。
            call_id: LLM tool_call ID。
            arguments: 工具参数。

        Returns:
            ToolMessage: 执行结果或"工具未授权"错误。
        """
        if name not in self._allowed:
            return ToolMessage(
                content=(
                    f"工具 '{name}' 在当前阶段（REPAIRING）未授权。"
                    f"当前仅允许: {', '.join(sorted(self._allowed))}"
                ),
                tool_call_id=call_id,
                name=name,
            )
        return await self._parent.execute(name, call_id, arguments)


# ── PermissionGate ──────────────────────────────────────

class PermissionGate:
    """
    三级权限门控 — 纯逻辑，无副作用。

    所有校验基准必须来自冻结的 ExecutionPlan 快照（唯一可信源）。
    禁止从 LLM 当前输出或 WorkingMemory 动态字段提取校验基准。
    """

    # 默认修补阶段白名单（当 plan.allowed_repair_tools 为空时使用）
    DEFAULT_REPAIR_WHITELIST: set[str] = {"read_file", "search_content"}

    # ── 计划级预审 ──────────────────────────────────

    @staticmethod
    def plan_level_audit(plan, audit_sink: Optional[list] = None) -> list[dict]:
        """
        扫描 Plan 中所有 required_tools，检测高危工具组合。

        从冻结的 ExecutionPlan.steps 中提取所有 tool 声明，
        与 HIGH_RISK_COMBOS 对照，返回审计发现列表。

        1.5.4: 通过 audit_sink 同步生成 SecurityAuditEvent，
        确保拦截决策本身也被完整记录（即使工具未执行）。

        Args:
            plan: 冻结的 ExecutionPlan 实例。
            audit_sink: 审计事件接收列表。传入时，每个高危发现会追加
                        SecurityAuditEvent 到该列表。

        Returns:
            审计发现列表，每项包含:
            - combo: 触发的高危工具组合
            - steps: 涉及的步骤 ID 列表
            - severity: "high" | "medium"
            - rule_id: 匹配的规则 ID（1.5.4 新增，可追溯）
        """
        from .observability import SecurityAuditEvent

        # 收集所有步骤中声明的工具
        declared_tools: set[str] = set()
        tool_to_steps: dict[str, list[int]] = {}
        step_params_map: dict[str, dict] = {}  # tool → step.params for fingerprint

        for step in plan.steps:
            if step.tool:
                declared_tools.add(step.tool)
                tool_to_steps.setdefault(step.tool, []).append(step.id)
                step_params_map[step.tool] = step.params or {}

        findings: list[dict] = []
        trace_id = getattr(plan, 'trace_id', '')

        for idx, combo in enumerate(HIGH_RISK_COMBOS):
            matched = combo & declared_tools
            if not matched:
                continue

            rule_id = f"HIGH_RISK_COMBOS[{idx}]"
            risk_score = compute_risk_score(matched)

            if len(matched) >= 2:
                # 多工具高危组合
                affected_steps = []
                for tool in matched:
                    affected_steps.extend(tool_to_steps.get(tool, []))
                finding = {
                    "combo": sorted(matched),
                    "steps": sorted(set(affected_steps)),
                    "severity": "high",
                    "reason": f"高危工具组合: {' + '.join(sorted(matched))}",
                    "rule_id": rule_id,
                }
                findings.append(finding)

                # 1.5.4: 生成审计事件
                if audit_sink is not None:
                    fp = compute_step_fingerprint(
                        step_params_map.get(next(iter(matched)), {})
                    )
                    audit_sink.append(SecurityAuditEvent(
                        decision="BLOCK",
                        rule_id=rule_id,
                        plan_step_fingerprint=fp,
                        risk_score=risk_score,
                        tool_name=", ".join(sorted(matched)),
                        phase="planning",
                        trace_id=trace_id,
                        metadata={"affected_steps": sorted(set(affected_steps))},
                    ))

            elif len(matched) == 1:
                # 单工具高危（如 delete_file）
                tool = next(iter(matched))
                finding = {
                    "combo": [tool],
                    "steps": tool_to_steps.get(tool, []),
                    "severity": "high",
                    "reason": f"高危工具: {tool}",
                    "rule_id": rule_id,
                }
                findings.append(finding)

                # 1.5.4: 生成审计事件
                if audit_sink is not None:
                    fp = compute_step_fingerprint(
                        step_params_map.get(tool, {})
                    )
                    audit_sink.append(SecurityAuditEvent(
                        decision="BLOCK",
                        rule_id=rule_id,
                        plan_step_fingerprint=fp,
                        risk_score=risk_score,
                        tool_name=tool,
                        phase="planning",
                        trace_id=trace_id,
                        metadata={"affected_steps": tool_to_steps.get(tool, [])},
                    ))

        return findings

    # ── 步骤级执行检查 ───────────────────────────────

    @staticmethod
    def step_level_check(
        step, tool_name: str, actual_params: dict,
        audit_sink: Optional[list] = None,
    ) -> bool:
        """
        比对 LLM 实际 tool_call 参数与 Plan 快照中声明的参数范围。

        唯一可信源：冻结的 ExecutionPlan.steps[i].params。
        禁止从 WorkingMemory 动态字段或 LLM 当前输出提取校验基准。

        检查规则：
            1. 工具名称必须匹配 Plan 中声明的 tool（如果声明了）。
            2. 实际参数不得超出 Plan 声明的参数范围。
               - Plan 中已声明的 key，实际值必须匹配（值相等 或 包含关系）。
               - 实际参数中 Plan 未声明的 key 视为越界。

        1.5.4: 通过 audit_sink 同步生成 SecurityAuditEvent，
        无论 ALLOW 还是 BLOCK 均记录完整决策链。

        Args:
            step: 冻结的 Step 模型（来自 ExecutionPlan）。
            tool_name: LLM 实际调用的工具名称。
            actual_params: LLM 实际传递的参数。
            audit_sink: 审计事件接收列表。传入时追加 SecurityAuditEvent。

        Returns:
            True 如果参数合规，False 如果越界需阻断。
        """
        from .observability import SecurityAuditEvent

        declared_params: dict = step.params or {}
        declared_tool = step.tool
        step_fp = compute_step_fingerprint(declared_params)
        trace_id = getattr(step, 'trace_id', '')
        plan = None  # 从 step 无法直接获取 plan 级别 trace_id

        result = True
        block_reason = ""

        # 1. 工具名称检查
        if declared_tool is not None and tool_name != declared_tool:
            result = False
            block_reason = (
                f"工具名不匹配: 声明={declared_tool}, 实际={tool_name}"
            )

        # 2. 如果 Plan 未声明任何参数约束，允许通过（LLM 自由发挥）
        elif not declared_params:
            result = True

        else:
            # 3. 检查每个实际参数
            for key, actual_value in actual_params.items():
                if key not in declared_params:
                    # 实际参数中出现了 Plan 未声明的 key → 越界
                    result = False
                    block_reason = f"越界参数: '{key}' 未在 Plan 中声明"
                    break

                declared_value = declared_params[key]

                # 字符串参数：值必须相等
                if isinstance(declared_value, str) and isinstance(actual_value, str):
                    if declared_value != actual_value:
                        result = False
                        block_reason = (
                            f"参数值不匹配: '{key}' "
                            f"声明='{declared_value}', 实际='{actual_value}'"
                        )
                        break
                # 其他类型：直接比较
                elif declared_value != actual_value:
                    result = False
                    block_reason = (
                        f"参数值不匹配: '{key}' "
                        f"声明={declared_value}, 实际={actual_value}"
                    )
                    break

        # 1.5.4: 生成审计事件
        if audit_sink is not None:
            audit_sink.append(SecurityAuditEvent(
                decision="ALLOW" if result else "BLOCK",
                rule_id="step_level_check",
                plan_step_fingerprint=step_fp,
                risk_score=0 if result else 70,
                tool_name=tool_name,
                tool_params=actual_params,
                phase="executing",
                trace_id=trace_id,
                metadata={
                    "step_id": step.id,
                    "declared_tool": declared_tool,
                    "block_reason": block_reason if not result else "",
                },
            ))

        return result

    # ── 修补阶段工具白名单 ───────────────────────────

    @staticmethod
    def get_repair_whitelist(plan) -> set[str]:
        """
        从 Plan 快照中读取修补阶段允许的工具白名单。

        优先使用 plan.allowed_repair_tools，为空时使用默认白名单。

        Args:
            plan: 冻结的 ExecutionPlan 实例。

        Returns:
            允许使用的工具名称集合。
        """
        if plan and hasattr(plan, 'allowed_repair_tools') and plan.allowed_repair_tools:
            return set(plan.allowed_repair_tools)
        return PermissionGate.DEFAULT_REPAIR_WHITELIST.copy()
