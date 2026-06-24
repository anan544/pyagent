"""
上下文组装器 — 数据模型。

定义了 PEVR 循环所需的所有强类型数据结构：
    - PEVRPhase: 四阶段枚举
    - WorkingMemory: 跨阶段共享的工作记忆
    - ContextRequest: 组装请求入参
    - SlotContent / SlotSet: 四槽位模型
    - PEVRResult: PEVR 循环最终产出
"""

from enum import Enum
from typing import Any, Optional, List
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── PEVR 阶段 ──────────────────────────────────────

class PEVRPhase(str, Enum):
    """PEVR 四阶段：规划 → 执行 → 验收 → 修补。"""
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    REPAIR = "repair"


# ── WorkingMemory — 强类型跨阶段记忆 ────────────────

class StepResult(BaseModel):
    """单步执行结果。"""
    step: str = ""
    result: str = ""
    status: str = "pending"  # pending | success | failed
    artifacts: dict[str, Any] = Field(default_factory=dict)


class WorkingMemory(BaseModel):
    """
    跨阶段共享的工作记忆。

    持久化于 PEVR 循环的整个生命周期，各阶段可读可写。

    plan 字段采用写时冻结策略：首次 set_plan() 写入后不可更改，
    后续所有阶段均引用此快照，防止模型在执行中遗忘或篡改计划。
    """

    artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="key → 内容映射。如 {'plan.md': '...', 'report.py': '...'}",
    )
    step_results: list[StepResult] = Field(
        default_factory=list,
        description="各执行步骤的结果记录",
    )
    notes: str = Field(
        default="",
        description="自由文本备注，Agent 可在各阶段追加",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="元数据：当前步骤编号、失败次数等",
    )
    plan: Optional[Any] = Field(
        default=None,
        description="冻结的 ExecutionPlan 快照。通过 set_plan() 写入，写入后不可变。",
    )
    # ── 1.5.3 新增：状态机相关字段 ──
    acceptance_criteria: str = Field(
        default="",
        description="验收标准原文（run() 入口时存储，VERIFY 阶段强制从此读取）。",
    )
    current_step_index: int = Field(
        default=0,
        description="当前执行步骤索引（1-based，0 表示未开始执行）。",
    )
    repair_count: int = Field(
        default=0,
        description="当前已进行的修补次数。",
    )

    def add_artifact(self, key: str, content: str):
        """添加或更新一个临时产物。"""
        self.artifacts[key] = content

    def add_step_result(self, step: str, result: str, status: str = "success"):
        """记录一个步骤的执行结果。"""
        self.step_results.append(StepResult(step=step, result=result, status=status))

    def last_failure(self) -> Optional[StepResult]:
        """获取最近的失败步骤（供修补阶段使用）。"""
        for sr in reversed(self.step_results):
            if sr.status == "failed":
                return sr
        return None

    def set_plan(self, plan) -> None:
        """
        冻结写入执行计划快照。

        仅允许写入一次（None → ExecutionPlan）。写入后尝试再次调用
        将抛出 ValueError，防止模型在执行中遗忘或篡改计划。

        Args:
            plan: 校验通过的 ExecutionPlan 实例。

        Raises:
            ValueError: 若 plan 已被设置（非 None）。
        """
        if self.plan is not None:
            raise ValueError(
                "Plan 快照已冻结，不可重复写入。"
                "如需更新计划，请创建新的 WorkingMemory 实例。"
            )
        self.plan = plan


# ── 四槽位模型 ──────────────────────────────────────

class SlotContent(BaseModel):
    """单个槽位的内容 + Token 预算。"""
    name: str = ""
    content: str = ""
    max_tokens: int = 0
    priority: int = 0  # 裁剪优先级，越高越晚被裁
    source: str = ""   # 内容来源标记（如 agent.md 文件路径）


class SlotSet(BaseModel):
    """
    四槽位集合：System / Plan / History / WorkingMemory。

    每个槽位有独立的 Token 预算和裁剪优先级。
    """
    system: SlotContent = Field(
        default_factory=lambda: SlotContent(name="system", priority=10)
    )
    plan: SlotContent = Field(
        default_factory=lambda: SlotContent(name="plan", priority=7)
    )
    history: SlotContent = Field(
        default_factory=lambda: SlotContent(name="history", priority=5)
    )
    working_memory: SlotContent = Field(
        default_factory=lambda: SlotContent(name="working_memory", priority=3)
    )
    # ── 1.5.4 新增：观测提示槽位 ──
    observability_hints: SlotContent = Field(
        default_factory=lambda: SlotContent(name="observability_hints", priority=2)
    )

    def total_tokens(self) -> int:
        """估算五槽位总 Token。"""
        return sum(
            max(1, len(s.content) // 4) if s.content else 0
            for s in [self.system, self.plan, self.history,
                      self.working_memory, self.observability_hints]
        )


# ── 上下文请求 ──────────────────────────────────────

class ContextRequest(BaseModel):
    """
    上下文组装请求。

    由 PEVRRunner 在每阶段调用 ContextAssembler.assemble() 时传入。
    """
    phase: PEVRPhase = PEVRPhase.EXECUTE
    system_prompt: str = ""
    context_files: list[str] = Field(
        default_factory=list,
        description="已加载的 agent.md 等多文件内容列表",
    )
    plan: str = Field(default="", description="当前计划快照")
    history: list[Any] = Field(
        default_factory=list,
        description="近期对话历史（Message 对象或 str）",
    )
    working_memory: WorkingMemory = Field(default_factory=WorkingMemory)
    token_budget: Optional[Any] = Field(
        default=None,
        description="TokenBudget 实例",
    )
    acceptance_criteria: str = Field(
        default="",
        description="验收标准原文（验收阶段强制注入）",
    )
    failure_summary: str = Field(
        default="",
        description="失败原因摘要（修补阶段使用）",
    )
    few_shot_examples: list[dict] = Field(
        default_factory=list,
        description="Few-shot 示例列表（Plan 阶段注入模板）。"
                    "每项含: label, task, output (ExecutionPlan JSON dict)。",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ── 上下文结果 ──────────────────────────────────────

@dataclass
class ContextResult:
    """ContextAssembler.assemble() 的返回结果。"""
    messages: list = field(default_factory=list)
    """组装好的消息列表，可直接传给 LLM"""

    total_tokens: int = 0
    """组装后消息的 Token 总数（估算）"""

    slot_tokens: dict[str, int] = field(default_factory=dict)
    """各槽位消耗 Token 明细"""

    was_trimmed: bool = False
    """是否触发了裁剪"""

    trimmed_details: str = ""
    """裁剪详情（调试用）"""


# ── PEVR 结果 ───────────────────────────────────────

@dataclass
class PEVRResult:
    """PEVRRunner.run() 的最终产出。"""
    success: bool = False
    plan: str = ""
    outputs: list[str] = field(default_factory=list)
    """执行阶段产出的文件/代码列表"""
    verification: str = ""
    """验收结论"""
    repair_count: int = 0
    """修补次数"""
    total_tokens_used: int = 0
    """全阶段 Token 消耗合计"""
    working_memory: Optional[WorkingMemory] = None


# ── 结构化规划：ExecutionPlan Schema ────────────────

class Step(BaseModel):
    """
    执行计划中的单个步骤。

    所有字段必填，禁止 Any 类型 — 确保 LLM 产出可验证。
    """
    id: int = Field(..., ge=0, description="步骤序号（从 0 开始）")
    description: str = Field(
        ..., min_length=1,
        description="步骤描述，说明要做什么",
    )
    action: str = Field(
        ...,
        min_length=1,
        description="动作类型：read / write / execute / review / ask",
    )
    tool: Optional[str] = Field(
        default=None,
        description="预计使用的工具名称。若无需工具则为 None",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="工具调用的参数（键值对）",
    )
    expected_output: str = Field(
        ...,
        min_length=1,
        description="本步骤的预期产出描述",
    )
    acceptance_criteria: str = Field(
        ...,
        min_length=1,
        description="本步骤的验收标准（需可量化）",
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="依赖的前置步骤 ID 列表",
    )
    risk_level: str = Field(
        default="low",
        description="风险等级：low / medium / high",
    )

    @field_validator("action")
    @classmethod
    def action_must_be_valid(cls, v: str) -> str:
        allowed = {"read", "write", "execute", "review", "ask"}
        if v not in allowed:
            raise ValueError(f"action 必须是 {allowed} 之一，收到: {v}")
        return v

    @field_validator("risk_level")
    @classmethod
    def risk_level_must_be_valid(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        if v not in allowed:
            raise ValueError(f"risk_level 必须是 {allowed} 之一，收到: {v}")
        return v

    @field_validator("depends_on")
    @classmethod
    def depends_on_no_self_ref(cls, v: list[int], info) -> list[int]:
        step_id = info.data.get("id")
        if step_id is not None and step_id in v:
            raise ValueError(f"步骤 {step_id} 不能依赖自身")
        return v


class AcceptanceCriteria(BaseModel):
    """
    结构化验收标准 — 拆分为可逐条验证的条件列表。
    """
    criteria: list[str] = Field(
        ...,
        min_length=1,
        description="逐条验收条件。每条需可量化/可判定真伪。",
    )
    quantifiable: bool = Field(
        default=True,
        description="所有条件是否均可量化（非模糊描述）",
    )
    needs_clarification: list[str] = Field(
        default_factory=list,
        description="需用户澄清的模糊点。非空时表示计划存在不确定性。",
    )


class ExecutionPlan(BaseModel):
    """
    PEVR 执行计划 — LLM 生成的强类型规划。

    不可变（frozen=True）：一旦创建 / 校验通过，内容锁定。
    后续所有阶段均引用此快照，防止模型在执行中遗忘或篡改计划。

    使用方式：
        plan = ExecutionPlan(
            steps=[Step(id=0, description="...", ...)],
            ...
        )
        # plan.steps = []  # ❌ 抛出 ValidationError（frozen）
    """
    model_config = ConfigDict(frozen=True)

    steps: list[Step] = Field(
        ...,
        min_length=1,
        description="执行步骤列表（至少 1 步）",
    )
    dependencies: dict[int, list[int]] = Field(
        default_factory=dict,
        description="步骤间依赖关系。key=步骤ID, value=依赖的步骤ID列表",
    )
    risk_level: str = Field(
        default="low",
        description="整体风险等级：low / medium / high",
    )
    estimated_total_steps: int = Field(
        default=0,
        description="预估总步骤数（≥ len(steps)）",
    )
    needs_clarification: list[str] = Field(
        default_factory=list,
        description="需用户澄清的模糊点。非空时表示计划存在不确定性，"
                    "应由人类确认后再执行。",
    )
    plan_summary: str = Field(
        default="",
        description="计划摘要（1-3 句话）",
    )
    # ── 1.5.3 新增：修补阶段工具白名单 ──
    allowed_repair_tools: list[str] = Field(
        default_factory=list,
        description="REPAIRING 阶段允许使用的工具名称列表。"
                    "空列表时使用系统默认白名单（read_file, search_content）。",
    )
    # ── 1.5.4 新增：全生命周期追踪 ID ──
    trace_id: str = Field(
        default="",
        description="全生命周期追踪 ID（Plan 生成时创建，UUID4）。"
                    "贯穿 PLANNING→EXECUTING→VERIFYING→REPAIRING 全阶段，"
                    "断点恢复后延续原追踪链路。",
    )

    @field_validator("risk_level")
    @classmethod
    def risk_level_must_be_valid(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        if v not in allowed:
            raise ValueError(f"risk_level 必须是 {allowed} 之一，收到: {v}")
        return v

    @field_validator("steps")
    @classmethod
    def steps_have_unique_ids(cls, v: list[Step]) -> list[Step]:
        ids = [s.id for s in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"步骤 ID 必须唯一，收到: {ids}")
        return v


# ── 异常 ──────────────────────────────────────────

class InvalidStateError(Exception):
    """
    阶段入口状态不合法时抛出。

    例如：VERIFY 阶段时 WorkingMemory.plan 为空，
    或 acceptance_criteria 缺失——这些是 Plan 快照损坏的信号，
    应提前暴露而非静默降级。
    """

    def __init__(self, message: str):
        super().__init__(message)


class PlanValidationError(Exception):
    """
    规划校验失败时抛出的异常。

    携带 validation_failures 字段，记录每次校验失败的具体信息，
    方便测试断言和调试追踪。
    """

    def __init__(
        self,
        message: str,
        validation_failures: list[dict] = None,
        attempt_number: int = 0,
    ):
        """
        Args:
            message: 错误描述。
            validation_failures: 校验失败明细列表，每项含:
                - rule: 失败规则名称
                - input_snippet: 触发失败的输入片段（截断至 200 字符）
                - error_type: schema | business | convergence
                - detail: 详细错误说明
            attempt_number: 第几次尝试失败（0-based）。
        """
        super().__init__(message)
        self.validation_failures = validation_failures or []
        self.attempt_number = attempt_number
