"""
PEVRRunner — PEVR 循环编排器（状态机驱动）。

将规划（Plan）→ 执行（Execute）→ 验收（Verify）→ 修补（Repair）
四个阶段编排为完整的一级循环，由 PEVRStateMachine 驱动状态流转。

状态流转图：
    PLANNING ──(plan_done)──▶ EXECUTING ──(all_steps_done)──▶ VERIFYING ──(passed)──▶ COMPLETED
        │                        │                               │
        └──(plan_failed)──▶ FAILED  ├──(permission_violation)──▶ REPAIRING ──(done)──▶ VERIFYING
                                     │                               │
                                     └──(fatal_error)──▶ FAILED ◀──(failed+max)─────│

1.5.3 关键变更：
    - 用枚举驱动状态机替代手写 if-else 流转
    - 三级权限门控嵌入状态转换边
    - 修补阶段 ScopedToolRegistry 隔离
    - 验收标准锚定 WorkingMemory.plan 快照
    - 每次状态转换后写入 PEVRCheckpoint

Agent 保持纯净：PEVRRunner 仅通过 Agent.run_with_messages() 和
Agent.llm.generate() 与 Agent 交互，不修改 Agent 内部状态。
"""

import asyncio
import json
import logging
import re
from typing import Optional

from .models import (
    PEVRPhase,
    PEVRResult,
    ContextRequest,
    WorkingMemory,
    ExecutionPlan,
    PlanValidationError,
)
from .assembler import ContextAssembler, PEVRContextAssembler
from .state_machine import (
    PEVRStateMachine,
    PEVRState,
    PEVREvent,
    InvalidTransitionError,
    GuardRejectedError,
)
from .permission import PermissionGate, ScopedToolRegistry
from .checkpoint import PEVRCheckpoint, save_checkpoint, delete_checkpoint
from .loader import ContextFileLoader
from .plan_validator import PlanValidator
from .observability import (
    ObservabilityContext,
    SecurityAuditEvent,
    TraceContext,
    compute_step_fingerprint,
)
from .repair_context import (
    RepairContext,
    CircuitBreaker,
    ConvergenceDetector,
    RepairLog,
)
from ...core.message import SystemMessage, UserMessage, AssistantMessage

logger = logging.getLogger("pyagent.harness.context")


# ── 常量和限制 ────────────────────────────────────

DEFAULT_MAX_REPAIRS = 3
DEFAULT_TOTAL_BUDGET = 64000
MAX_PLAN_RETRIES = 3

# ── Few-shot 示例（Plan 阶段注入）──────────────────

FEW_SHOT_EXAMPLES = [
    {
        "label": "单步审查",
        "task": "审查 agent.py 的 ReAct 循环实现",
        "output": {
            "steps": [
                {
                    "id": 0,
                    "description": "阅读 agent.py 中 ReAct 循环的实现代码",
                    "action": "read",
                    "tool": "read_file",
                    "params": {"file_path": "pyagent/core/agent.py"},
                    "expected_output": "完整理解 _react_loop() 方法的逻辑流程",
                    "acceptance_criteria": "已确认 _react_loop 方法的输入、循环条件、工具调用、输出格式",
                    "depends_on": [],
                    "risk_level": "low",
                },
                {
                    "id": 1,
                    "description": "审查循环终止条件是否在所有路径上可达",
                    "action": "review",
                    "tool": None,
                    "params": {},
                    "expected_output": "确认不存在无限循环的可能路径",
                    "acceptance_criteria": "所有分支均在 max_iterations 内终止，无逻辑死角",
                    "depends_on": [0],
                    "risk_level": "medium",
                },
                {
                    "id": 2,
                    "description": "汇总审查结论并输出报告",
                    "action": "write",
                    "tool": "write_file",
                    "params": {"file_path": "review_report.md"},
                    "expected_output": "包含审查发现的 Markdown 报告",
                    "acceptance_criteria": "报告包含至少 1 条结论，格式正确",
                    "depends_on": [1],
                    "risk_level": "low",
                },
            ],
            "dependencies": {1: [0], 2: [1]},
            "risk_level": "medium",
            "estimated_total_steps": 3,
            "needs_clarification": [],
            "plan_summary": "三步审查 ReAct 循环实现：阅读代码、分析终止条件、输出报告",
            "allowed_repair_tools": ["read_file", "search_content", "write_file"],
        },
    },
    {
        "label": "多步重构",
        "task": "提取 context_assembler 模块并保持向后兼容",
        "output": {
            "steps": [
                {
                    "id": 0,
                    "description": "分析现有 agent.py 中消息拼装逻辑的边界",
                    "action": "read",
                    "tool": "read_file",
                    "params": {"file_path": "pyagent/core/agent.py"},
                    "expected_output": "明确需提取的代码范围（103-143 行）",
                    "acceptance_criteria": "已标注所有拼装相关行号，确认无遗漏",
                    "depends_on": [],
                    "risk_level": "low",
                },
                {
                    "id": 1,
                    "description": "创建 ReactContextAssembler 类，复制原始拼装逻辑",
                    "action": "write",
                    "tool": "write_file",
                    "params": {"file_path": "pyagent/harness/context/react_assembler.py"},
                    "expected_output": "行为与原 agent.py 完全一致的 ReactContextAssembler",
                    "acceptance_criteria": "所有 84 个已有测试通过，无行为变更",
                    "depends_on": [0],
                    "risk_level": "high",
                },
                {
                    "id": 2,
                    "description": "在 Agent.__init__ 中注入 context_assembler 参数",
                    "action": "write",
                    "tool": "write_file",
                    "params": {"file_path": "pyagent/core/agent.py"},
                    "expected_output": "Agent 支持可选的 context_assembler DI",
                    "acceptance_criteria": "无 assembler 时行为不变；传入 assembler 时走 DI 路径",
                    "depends_on": [1],
                    "risk_level": "high",
                },
                {
                    "id": 3,
                    "description": "运行全量测试确认无回归",
                    "action": "execute",
                    "tool": "execute_python",
                    "params": {"command": "pytest tests/ -q"},
                    "expected_output": "所有测试通过",
                    "acceptance_criteria": "167 个测试全部通过，0 失败",
                    "depends_on": [2],
                    "risk_level": "high",
                },
                {
                    "id": 4,
                    "description": "人工确认测试结果并审核代码变更",
                    "action": "review",
                    "tool": None,
                    "params": {},
                    "expected_output": "确认重构正确且无副作用",
                    "acceptance_criteria": "代码 diff 清晰、无冗余修改、向后兼容性确认",
                    "depends_on": [3],
                    "risk_level": "medium",
                },
            ],
            "dependencies": {1: [0], 2: [1], 3: [2], 4: [3]},
            "risk_level": "high",
            "estimated_total_steps": 5,
            "needs_clarification": [],
            "plan_summary": "五步安全重构：分析→提取 Assembler→注入 Agent→回归测试→人工审核",
            "allowed_repair_tools": ["read_file", "search_content", "write_file", "execute_python"],
        },
    },
]


class PEVRRunner:
    """
    PEVR 循环运行器（状态机驱动）。

    1.5.3 重构：用枚举驱动状态机替代手写 if-else 流转。
    状态转换条件、入口/出口动作、权限检查钩子声明式配置。
    状态机本身是纯逻辑，所有副作用（LLM调用、工具执行、Memory写入）
    通过回调函数注入，确保状态机可独立单元测试。

    使用方式：
        from pyagent.harness import create_agent_from_yaml
        from pyagent.harness.context import PEVRRunner, PEVRContextAssembler

        agent = create_agent_from_yaml("config.dev.yaml")
        assembler = PEVRContextAssembler(total_budget=64000)
        runner = PEVRRunner(agent, assembler)

        result = await runner.run(
            task="审查并优化 pyagent/core/agent.py",
            acceptance_criteria="所有测试通过，无新增警告",
        )
        print(result.success, result.verification)
    """

    def __init__(
        self,
        agent,
        assembler: Optional[ContextAssembler] = None,
        context_files: Optional[list[str]] = None,
        config_dir: str = ".",
        total_budget: int = DEFAULT_TOTAL_BUDGET,
        max_repairs: int = DEFAULT_MAX_REPAIRS,
        confirmation_callback=None,
        enable_checkpoint: bool = False,
        enable_observability: bool = True,
        audit_fallback_path: str = ".claude/audit_fallback.jsonl",
        governance=None,
    ):
        """
        Args:
            agent: Agent 实例（需支持 run_with_messages()）。
            assembler: ContextAssembler 实例。None 时自动创建 PEVRContextAssembler。
            context_files: context_files 路径列表（agent.md 等）。
            config_dir: 配置文件目录，用于解析相对路径。
            total_budget: 总 Token 预算。
            max_repairs: 最大修补次数。
            confirmation_callback: 计划级预审用户确认回调。
                签名: async (findings: list[dict]) -> bool
                True = 用户确认继续, False = 中止。
            enable_checkpoint: 是否启用断点持久化（默认 False）。
            enable_observability: 1.5.4 是否启用可观测性（默认 True）。
            audit_fallback_path: 1.5.4 P0 审计事件兜底文件路径。
            governance: 可选的 GovernanceWrapper 实例（v0.10.0）。
                        用于安全治理前置门控。None 时跳过（向后兼容）。
        """
        self.agent = agent
        self.assembler = assembler or PEVRContextAssembler(total_budget=total_budget)
        self.loader = ContextFileLoader(config_dir)
        self.context_files = context_files or []
        self.max_repairs = max_repairs
        self.total_budget = total_budget
        self.confirmation_callback = confirmation_callback
        self.enable_checkpoint = enable_checkpoint
        self.enable_observability = enable_observability
        self.audit_fallback_path = audit_fallback_path
        self._governance = governance

        # 运行时状态（每次 run() 重置）
        self._wm = WorkingMemory()
        self._total_tokens_used = 0
        self._sm: Optional[PEVRStateMachine] = None
        self._original_registry = None  # 保存原始 ToolRegistry 用于恢复
        self._obs: Optional[ObservabilityContext] = None  # 1.5.4
        self._circuit_breaker: Optional[CircuitBreaker] = None  # 1.5.4
        self._convergence_detector: Optional[ConvergenceDetector] = None  # 1.5.4
        self._repair_log: Optional[RepairLog] = None  # 1.5.4

    # ── 主入口 ────────────────────────────────────

    async def run(
        self,
        task: str,
        acceptance_criteria: str,
        session_id: Optional[str] = None,
    ) -> PEVRResult:
        """
        执行完整的 PEVR 循环（状态机驱动）。

        Args:
            task: 用户任务描述。
            acceptance_criteria: 验收标准原文。
            session_id: 可选会话 ID。

        Returns:
            PEVRResult: 包含成功状态、计划、产出、验收结论等。

        Raises:
            PlanValidationError: 规划校验在 MAX_PLAN_RETRIES 次重试后仍未通过。
        """
        # ── 1. 初始化运行时状态 ──
        self._wm = WorkingMemory()
        self._wm.metadata["task"] = task
        self._wm.metadata["session_id"] = session_id
        self._wm.acceptance_criteria = acceptance_criteria  # 1.5.3: 存入 WM 供 VERIFY 强制读取
        self._total_tokens_used = 0
        self._outputs: list[str] = []

        # v0.10.1: 重置安全治理状态（新 PEVR 会话开始）
        if hasattr(self.agent, 'reset_security_state'):
            self.agent.reset_security_state()

        # 1.5.4: 初始化修补熔断与收敛检测
        self._circuit_breaker = CircuitBreaker(max_repairs=self.max_repairs)
        self._convergence_detector = ConvergenceDetector(threshold=0.7)
        self._repair_log = RepairLog()

        # 1.5.4: 初始化观测上下文（独立于 WorkingMemory）
        if self.enable_observability:
            self._obs = ObservabilityContext(
                trace_id="",  # Plan 生成后回填
                fallback_path=self.audit_fallback_path,
            )
        else:
            self._obs = None

        # v0.10.0: 初始化安全治理上下文
        self._setup_governance_context(phase="planning")

        # 加载上下文规范文件
        loaded_files = self.loader.load(self.context_files)
        system_prompt = self.agent.config.system_prompt

        # ── 2. 构建状态机 ──
        sm = self._build_state_machine(task, acceptance_criteria, loaded_files,
                                       system_prompt, session_id)
        self._sm = sm
        self._original_registry = self.agent.tool_registry

        # ── 3. 启动状态机（PLANNING → EXECUTING → ... → COMPLETED/FAILED）──
        try:
            await sm.step(PEVREvent.PLAN_DONE)
        except PlanValidationError:
            logger.error("[PEVR] 规划校验失败，PEVR 循环中止")
            if not sm.is_terminal():
                try:
                    await sm.step(PEVREvent.PLAN_FAILED)
                except Exception:
                    pass
            raise
        except Exception as e:
            logger.error("[PEVR] 状态机异常: %s", e)
            if not sm.is_terminal():
                try:
                    await sm.step(PEVREvent.FATAL_ERROR)
                except Exception:
                    pass
        finally:
            # 1.5.4: 清理观测上下文
            if self._obs:
                try:
                    await self._obs.stop()
                except Exception:
                    pass

        # ── 4. 汇总结果 ──
        return self._build_result(sm)

    # ── 状态机构建 ────────────────────────────────────

    def _setup_governance_context(self, phase: str, state: str = ""):
        """v0.10.0: 更新 GovernanceWrapper 的执行上下文。

        PEVRRunner 在状态转换时调用此方法，将当前 phase/state/observability
        注入到 GovernanceWrapper 的活跃上下文中。Agent._execute_tool() 在每次
        工具调用前读取此上下文进行安全校验。

        Args:
            phase: PEVR 阶段（"planning"/"executing"/"verifying"/"repairing"）。
            state: 可选的状态机 state 字符串。
        """
        wrapper = getattr(self.agent, '_governance', None)
        if wrapper is None:
            return
        from .security_governance import ExecutionContext
        ctx = ExecutionContext(
            phase=phase,
            state=state or phase,
            observability=self._obs,
            trace_id=self._obs.trace_id if self._obs else "",
        )
        wrapper.set_active_context(ctx)

    def _build_state_machine(
        self,
        task: str,
        acceptance_criteria: str,
        loaded_files: list[str],
        system_prompt: str,
        session_id: Optional[str],
    ) -> PEVRStateMachine:
        """
        构建 PEVR 状态机并注册所有转换。

        所有副作用（LLM 调用、工具执行、Memory 写入）通过回调注入。
        回调通过 ctx["self"] 访问 PEVRRunner 实例。
        """
        sm = PEVRStateMachine(initial_state=PEVRState.PLANNING)

        # 注入运行时上下文（供回调使用）
        sm.context.update({
            "self": self,
            "task": task,
            "acceptance_criteria": acceptance_criteria,
            "loaded_files": loaded_files,
            "system_prompt": system_prompt,
            "session_id": session_id,
            "plan": None,           # ExecutionPlan 实例（PLANNING 完成后设置）
            "plan_text": "",        # Plan JSON 文本
            "steps": [],            # 步骤列表（dict）
            "outputs": [],          # 执行产出
            "verification": "",     # 验收结论
            "repair_count": 0,
            "current_step_index": 0,
            "failure_reason": "",   # 当前失败原因
            # 1.5.4: 观测与修补上下文
            "observability": self._obs,
            "circuit_breaker": self._circuit_breaker,
            "convergence_detector": self._convergence_detector,
            "repair_log": self._repair_log,
            "audit_sink": [],       # 当前阶段的审计事件收集列表
        })

        # ── 转换表 ──
        # 每条转换: from_state, event, to_state, [guards], [before], [after]

        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            after=[_after_planning],
        )
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_FAILED,
            to_state=PEVRState.FAILED,
        )

        sm.add_transition(
            PEVRState.EXECUTING, PEVREvent.ALL_STEPS_DONE,
            to_state=PEVRState.VERIFYING,
            after=[_after_execute_done],
        )
        sm.add_transition(
            PEVRState.EXECUTING, PEVREvent.PERMISSION_VIOLATION,
            to_state=PEVRState.REPAIRING,
            before=[_before_enter_repair],
            after=[_after_enter_repair],
        )
        sm.add_transition(
            PEVRState.EXECUTING, PEVREvent.FATAL_ERROR,
            to_state=PEVRState.FAILED,
        )

        sm.add_transition(
            PEVRState.VERIFYING, PEVREvent.VERIFY_PASSED,
            to_state=PEVRState.COMPLETED,
        )
        sm.add_transition(
            PEVRState.VERIFYING, PEVREvent.VERIFY_FAILED,
            to_state=PEVRState.REPAIRING,
            guards=[_guard_can_repair],
            before=[_before_enter_repair],
            after=[_after_enter_repair],
        )
        sm.add_transition(
            PEVRState.VERIFYING, PEVREvent.FATAL_ERROR,
            to_state=PEVRState.FAILED,
        )

        sm.add_transition(
            PEVRState.REPAIRING, PEVREvent.REPAIR_DONE,
            to_state=PEVRState.VERIFYING,
            before=[_before_exit_repair],
            after=[_after_repair_done],
        )
        sm.add_transition(
            PEVRState.REPAIRING, PEVREvent.REPAIR_FAILED,
            to_state=PEVRState.VERIFYING,
            before=[_before_exit_repair],
            guards=[_guard_can_retry_verify],
        )

        return sm

    # ── 结果构建 ────────────────────────────────────

    def _build_result(self, sm: PEVRStateMachine) -> PEVRResult:
        """从状态机构建 PEVRResult。"""
        ctx = sm.context
        plan = ctx.get("plan")
        plan_text = ctx.get("plan_text", "")
        if plan and hasattr(plan, 'model_dump_json'):
            plan_text = plan.model_dump_json(indent=2)

        return PEVRResult(
            success=(sm.current_state == PEVRState.COMPLETED),
            plan=plan_text,
            outputs=ctx.get("outputs", []),
            verification=ctx.get("verification", ""),
            repair_count=ctx.get("repair_count", 0),
            total_tokens_used=self._total_tokens_used,
            working_memory=self._wm,
        )

    # ── 规划校验子循环（不变，由 after 回调调用）───────

    async def _generate_and_validate_plan(
        self,
        task: str,
        criteria: str,
        files: list[str],
        system: str,
        few_shot_examples: Optional[list[dict]] = None,
    ) -> ExecutionPlan:
        """
        规划生成 + 校验修正闭环。

        流程：
            1. 通过 ContextAssembler 组装 Plan 阶段 Prompt（含 Few-shot）
            2. LLM 生成 → 解析 JSON → Pydantic 结构校验 → 业务规则校验
            3. 校验失败时：将错误信息作为新 UserMessage 回传模型修正
            4. 最多重试 MAX_PLAN_RETRIES 次
            5. 通过后返回冻结的 ExecutionPlan
        """
        examples = few_shot_examples if few_shot_examples is not None else FEW_SHOT_EXAMPLES
        validator = PlanValidator()
        all_failures: list[dict] = []

        request = ContextRequest(
            phase=PEVRPhase.PLAN,
            system_prompt=system,
            context_files=files,
            plan=task,
            working_memory=self._wm,
            acceptance_criteria=criteria,
            few_shot_examples=examples,
        )
        result = await self.assembler.assemble(request)
        self._total_tokens_used += result.total_tokens
        messages = result.messages

        for attempt in range(MAX_PLAN_RETRIES + 1):
            logger.info("[PEVR:PLAN] 生成尝试 %d/%d", attempt + 1, MAX_PLAN_RETRIES + 1)

            response: AssistantMessage = await self.agent.llm.generate(
                messages, tools=[]
            )
            raw_content = response.content or ""

            raw_data = _extract_json(raw_content)
            if raw_data is None:
                failure = {
                    "rule": "json_parse",
                    "input_snippet": raw_content[:200],
                    "error_type": "schema",
                    "detail": "LLM 输出中未找到可解析的 JSON。请确保仅输出 JSON。",
                }
                all_failures.append(failure)
                messages = _append_correction(
                    messages, response, failure, attempt, MAX_PLAN_RETRIES
                )
                continue

            failures = validator.validate(raw_data)
            if not failures:
                try:
                    plan = ExecutionPlan.model_validate(raw_data)
                    logger.info(
                        "[PEVR:PLAN] 校验通过（尝试 %d/%d），%d 个步骤",
                        attempt + 1, MAX_PLAN_RETRIES + 1, len(plan.steps),
                    )
                    return plan
                except Exception as e:
                    failures = [{
                        "rule": "pydantic_construct",
                        "input_snippet": str(raw_data)[:200],
                        "error_type": "schema",
                        "detail": str(e),
                    }]

            all_failures.extend(failures)
            logger.warning(
                "[PEVR:PLAN] 校验失败（尝试 %d/%d）：%d 条问题",
                attempt + 1, MAX_PLAN_RETRIES + 1, len(failures),
            )
            for f in failures:
                logger.debug("  - [%s] %s", f["rule"], f["detail"][:100])

            if attempt < MAX_PLAN_RETRIES:
                failure_feedback = _format_failure_feedback(failures)
                messages = _append_correction(
                    messages, response, failure_feedback, attempt, MAX_PLAN_RETRIES,
                )

        raise PlanValidationError(
            f"规划校验在 {MAX_PLAN_RETRIES + 1} 次尝试后仍未通过",
            validation_failures=all_failures,
            attempt_number=MAX_PLAN_RETRIES,
        )

    # ── 各阶段实现（由状态机回调调用）─────────────

    async def _run_execute(
        self, step: dict, files: list[str], system: str,
        step_index: int, total_steps: int,
        session_id: Optional[str] = None,
    ) -> str:
        """Execute 阶段：通过 ReAct 循环执行一个步骤。"""
        step_text = (
            f"步骤 {step_index}/{total_steps}: {step.get('description', str(step))}\n"
            f"预期产出: {step.get('expected_output', '完成此步骤')}"
        )
        request = ContextRequest(
            phase=PEVRPhase.EXECUTE,
            system_prompt=system,
            context_files=files,
            plan=step_text,
            working_memory=self._wm,
        )
        result = await self.assembler.assemble(request)
        self._total_tokens_used += result.total_tokens

        if hasattr(self.agent, 'run_with_messages'):
            reply = await self.agent.run_with_messages(
                messages=result.messages,
                session_id=session_id,
            )
        else:
            response = await self.agent.llm.generate(
                result.messages, tools=self.agent.tool_registry.get_all_schemas(),
            )
            reply = response.content or ""

        return reply

    async def _run_verify(
        self, criteria: str, plan: str, outputs: list[str],
        files: list[str], system: str,
    ) -> str:
        """Verify 阶段：LLM 对照验收标准判断是否通过。"""
        # v0.10.0: 更新安全治理阶段上下文 → VERIFYING
        self._setup_governance_context("verifying")

        self._wm.artifacts["outputs"] = "\n---\n".join(outputs)
        request = ContextRequest(
            phase=PEVRPhase.VERIFY,
            system_prompt=system,
            context_files=files,
            plan=plan,
            working_memory=self._wm,
            acceptance_criteria=criteria,  # 1.5.3: Assembler 会强制从 WM 读取
        )
        result = await self.assembler.assemble(request)
        self._total_tokens_used += result.total_tokens

        response = await self.agent.llm.generate(result.messages, tools=[])
        return response.content or ""

    async def _run_repair(
        self, failure_summary: str, files: list[str], system: str,
        session_id: Optional[str] = None,
    ) -> str:
        """Repair 阶段：基于失败原因修补。"""
        last_failure = self._wm.last_failure()
        step_text = (
            f"修复: {last_failure.step if last_failure else '上一失败步骤'}"
        )
        request = ContextRequest(
            phase=PEVRPhase.REPAIR,
            system_prompt=system,
            context_files=files,
            plan=step_text,
            working_memory=self._wm,
            failure_summary=failure_summary,
        )
        result = await self.assembler.assemble(request)
        self._total_tokens_used += result.total_tokens

        if hasattr(self.agent, 'run_with_messages'):
            reply = await self.agent.run_with_messages(
                messages=result.messages,
                session_id=session_id,
            )
        else:
            response = await self.agent.llm.generate(
                result.messages, tools=self.agent.tool_registry.get_all_schemas(),
            )
            reply = response.content or ""

        return reply

    # ── 检查点 ────────────────────────────────────

    async def _save_checkpoint(self):
        """保存当前状态检查点。"""
        if not self.enable_checkpoint or not self._sm:
            return
        sm = self._sm
        ctx = sm.context
        checkpoint = PEVRCheckpoint(
            state=sm.current_state.value,
            plan_ref_id=ctx.get("session_id", ""),
            current_step_index=ctx.get("current_step_index", 0),
            repair_count=ctx.get("repair_count", 0),
            total_steps=len(ctx.get("steps", [])),
            trace_id=ctx.get("trace_id", ""),  # 1.5.4
            metadata={
                "task": ctx.get("task", ""),
                "acceptance_criteria": ctx.get("acceptance_criteria", ""),
            },
        )
        await save_checkpoint(
            checkpoint,
            self.agent.memory if hasattr(self.agent, 'memory') else None,
            ctx.get("session_id"),
        )

    async def restore_from_checkpoint(self, session_id: str) -> Optional[PEVRResult]:
        """
        从检查点恢复中断的任务。

        Args:
            session_id: 会话 ID。

        Returns:
            PEVRResult 或 None（无可恢复的检查点）。
        """
        from .checkpoint import load_checkpoint

        memory = self.agent.memory if hasattr(self.agent, 'memory') else None
        checkpoint = await load_checkpoint(memory, session_id)
        if checkpoint is None:
            logger.info("[PEVR] 未找到检查点（session=%s），从头开始", session_id)
            return None

        logger.info(
            "[PEVR] 从检查点恢复: state=%s step=%d/%d repair=%d",
            checkpoint.state, checkpoint.current_step_index,
            checkpoint.total_steps, checkpoint.repair_count,
        )

        # 重建 WorkingMemory 状态
        self._wm.repair_count = checkpoint.repair_count
        self._wm.current_step_index = checkpoint.current_step_index
        self._wm.metadata["session_id"] = session_id
        self._wm.acceptance_criteria = checkpoint.metadata.get("acceptance_criteria", "")

        # 从指定步骤继续执行
        task = checkpoint.metadata.get("task", "")
        acceptance_criteria = checkpoint.metadata.get("acceptance_criteria", "")

        # 加载文件并重建状态机
        loaded_files = self.loader.load(self.context_files)
        system_prompt = self.agent.config.system_prompt

        sm = self._build_state_machine(task, acceptance_criteria, loaded_files,
                                       system_prompt, session_id)
        sm.context["repair_count"] = checkpoint.repair_count
        sm.context["current_step_index"] = checkpoint.current_step_index

        # Jump to recovery state
        recovery_state = PEVRState(checkpoint.state)
        sm.current_state = recovery_state

        self._sm = sm
        self._original_registry = self.agent.tool_registry

        # 根据恢复状态继续执行
        if recovery_state == PEVRState.EXECUTING:
            # 从当前步骤继续执行
            await _continue_executing(sm.context)
        elif recovery_state == PEVRState.VERIFYING:
            await _after_execute_done(sm.context)
        elif recovery_state == PEVRState.REPAIRING:
            await _after_enter_repair(sm.context)
        elif recovery_state == PEVRState.COMPLETED or recovery_state == PEVRState.FAILED:
            return self._build_result(sm)

        # 清理检查点（任务完成）
        if sm.is_terminal():
            await delete_checkpoint(memory, session_id)

        return self._build_result(sm)


# ── 状态机回调函数（模块级，通过 ctx["self"] 访问 runner）──

async def _plan_level_audit(ctx: dict) -> bool:
    """
    计划级预审守卫：扫描高危工具组合，触发用户确认。

    唯一可信源：ctx["plan"]（冻结的 ExecutionPlan）。
    """
    runner = ctx["self"]
    plan = ctx.get("plan")
    if plan is None:
        return True  # 无 Plan 则放行（后续阶段会拦截）

    findings = PermissionGate.plan_level_audit(plan, audit_sink=ctx.get("audit_sink"))
    if not findings:
        return True  # 无高危组合，放行

    logger.warning("[PEVR:AUDIT] 计划级预审发现 %d 条高危项", len(findings))
    for f in findings:
        logger.warning("  - %s (步骤: %s)", f["reason"], f["steps"])

    if runner.confirmation_callback:
        # 异步回调获取用户决策
        import asyncio
        if asyncio.iscoroutinefunction(runner.confirmation_callback):
            confirmed = await runner.confirmation_callback(findings)
        else:
            confirmed = runner.confirmation_callback(findings)
        if not confirmed:
            logger.warning("[PEVR:AUDIT] 用户拒绝高危操作，转换中止")
            return False
    else:
        # 无回调 → 自动拒绝高危组合（安全优先）
        logger.warning(
            "[PEVR:AUDIT] 未配置 confirmation_callback，高危组合自动拒绝"
        )
        return False

    return True


async def _after_planning(ctx: dict):
    """
    PLANNING → EXECUTING 转换的入口动作。

    1. 调用 _generate_and_validate_plan 生成计划
    2. 冻结计划到 WorkingMemory
    3. 执行计划级预审
    4. 成功 → 保持 EXECUTING 状态，开始执行步骤
    5. 失败 → 转入 FAILED
    """
    runner = ctx["self"]
    sm = runner._sm
    task = ctx["task"]
    acceptance_criteria = ctx["acceptance_criteria"]
    loaded_files = ctx["loaded_files"]
    system_prompt = ctx["system_prompt"]
    session_id = ctx.get("session_id")

    try:
        # 1. 生成并校验计划
        execution_plan = await runner._generate_and_validate_plan(
            task, acceptance_criteria, loaded_files, system_prompt,
        )
    except PlanValidationError:
        logger.error("[PEVR] 规划校验失败，转入 FAILED")
        await sm.step(PEVREvent.PLAN_FAILED)
        return

    # 2. 冻结计划快照 → WorkingMemory
    runner._wm.set_plan(execution_plan)
    runner._wm.add_artifact("plan.md", execution_plan.model_dump_json(indent=2))

    steps = [s.model_dump() for s in execution_plan.steps]
    plan_text = execution_plan.model_dump_json(indent=2)
    logger.info("[PEVR] 计划包含 %d 个步骤", len(steps))
    runner._wm.metadata["total_steps"] = len(steps)

    # 1.5.4: 生成并回填 trace_id
    trace_id = TraceContext.generate()
    # ExecutionPlan 是 frozen，需要通过 object.__setattr__ 绕过
    object.__setattr__(execution_plan, 'trace_id', trace_id)
    ctx["trace_id"] = trace_id

    # 1.5.4: 启动观测上下文
    if runner._obs:
        runner._obs.trace_id = trace_id
        await runner._obs.start()
        logger.info("[PEVR:OBS] 观测上下文已启动，trace_id=%s", trace_id[:8])

    # 3. 更新上下文
    ctx["plan"] = execution_plan
    ctx["plan_text"] = plan_text
    ctx["steps"] = steps

    # 4. 计划级预审（1.5.4: 传入 audit_sink）
    audit_sink = ctx.get("audit_sink", [])
    audit_ok = await _plan_level_audit(ctx)
    # 将计划级审计事件推送到观测上下文
    if runner._obs and audit_sink:
        for event in audit_sink:
            runner._obs.emit(event)
        audit_sink.clear()

    if not audit_ok:
        logger.error("[PEVR:AUDIT] 计划级预审未通过，转入 FAILED")
        await sm.step(PEVREvent.FATAL_ERROR)
        return

    # 5. 保存检查点
    await runner._save_checkpoint()

    # v0.10.0: 更新安全治理阶段上下文 → EXECUTING
    runner._setup_governance_context("executing")

    # 6. 进入执行阶段——逐个执行步骤
    await _continue_executing(ctx)


async def _continue_executing(ctx: dict):
    """
    EXECUTING 状态的核心逻辑：逐个执行 Plan 中的步骤。

    每步执行前进行步骤级权限检查，参数越界则转入 REPAIRING。
    """
    runner = ctx["self"]
    sm = runner._sm
    steps = ctx.get("steps", [])
    loaded_files = ctx["loaded_files"]
    system_prompt = ctx["system_prompt"]
    session_id = ctx.get("session_id")
    plan = ctx.get("plan")

    outputs = ctx.get("outputs", [])
    start_index = ctx.get("current_step_index", 0)

    for i in range(start_index, len(steps)):
        step_dict = steps[i]
        step_index = i + 1  # 1-based for display
        logger.info("[PEVR] Phase: EXECUTE step %d/%d", step_index, len(steps))
        runner._wm.metadata["current_step"] = step_index
        runner._wm.current_step_index = step_index
        ctx["current_step_index"] = step_index

        # 1.5.3: 提取当前步骤的 Plan 片段为不可变对象（用于权限检查）
        if plan and hasattr(plan, 'steps') and i < len(plan.steps):
            frozen_step = plan.steps[i]
        else:
            frozen_step = None

        # 执行步骤
        step_result = await runner._run_execute(
            step_dict, loaded_files, system_prompt,
            step_index=step_index, total_steps=len(steps),
            session_id=session_id,
        )

        # 1.5.3: 步骤级权限检查（在记录结果前）
        # 注意：此检查需要在实际工具调用参数上执行。
        # 由于 _run_execute 内部通过 agent.run_with_messages() 执行，
        # 工具调用由 Agent 内部处理，此处的检查主要用于：
        # - 步骤完成后验证 Step 声明的一致性
        # - 为 REPAIRING 阶段提供上下文
        # 实际的实时权限拦截由 Agent._react_loop 中的工具执行钩子完成
        # （见 Agent._execute_tool → step_level_check）

        # 记录结果
        runner._wm.add_step_result(
            step=step_dict.get("description", str(step_dict)),
            result=step_result,
            status="success",
        )
        outputs.append(step_result)
        ctx["outputs"] = outputs

        # 保存检查点
        await runner._save_checkpoint()

    # 所有步骤执行完成 → VERIFYING
    logger.info("[PEVR] 所有步骤执行完成，转入 VERIFYING")
    await sm.step(PEVREvent.ALL_STEPS_DONE)


async def _after_execute_done(ctx: dict):
    """
    EXECUTING → VERIFYING 转换的入口动作。

    运行验收检查，根据结果转入 COMPLETED 或 REPAIRING。
    """
    runner = ctx["self"]
    sm = runner._sm
    acceptance_criteria = ctx["acceptance_criteria"]
    plan_text = ctx.get("plan_text", "")
    outputs = ctx.get("outputs", [])
    loaded_files = ctx["loaded_files"]
    system_prompt = ctx["system_prompt"]

    logger.info("[PEVR] Phase: VERIFY")
    verification = await runner._run_verify(
        acceptance_criteria, plan_text, outputs,
        loaded_files, system_prompt,
    )
    ctx["verification"] = verification
    passed = _parse_verification(verification)

    if passed:
        logger.info("[PEVR] 验收通过，转入 COMPLETED")
        await sm.step(PEVREvent.VERIFY_PASSED)
    else:
        repair_count = ctx.get("repair_count", 0)
        if repair_count < runner.max_repairs:
            logger.info("[PEVR] 验收未通过，转入 REPAIRING（%d/%d）",
                        repair_count + 1, runner.max_repairs)
            await sm.step(PEVREvent.VERIFY_FAILED)
        else:
            logger.error("[PEVR] 修补次数耗尽（%d/%d），转入 FAILED",
                         repair_count, runner.max_repairs)
            await sm.step(PEVREvent.FATAL_ERROR)

    await runner._save_checkpoint()


def _guard_can_repair(ctx: dict) -> bool:
    """守卫：修补次数是否在限制内 + 熔断检测。"""
    runner = ctx["self"]
    repair_count = ctx.get("repair_count", 0)

    # 1.5.4: 检查 CircuitBreaker
    cb = ctx.get("circuit_breaker")
    if cb and cb.is_tripped():
        logger.warning("[PEVR:GUARD] CircuitBreaker 已熔断，拒绝再修补")
        return False

    # 1.5.4: 检查收敛检测是否已标记原地打转
    if runner._wm.metadata.get("convergence_detected"):
        logger.warning("[PEVR:GUARD] 收敛检测已标记原地打转，拒绝再修补")
        return False

    return repair_count < runner.max_repairs


def _guard_can_retry_verify(ctx: dict) -> bool:
    """守卫：REPAIR_FAILED 后是否还能再试。"""
    runner = ctx["self"]
    repair_count = ctx.get("repair_count", 0)
    # REPAIR_FAILED 也消耗一次修补次数
    can_retry = repair_count < runner.max_repairs
    if not can_retry:
        logger.error("[PEVR] 修补失败且次数耗尽")
    return can_retry


async def _before_enter_repair(ctx: dict):
    """
    进入 REPAIRING 前的出口动作。

    1. 递增修补计数 + 熔断检测
    2. 收敛检测（原地打转）
    3. 创建 RepairContext（隔离上下文）
    4. 保存失败上下文
    5. 创建 ScopedToolRegistry 并 swap Agent 的 tool_registry
    """
    runner = ctx["self"]

    # v0.10.0: 更新安全治理阶段上下文 → REPAIRING
    runner._setup_governance_context("repairing")

    repair_count = ctx.get("repair_count", 0) + 1
    ctx["repair_count"] = repair_count
    runner._wm.repair_count = repair_count

    # 1.5.4: 熔断检测
    cb: CircuitBreaker = ctx.get("circuit_breaker")
    if cb and cb.is_tripped():
        logger.error("[PEVR:REPAIR] 熔断已触发（%d/%d），无法再修补",
                     cb.attempt_count, cb.max_repairs)
        # 不继续修补，由守卫 _guard_can_repair 拦截
        return

    if cb:
        cb.record_attempt(
            detail=f"修补尝试 {repair_count}",
            failure_reason=ctx.get("failure_reason", ""),
        )

    # 1.5.4: 收敛检测（在进入修补前检查上一轮 changes_made）
    cd: ConvergenceDetector = ctx.get("convergence_detector")
    previous_changes = ctx.get("_last_changes_made", "")
    if cd and previous_changes and cd.check_convergence(previous_changes):
        logger.warning(
            "[PEVR:REPAIR] 收敛检测: 上一轮修补未产生实质变化（相似度=%.3f），疑似原地打转",
            cd.last_score,
        )
        # 提前熔断：强制消耗剩余修补次数
        if cb:
            while cb.attempt_count < cb.max_repairs:
                cb.record_attempt(detail="收敛检测: 原地打转", failure_reason="")
        runner._wm.metadata["convergence_detected"] = True
        runner._wm.metadata["convergence_score"] = cd.last_score

    # 保存失败原因
    last_failure = runner._wm.last_failure()
    verification = ctx.get("verification", "")
    failure_summary = (
        f"验收未通过: {verification}\n"
        f"最后失败步骤: {last_failure.step if last_failure else '未知'}"
    )
    ctx["failure_reason"] = failure_summary

    # 1.5.3: 创建 ScopedToolRegistry（修补特权降级）
    plan = ctx.get("plan")
    whitelist = PermissionGate.get_repair_whitelist(plan)
    logger.info("[PEVR:REPAIR] 工具白名单: %s", sorted(whitelist))
    scoped = ScopedToolRegistry(runner._original_registry, whitelist)
    runner.agent.tool_registry = scoped

    # 1.5.4: 创建 RepairContext（隔离上下文，不注入完整 History）
    repair_ctx = RepairContext(
        failed_step_description=(
            last_failure.step if last_failure else "未知步骤"
        ),
        failure_reason=failure_summary,
        acceptance_criteria=ctx.get("acceptance_criteria", ""),
        available_tools=sorted(whitelist),
        previous_repair_summary=runner._repair_log.last_summary()
            if runner._repair_log else "",
        repair_attempt=repair_count,
        max_repairs=runner.max_repairs,
    )
    # 通过 metadata 注入到 WorkingMemory（assembler 从中读取）
    runner._wm.metadata["repair_context"] = repair_ctx

    logger.info("[PEVR] Phase: REPAIR (attempt %d/%d)",
                repair_count, runner.max_repairs)


async def _after_enter_repair(ctx: dict):
    """
    REPAIRING 状态的入口动作：执行修补。

    1.5.4 增强：
    - 强制 JSON 输出解析（changes_made + fixed）
    - 收敛检测（连续相同 changes_made → REPAIR_FAILED）
    - RepairLog 记录
    - 修补完成后自动转入 VERIFYING 重新验收
    """
    runner = ctx["self"]
    sm = runner._sm
    failure_reason = ctx.get("failure_reason", "")
    loaded_files = ctx["loaded_files"]
    system_prompt = ctx["system_prompt"]
    session_id = ctx.get("session_id")
    repair_count = ctx.get("repair_count", 0)

    # 执行修补
    repair_result = await runner._run_repair(
        failure_reason, loaded_files, system_prompt,
        session_id=session_id,
    )

    # 1.5.4: 解析 JSON 输出，提取 changes_made
    repair_json = _parse_repair_json(repair_result)
    if repair_json is None:
        # JSON 解析失败 → REPAIR_FAILED
        logger.warning("[PEVR:REPAIR] 修补输出 JSON 解析失败，转入 REPAIR_FAILED")
        runner._wm.metadata["repair_parse_error"] = repair_result[:200]
        await sm.step(PEVREvent.REPAIR_FAILED)
        return

    changes_made = repair_json.get("changes_made", "")
    fixed = repair_json.get("fixed", False)
    output_text = repair_json.get("output", repair_result)

    # 1.5.4: 收敛检测记录
    cd: ConvergenceDetector = ctx.get("convergence_detector")
    ctx["_last_changes_made"] = changes_made

    convergence_score = cd.last_score if cd else 0.0

    # 1.5.4: RepairLog 记录
    rl: RepairLog = ctx.get("repair_log")
    if rl:
        rl.record(
            attempt=repair_count,
            changes_made=changes_made,
            fixed=fixed,
            convergence_score=convergence_score,
        )

    # 1.5.4: 生成修补审计事件
    obs = ctx.get("observability")
    if obs and rl:
        trace_id = ctx.get("trace_id", "")
        for event in rl.to_audit_events(trace_id):
            obs.emit(event)

    outputs = ctx.get("outputs", [])
    outputs.append(output_text)
    ctx["outputs"] = outputs

    # 1.5.4: 若未修复且熔断已触发 → 直接 FAILED
    cb: CircuitBreaker = ctx.get("circuit_breaker")
    if not fixed and cb and cb.is_tripped():
        logger.error("[PEVR:REPAIR] 未修复且熔断已触发，转入 FAILED")
        await sm.step(PEVREvent.FATAL_ERROR)
        return

    # 修补完成 → 回到 VERIFYING
    await sm.step(PEVREvent.REPAIR_DONE)


async def _before_exit_repair(ctx: dict):
    """
    离开 REPAIRING 前的出口动作：恢复原始 ToolRegistry。

    确保 ScopedToolRegistry 自动销毁，原注册表零修改。
    """
    runner = ctx["self"]
    if runner._original_registry is not None:
        runner.agent.tool_registry = runner._original_registry
        logger.debug("[PEVR:REPAIR] 已恢复原始 ToolRegistry")

    # v0.10.0: 更新安全治理阶段上下文 → VERIFYING
    runner._setup_governance_context("verifying")


async def _after_repair_done(ctx: dict):
    """
    REPAIRING → VERIFYING 转换的入口动作。

    重新运行验收检查。
    """
    await _after_execute_done(ctx)


# ── 规划校验辅助函数（不变）──────────────────────

def _extract_json(text: str) -> Optional[dict]:
    """从 LLM 输出中提取 JSON dict。"""
    md_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def _format_failure_feedback(failures: list[dict]) -> str:
    """将校验失败列表格式化为 LLM 可理解的修正请求。"""
    parts = [
        "你的上一个输出未通过以下校验，请修正后重新输出：",
        "",
    ]
    for i, f in enumerate(failures, 1):
        parts.append(f"{i}. [{f['rule']}] ({f['error_type']}) {f['detail']}")
        if f.get("input_snippet"):
            parts.append(f"   触发片段: {f['input_snippet'][:150]}")

    parts.append("")
    parts.append("请严格遵循 ExecutionPlan JSON Schema 重新输出。注意：")
    parts.append("- 所有 Step 字段 (description, action, expected_output, acceptance_criteria) 必须非空")
    parts.append("- action 限值: read | write | execute | review | ask")
    parts.append("- risk_level 限值: low | medium | high")
    parts.append("- 不要输出任何解释文字，仅输出 JSON")
    return "\n".join(parts)


def _append_correction(
    messages: list,
    last_response,
    failure_info,
    attempt: int,
    max_attempts: int,
) -> list:
    """向消息列表追加修正请求。"""
    new_messages = list(messages)
    if hasattr(last_response, 'content'):
        new_messages.append(last_response)

    if isinstance(failure_info, dict):
        feedback = _format_failure_feedback([failure_info])
    else:
        feedback = str(failure_info)

    correction_msg = (
        f"## 校验失败（尝试 {attempt + 1}/{max_attempts + 1}）\n\n"
        f"{feedback}"
    )
    new_messages.append(UserMessage(content=correction_msg))
    return new_messages


# ── 辅助解析函数 ──────────────────────────────────

def _parse_steps(plan: str) -> list[dict]:
    """从 LLM 输出的计划中提取步骤列表。"""
    json_match = re.search(r'\{[\s\S]*"steps"[\s\S]*\}', plan)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("steps", [])
        except json.JSONDecodeError:
            pass

    steps = []
    for line in plan.split("\n"):
        m = re.match(r'\s*(\d+)[\.\)]\s*(.+)', line)
        if m:
            steps.append({"id": int(m.group(1)), "description": m.group(2).strip()})
    return steps


def _parse_repair_json(raw: str) -> dict | None:
    """
    从修补输出中提取 JSON。

    1.5.4: 容错解析 — 优先提取 ```json 块，回退到裸 JSON 匹配。
    解析失败返回 None（由调用方转入 REPAIR_FAILED）。

    Args:
        raw: LLM 原始输出。

    Returns:
        dict 含 fixed / changes_made / output，或 None（解析失败）。
    """
    if not raw:
        return None

    # 1. 优先提取 ```json 块
    md_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 2. 回退：匹配裸 JSON 对象
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    logger.debug("[PEVR:REPAIR] JSON 解析失败，原始输出前 200 字符: %s", raw[:200])
    return None


def _parse_verification(verification: str) -> bool:
    """从验收输出中提取通过/失败。"""
    json_match = re.search(r'\{[\s\S]*"passed"[\s\S]*\}', verification)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return data.get("passed", False)
        except json.JSONDecodeError:
            pass

    text = verification.lower()
    failed_markers = ["未通过", "不通过", "失败", "failed", "not passed"]
    passed_markers = ["通过", "passed", "成功", "success"]

    for marker in failed_markers:
        if marker in text:
            return False
    for marker in passed_markers:
        if marker in text:
            return True

    return True
