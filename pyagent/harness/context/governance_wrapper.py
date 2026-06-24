"""
治理包装器 — Executor 层透明拦截（方案 B + 优化 1/4）。

GovernanceWrapper 包裹在 Agent._execute_tool() 外部，对每个工具调用
执行 Layer 1（前置门控）+ Layer 3（熔断 + 性能监控）检查。

设计原则：
    - 接收 ExecutionContext（frozen），不依赖 Agent 可变属性
    - 安全拦截返回 ToolMessage（非异常），LLM 可自我修正
    - 精确性能埋点：仅度量 SecurityGovernance.check() 耗时
    - 不捕获工具执行异常（由 Agent._execute_tool 统一处理）

使用方式：
    # 创建
    wrapper = GovernanceWrapper(governance, circuit_breaker)

    # 在 Agent._execute_tool 中
    ctx = ExecutionContext(phase="executing", trace_id="...")
    tool_msg = await wrapper.execute_tool(tool_call, tool_registry, ctx)
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...core.message import ToolCall, ToolMessage
    from .security_governance import ExecutionContext, SecurityGovernance
    from .security_circuit_breaker import SecurityCircuitBreaker

logger = logging.getLogger("pyagent.security.governance_wrapper")


class GovernanceWrapper:
    """工具执行安全治理包装器。

    在每个工具调用前执行安全检查，在拦截时返回 ToolMessage，
    正常时委托给 ToolRegistry 执行。
    """

    def __init__(
        self,
        governance: "SecurityGovernance",
        circuit_breaker: "SecurityCircuitBreaker",
        enable_perf_monitor: bool = True,
    ):
        """
        Args:
            governance: SecurityGovernance 实例（Layer 1 编排器）。
            circuit_breaker: SecurityCircuitBreaker 实例（Layer 3 熔断）。
            enable_perf_monitor: 是否启用性能监控与自动降级。
        """
        self._gov = governance
        self._cb = circuit_breaker
        self._enable_perf = enable_perf_monitor

        # 活跃执行上下文 — PEVRRunner 在状态转换时设置，Agent._execute_tool 读取
        from .security_governance import ExecutionContext
        self._active_ctx: ExecutionContext = ExecutionContext()

    # ── 主入口 ────────────────────────────────────

    async def execute_tool(
        self,
        tool_call: "ToolCall",
        tool_registry: Any,
        ctx: "ExecutionContext",
    ) -> "ToolMessage":
        """执行工具调用，带完整安全治理检查。

        流程：
            1. 熔断器检查（Layer 3）— OPEN 状态直接返回熔断消息
            2. 前置门控（Layer 1）— SecurityGovernance.check()
            3. 性能监控（Layer 3）— 超阈值自动降级
            4. BLOCK → 记录熔断器 + 返回 ToolMessage
            5. ALLOW → 委托 ToolRegistry 执行 + 记录风险上下文

        Args:
            tool_call: LLM 返回的 ToolCall。
            tool_registry: ToolRegistry 或 ScopedToolRegistry 实例。
            ctx: 执行上下文（phase, state, observability, trace_id）。

        Returns:
            ToolMessage — 工具执行结果或安全拦截消息。
        """
        from ...core.message import ToolMessage

        tool_name = tool_call.function_name
        tool_params = tool_call.arguments
        call_id = tool_call.id

        # ── 0. 熔断器检查（Layer 3）──
        if self._cb.is_tripped():
            logger.warning(
                "[GovernanceWrapper] 熔断器已触发 (%s)，拒绝工具 '%s'",
                self._cb.state.value, tool_name,
            )
            return ToolMessage(
                content=(
                    f"[安全熔断] 安全事件熔断器已触发（{self._cb.tripped_count} 次）。\n"
                    f"工具 '{tool_name}' 被阻止。\n"
                    f"当前冷却时间: {self._cb.current_cooldown:.0f}s\n"
                    f"请人工介入检查安全事件日志，或等待冷却后自动恢复。"
                ),
                tool_call_id=call_id,
                name=tool_name,
            )

        # ── 1. 前置门控（Layer 1）+ 精确性能埋点（优化 4）──
        t0 = time.monotonic()
        decision = self._gov.check(tool_name, tool_params, ctx)
        elapsed_ms = (time.monotonic() - t0) * 1000

        # 将耗时记录到决策中
        decision = decision.__class__(
            allowed=decision.allowed,
            rule_id=decision.rule_id,
            risk_score=decision.risk_score,
            reason=decision.reason,
            blocked_message=decision.blocked_message,
            performance_ms=elapsed_ms,
        )

        # ── 2. 性能降级检查（优化 4）──
        if self._enable_perf and elapsed_ms > self._gov.degrade_threshold_ms:
            self._gov.disable_combo_detection()
            logger.warning(
                "[GovernanceWrapper] 安全校验耗时 %.1fms > %.0fms 阈值，"
                "已触发性能降级（combo 检测已禁用）",
                elapsed_ms, self._gov.degrade_threshold_ms,
            )

        # ── 3. BLOCK 处理 ──
        if not decision.allowed:
            self._cb.record_block(
                tool_name=tool_name,
                rule_id=decision.rule_id,
            )
            logger.info(
                "[GovernanceWrapper] BLOCK tool=%s rule=%s risk=%d reason=%s",
                tool_name, decision.rule_id, decision.risk_score, decision.reason,
            )
            return ToolMessage(
                content=decision.blocked_message,
                tool_call_id=call_id,
                name=tool_name,
            )

        # ── 4. ALLOW — 委托执行 ──
        # HALF_OPEN 状态下试探成功 → 恢复 CLOSED
        self._cb.record_allow()

        result = await tool_registry.execute(
            name=tool_name,
            call_id=call_id,
            arguments=tool_params,
        )

        # 记录到风险上下文（用于后续 combo 检测）
        self._gov._session_risk.record_call(tool_name, tool_params)

        return result

    # ── 上下文管理 ────────────────────────────────

    def set_active_context(self, ctx: "ExecutionContext"):
        """设置当前活跃的执行上下文（PEVRRunner 调用）。

        在状态转换时，PEVRRunner 构造包含 phase/state/observability 的
        ExecutionContext，通过此方法注入。Agent._execute_tool() 在每次
        工具调用前读取此上下文。

        Args:
            ctx: 新的执行上下文。
        """
        self._active_ctx = ctx

    def get_active_context(self) -> "ExecutionContext":
        """获取当前活跃的执行上下文（Agent._execute_tool 调用）。"""
        return self._active_ctx

    def reset_session(self):
        """重置当前会话的安全状态（v0.10.1 新增）。

        在新会话/新请求开始时调用，重置：
        - SessionRiskContext：清空滑动窗口中的工具调用记录
        - SecurityCircuitBreaker：重置到 CLOSED 状态

        这确保每个会话拥有独立的安全上下文，防止跨会话污染。
        在 HTTP 多用户模式下尤为重要——不同用户的请求不应互相影响。
        """
        from .security_governance import ExecutionContext
        if self._gov._session_risk is not None:
            self._gov._session_risk.reset()
        self._cb.reset()
        self._active_ctx = ExecutionContext()

    # ── 查询属性 ──────────────────────────────────

    @property
    def is_degraded(self) -> bool:
        """安全治理是否处于性能降级状态。"""
        return self._gov.is_degraded

    @property
    def circuit_breaker(self) -> "SecurityCircuitBreaker":
        """获取熔断器实例。"""
        return self._cb
