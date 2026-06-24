"""
安全治理引擎 — Layer 1 工具前置门控编排器。

SecurityGovernance 是独立于执行模式的透明拦截器：
    对下：不关心当前是 ReAct 循环还是 PEVR 状态机，只拦截工具调用
    对上：通过 SecurityDecision 返回结构化的允许/阻止结果

Layer 1 检查清单（按优先级顺序，首个 BLOCK 即短路）：
    1. Phase restriction — O(1) 阶段快速过滤（PLANNING/VERIFYING 禁止写入工具）
    2. Combo risk detection — 双层匹配：快路径 set-based + 精炼路径 field-level match_on
    3. Parameter whitelist — 命令前缀/危险模式/路径穿越

使用方式：
    from pyagent.harness.context.security_governance import (
        SecurityGovernance, SecurityDecision, ExecutionContext,
    )

    gov = SecurityGovernance(config, session_risk, param_validator)
    decision = gov.check("write_file", {"path": "a.py"}, ExecutionContext(phase="executing"))
    if not decision.allowed:
        print(decision.blocked_message)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .session_risk_context import SessionRiskContext
    from .parameter_validator import ParameterWhitelistValidator
    from .combo_rules import ComboRuleEngine

logger = logging.getLogger("pyagent.security.governance")

# ═══════════════════════════════════════════════════════════════
# 不可变值对象
# ═══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SecurityDecision:
    """安全校验结果 — 不可变值对象。

    Attributes:
        allowed: True = 允许执行, False = 阻止。
        rule_id: 匹配的规则 ID（如 "phase_restriction", "HIGH_RISK_COMBOS[0]"）。
        risk_score: 风险评分 0-100。
        reason: 人类可读的拦截原因。
        blocked_message: 当 allowed=False 时，返回给 LLM 的 ToolMessage 内容。
        performance_ms: 本次校验耗时（毫秒）。
    """

    allowed: bool
    rule_id: str
    risk_score: int
    reason: str
    blocked_message: str = ""
    performance_ms: float = 0.0

    @staticmethod
    def allow(rule_id: str = "default_allow", performance_ms: float = 0.0) -> "SecurityDecision":
        """构造 ALLOW 决策。"""
        return SecurityDecision(
            allowed=True, rule_id=rule_id, risk_score=0,
            reason="", performance_ms=performance_ms,
        )

    @staticmethod
    def block(
        rule_id: str,
        risk_score: int,
        reason: str,
        tool_name: str = "",
    ) -> "SecurityDecision":
        """构造 BLOCK 决策，自动生成面向 LLM 的 blocked_message。"""
        msg_parts = [f"[安全拦截] 工具 '{tool_name}' 被阻止。"]
        msg_parts.append(f"规则: {rule_id}")
        msg_parts.append(f"原因: {reason}")
        msg_parts.append("建议: 请尝试其他方式完成任务，或请求人工审查此规则。")
        return SecurityDecision(
            allowed=False,
            rule_id=rule_id,
            risk_score=risk_score,
            reason=reason,
            blocked_message="\n".join(msg_parts),
        )


@dataclass(frozen=True)
class ExecutionContext:
    """执行上下文 — 不可变，由调用方在每次工具执行前构造。

    设计原则（优化 1）：
        - 不可变 → 无竞态条件
        - PEVRRunner 在状态转换回调中构造新实例
        - ReAct 模式使用 ExecutionContext() 默认值
        - 测试可直接构造，无需 mock Agent 属性

    Attributes:
        phase: PEVR 阶段（"planning"/"executing"/"verifying"/"repairing"）或 None（ReAct）。
        state: 状态机当前状态（PEVRState 值字符串）或 None。
        observability: ObservabilityContext 实例，用于审计事件发送（可选）。
        trace_id: 生命周期追踪 ID。
    """

    phase: Optional[str] = None
    state: Optional[str] = None
    observability: Any = None
    trace_id: str = ""

    def with_phase(self, phase: Optional[str]) -> "ExecutionContext":
        """返回新的 ExecutionContext，仅替换 phase。"""
        return ExecutionContext(
            phase=phase,
            state=self.state,
            observability=self.observability,
            trace_id=self.trace_id,
        )


# ═══════════════════════════════════════════════════════════════
# SecurityGovernance — Layer 1 编排器
# ═══════════════════════════════════════════════════════════════


class SecurityGovernance:
    """安全治理引擎 — 编排 Layer 1 三项检查。

    检查顺序（首个 BLOCK 短路）：
        1. Phase restriction（O(1) 快速过滤）
        2. Combo risk detection（双层匹配）
        3. Parameter whitelist validation

    性能降级（优化 4）：
        当 check() 耗时超过 degrade_threshold_ms 时，自动禁用 combo_detection
        （最耗时的检查），保留 phase restriction + param whitelist。
    """

    def __init__(
        self,
        phase_restrictions_enabled: bool = True,
        blocked_in_readonly_phases: Optional[list[str]] = None,
        combo_detection_enabled: bool = True,
        param_whitelist_enabled: bool = True,
        session_risk: Optional["SessionRiskContext"] = None,
        param_validator: Optional["ParameterWhitelistValidator"] = None,
        combo_rule_engine: Optional["ComboRuleEngine"] = None,
        degrade_threshold_ms: float = 50.0,
    ):
        """
        Args:
            phase_restrictions_enabled: 是否启用阶段限制。
            blocked_in_readonly_phases: 在只读阶段禁止的工具名列表。
            combo_detection_enabled: 是否启用组合风险检测。
            param_whitelist_enabled: 是否启用参数白名单校验。
            session_risk: 会话风险上下文（用于 combo 检测）。
            param_validator: 参数白名单校验器。
            combo_rule_engine: 字段级 combo 规则引擎（可选）。
            degrade_threshold_ms: 性能降级阈值（毫秒）。
        """
        self._phase_enabled = phase_restrictions_enabled
        self._blocked_in_readonly = blocked_in_readonly_phases or [
            "write_file", "execute_python", "execute_command", "delete_file",
        ]
        self._combo_enabled = combo_detection_enabled
        self._param_enabled = param_whitelist_enabled
        self._session_risk = session_risk
        self._param_validator = param_validator
        self._combo_engine = combo_rule_engine
        self._degrade_threshold_ms = degrade_threshold_ms

        # 性能降级状态
        self._degraded = False
        self._degraded_at: float = 0.0

    # ── 主入口 ────────────────────────────────────

    def check(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        ctx: ExecutionContext,
    ) -> SecurityDecision:
        """执行 Layer 1 全部检查。

        Args:
            tool_name: 工具名称。
            tool_params: LLM 传入的工具参数。
            ctx: 执行上下文（阶段、审计、trace_id）。

        Returns:
            SecurityDecision — allowed=True 放行，allowed=False 阻止。
        """
        # 若已性能降级，跳过 combo 检测
        combo_enabled = self._combo_enabled and not self._degraded

        # ── 1. Phase restriction (O(1) fast-fail) ──
        if self._phase_enabled and ctx.phase:
            decision = self._check_phase_restriction(tool_name, ctx.phase)
            if decision is not None:
                self._emit(decision, tool_name, tool_params, ctx)
                return decision

        # ── 2. Combo risk detection ──
        if combo_enabled and self._session_risk is not None:
            decision = self._check_combo_risk(tool_name, tool_params, ctx)
            if decision is not None:
                self._emit(decision, tool_name, tool_params, ctx)
                return decision

        # ── 3. Parameter whitelist validation ──
        if self._param_enabled and self._param_validator is not None:
            decision = self._param_validator.validate(tool_name, tool_params)
            if decision is not None:
                self._emit(decision, tool_name, tool_params, ctx)
                return decision

        # ── ALLOW ──
        decision = SecurityDecision.allow()
        self._emit(decision, tool_name, tool_params, ctx)
        return decision

    # ── 单项检查 ──────────────────────────────────

    def _check_phase_restriction(
        self, tool_name: str, phase: str,
    ) -> Optional[SecurityDecision]:
        """阶段限制：PLANNING/VERIFYING 禁止写入类工具。"""
        if phase not in ("planning", "verifying"):
            return None
        if tool_name in self._blocked_in_readonly:
            return SecurityDecision.block(
                "phase_restriction",
                risk_score=80,
                reason=(
                    f"工具 '{tool_name}' 不允许在 {phase.upper()} 阶段执行。"
                    f"该阶段为只读阶段，仅允许读取类操作。"
                ),
                tool_name=tool_name,
            )
        return None

    def _check_combo_risk(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        ctx: ExecutionContext,
    ) -> Optional[SecurityDecision]:
        """组合风险检测：双层匹配。

        快路径：检查现有 HIGH_RISK_COMBOS（set-based，不改动 permission.py）。
        精炼路径：若有 combo_rules 配置且命中快路径，额外要求字段级匹配。
        """
        from .permission import HIGH_RISK_COMBOS

        # 快路径：获取会话窗口内的工具名集合 + 当前候选工具
        assert self._session_risk is not None  # guarded by caller
        recent = self._session_risk.recent_tool_names(include_current=tool_name)
        recent_set = set(recent)

        for idx, combo in enumerate(HIGH_RISK_COMBOS):
            if not combo.issubset(recent_set):
                continue

            # 快路径命中 — 检查是否有精炼规则
            rule_id = f"HIGH_RISK_COMBOS[{idx}]"
            if self._combo_engine is not None:
                refined = self._combo_engine.match(
                    combo=combo,
                    current_tool=tool_name,
                    current_params=tool_params,
                    recent_records=self._session_risk.recent_records(),
                )
                if refined is False:
                    # 精炼路径不匹配 → 放行（降低误报）
                    logger.debug(
                        "[SecurityGovernance] Combo %s 快路径命中但精炼路径不匹配，放行",
                        sorted(combo),
                    )
                    continue
                if isinstance(refined, str):
                    rule_id = refined  # 精炼规则名称

            # 命中！
            from .observability import compute_risk_score
            score = compute_risk_score(combo)
            return SecurityDecision.block(
                rule_id,
                risk_score=score,
                reason=f"高危工具组合检测: {' + '.join(sorted(combo))}",
                tool_name=tool_name,
            )

        return None

    # ── 审计事件发送 ──────────────────────────────

    def _emit(
        self,
        decision: SecurityDecision,
        tool_name: str,
        tool_params: dict[str, Any],
        ctx: ExecutionContext,
    ):
        """发送 SecurityAuditEvent（若 ObservabilityContext 可用）。"""
        obs = ctx.observability
        if obs is None:
            return
        try:
            obs.log_decision(
                decision="ALLOW" if decision.allowed else "BLOCK",
                rule_id=decision.rule_id,
                tool_name=tool_name,
                tool_params=tool_params,
                risk_score=decision.risk_score,
                phase=ctx.phase or "",
                state=ctx.state or "",
                metadata={"reason": decision.reason},
            )
        except Exception:
            logger.debug("[SecurityGovernance] 审计事件发送失败", exc_info=True)

    # ── 性能降级 ──────────────────────────────────

    def disable_combo_detection(self):
        """性能降级：禁用 combo 检测（保留 phase + param whitelist）。"""
        if not self._degraded:
            self._degraded = True
            self._degraded_at = time.monotonic()
            logger.warning(
                "[SecurityGovernance] 性能降级：combo 检测已禁用 "
                "(check 耗时超过 %.1fms 阈值)。"
                "phase restriction + param whitelist 仍生效。",
                self._degrade_threshold_ms,
            )

    @property
    def is_degraded(self) -> bool:
        """是否处于性能降级状态。"""
        return self._degraded

    @property
    def degrade_threshold_ms(self) -> float:
        """性能降级阈值（毫秒）。"""
        return self._degrade_threshold_ms
