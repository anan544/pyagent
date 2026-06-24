"""
安全治理层集成测试。

覆盖：
    - SecurityDecision / ExecutionContext 模型
    - SecurityGovernance: phase restriction / combo risk / param whitelist
    - GovernanceWrapper: 熔断拦截 / 正常放行 / 性能降级
    - 审计事件可选注入
"""

import sys
sys.path.insert(0, '.')
import time
import pytest
from unittest.mock import Mock, AsyncMock, patch
from pyagent.harness.context.security_governance import (
    SecurityDecision, ExecutionContext, SecurityGovernance,
)
from pyagent.harness.context.governance_wrapper import GovernanceWrapper
from pyagent.harness.context.session_risk_context import SessionRiskContext
from pyagent.harness.context.security_circuit_breaker import SecurityCircuitBreaker
from pyagent.harness.context.parameter_validator import ParameterWhitelistValidator
from pyagent.harness.context.combo_rules import ComboRule, ComboRuleEngine


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

class _FakeToolCall:
    def __init__(self, name, args, call_id="call_1"):
        self.function_name = name
        self.arguments = args
        self.id = call_id


class _FakeToolMessage:
    def __init__(self, content, tool_call_id="", name=""):
        self.content = content
        self.tool_call_id = tool_call_id
        self.name = name


class _FakeRegistry:
    """Mock ToolRegistry for testing."""
    def __init__(self):
        self.executed = []

    async def execute(self, name, call_id, arguments):
        self.executed.append({"name": name, "call_id": call_id, "arguments": arguments})
        return _FakeToolMessage(content=f"executed {name}", tool_call_id=call_id, name=name)


# ═══════════════════════════════════════════════════════════════
# Test: SecurityDecision & ExecutionContext
# ═══════════════════════════════════════════════════════════════

class TestSecurityDecision:
    def test_allow_decision(self):
        d = SecurityDecision.allow()
        assert d.allowed is True
        assert d.rule_id == "default_allow"
        assert d.risk_score == 0
        assert d.blocked_message == ""

    def test_block_decision(self):
        d = SecurityDecision.block("test_rule", 90, "dangerous operation", tool_name="test_tool")
        assert d.allowed is False
        assert d.rule_id == "test_rule"
        assert d.risk_score == 90
        assert "test_tool" in d.blocked_message
        assert "test_rule" in d.blocked_message

    def test_allow_with_performance(self):
        d = SecurityDecision.allow(performance_ms=12.5)
        assert d.performance_ms == 12.5

    def test_is_frozen(self):
        d = SecurityDecision.allow()
        with pytest.raises(Exception):
            d.allowed = False


class TestExecutionContext:
    def test_defaults(self):
        ctx = ExecutionContext()
        assert ctx.phase is None
        assert ctx.state is None
        assert ctx.observability is None
        assert ctx.trace_id == ""

    def test_with_phase(self):
        ctx = ExecutionContext(phase="executing", trace_id="abc")
        ctx2 = ctx.with_phase("verifying")
        assert ctx2.phase == "verifying"
        assert ctx2.trace_id == "abc"
        # Original unchanged
        assert ctx.phase == "executing"

    def test_is_frozen(self):
        ctx = ExecutionContext(phase="executing")
        with pytest.raises(Exception):
            ctx.phase = "verifying"


# ═══════════════════════════════════════════════════════════════
# Test: SecurityGovernance (Layer 1 checks)
# ═══════════════════════════════════════════════════════════════

class TestSecurityGovernancePhaseRestriction:
    """Layer 1.1: 阶段限制。"""

    @pytest.fixture
    def gov(self):
        return SecurityGovernance(
            phase_restrictions_enabled=True,
            combo_detection_enabled=False,
            param_whitelist_enabled=False,
        )

    def test_blocks_write_in_planning(self, gov):
        ctx = ExecutionContext(phase="planning")
        d = gov.check("write_file", {"path": "a.py"}, ctx)
        assert not d.allowed
        assert "phase_restriction" in d.rule_id

    def test_blocks_write_in_verifying(self, gov):
        ctx = ExecutionContext(phase="verifying")
        d = gov.check("execute_python", {"code": "x"}, ctx)
        assert not d.allowed

    def test_allows_write_in_executing(self, gov):
        ctx = ExecutionContext(phase="executing")
        d = gov.check("write_file", {"path": "a.py"}, ctx)
        assert d.allowed

    def test_allows_write_in_repairing(self, gov):
        ctx = ExecutionContext(phase="repairing")
        d = gov.check("write_file", {"path": "fix.py"}, ctx)
        assert d.allowed

    def test_allows_read_in_all_phases(self, gov):
        for phase in ["planning", "executing", "verifying", "repairing"]:
            ctx = ExecutionContext(phase=phase)
            d = gov.check("read_file", {"path": "a.py"}, ctx)
            assert d.allowed, f"read_file should be allowed in {phase}"

    def test_no_restriction_when_phase_is_none(self, gov):
        ctx = ExecutionContext()  # ReAct mode
        d = gov.check("write_file", {"path": "a.py"}, ctx)
        assert d.allowed

    def test_disabled_phase_config_skips(self):
        gov = SecurityGovernance(
            phase_restrictions_enabled=False,
            combo_detection_enabled=False,
            param_whitelist_enabled=False,
        )
        ctx = ExecutionContext(phase="planning")
        d = gov.check("write_file", {"path": "a.py"}, ctx)
        assert d.allowed


class TestSecurityGovernanceComboRisk:
    """Layer 1.2: 组合风险检测。"""

    @pytest.fixture
    def gov(self):
        risk = SessionRiskContext(window_seconds=300.0)
        return SecurityGovernance(
            phase_restrictions_enabled=False,
            combo_detection_enabled=True,
            param_whitelist_enabled=False,
            session_risk=risk,
        )

    def test_blocks_write_file_plus_execute_python(self, gov):
        ctx = ExecutionContext(phase="executing")
        # Record write_file in session
        gov._session_risk.record_call("write_file", {"path": "a.py"})
        # Now try execute_python
        d = gov.check("execute_python", {"code": "print(1)", "path": "a.py"}, ctx)
        assert not d.allowed, f"Expected BLOCK, got ALLOW"
        assert "HIGH_RISK_COMBOS" in d.rule_id

    def test_allows_single_write_file(self, gov):
        ctx = ExecutionContext(phase="executing")
        d = gov.check("write_file", {"path": "log.txt"}, ctx)
        assert d.allowed

    def test_allows_single_execute_python_without_write(self, gov):
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_python", {"code": "print(1)"}, ctx)
        assert d.allowed

    def test_no_false_positive_for_safe_combo(self, gov):
        ctx = ExecutionContext(phase="executing")
        gov._session_risk.record_call("read_file", {"path": "a.py"})
        d = gov.check("search_content", {"pattern": "TODO"}, ctx)
        assert d.allowed

    def test_delete_file_alone_is_high_risk(self, gov):
        ctx = ExecutionContext(phase="executing")
        d = gov.check("delete_file", {"path": "important.txt"}, ctx)
        assert not d.allowed  # delete_file is a single-tool HIGH_RISK_COMBO

    def test_combo_without_session_risk_passes(self):
        gov = SecurityGovernance(
            phase_restrictions_enabled=False,
            combo_detection_enabled=True,
            param_whitelist_enabled=False,
            session_risk=None,  # No risk context
        )
        ctx = ExecutionContext(phase="executing")
        d = gov.check("write_file", {"path": "a.py"}, ctx)
        assert d.allowed  # Can't check without risk context

    def test_disabled_combo_config_skips(self, gov):
        gov._combo_enabled = False
        ctx = ExecutionContext(phase="executing")
        gov._session_risk.record_call("write_file", {"path": "a.py"})
        d = gov.check("execute_python", {"code": "x"}, ctx)
        assert d.allowed


class TestSecurityGovernanceParamWhitelist:
    """Layer 1.3: 参数白名单校验。"""

    @pytest.fixture
    def gov(self):
        return SecurityGovernance(
            phase_restrictions_enabled=False,
            combo_detection_enabled=False,
            param_whitelist_enabled=True,
            param_validator=ParameterWhitelistValidator(),
        )

    def test_blocks_dangerous_command(self, gov):
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_command", {"command": "rm -rf /"}, ctx)
        assert not d.allowed
        assert "blocked_pattern" in d.rule_id

    def test_allows_safe_command(self, gov):
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_command", {"command": "pytest tests/"}, ctx)
        assert d.allowed

    def test_blocks_path_traversal(self, gov):
        ctx = ExecutionContext(phase="executing")
        d = gov.check("write_file", {"path": "../../../etc/passwd"}, ctx)
        assert not d.allowed

    def test_disabled_param_config_skips(self, gov):
        gov._param_enabled = False
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_command", {"command": "rm -rf /"}, ctx)
        assert d.allowed


class TestSecurityGovernanceDegradation:
    """性能降级行为。"""

    def test_degrade_disables_combo_detection(self):
        risk = SessionRiskContext()
        gov = SecurityGovernance(
            combo_detection_enabled=True,
            param_whitelist_enabled=False,
            session_risk=risk,
        )
        assert not gov.is_degraded
        gov.disable_combo_detection()
        assert gov.is_degraded
        # Combo detection should be skipped
        risk.record_call("write_file", {"path": "a.py"})
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_python", {"code": "x"}, ctx)
        assert d.allowed  # Should pass because combo check is degraded

    def test_phase_restriction_still_works_after_degrade(self):
        gov = SecurityGovernance(
            phase_restrictions_enabled=True,
            combo_detection_enabled=True,
            param_whitelist_enabled=False,
            session_risk=SessionRiskContext(),
        )
        gov.disable_combo_detection()
        ctx = ExecutionContext(phase="planning")
        d = gov.check("write_file", {"path": "x.py"}, ctx)
        assert not d.allowed  # Phase restriction still active


class TestSecurityGovernanceComboRefinement:
    """双层匹配：快路径 + 精炼路径。"""

    @pytest.fixture
    def gov(self):
        risk = SessionRiskContext()
        rules = [
            ComboRule(
                name="write_then_exec_same_file",
                sequence=["write_file", "execute_python"],
                match_on="file_path",
            ),
        ]
        engine = ComboRuleEngine(rules)
        return SecurityGovernance(
            phase_restrictions_enabled=False,
            combo_detection_enabled=True,
            param_whitelist_enabled=False,
            session_risk=risk,
            combo_rule_engine=engine,
        )

    def test_same_file_path_blocked(self, gov):
        gov._session_risk.record_call("write_file", {"path": "a.py"})
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_python", {"code": "x", "path": "a.py"}, ctx)
        assert not d.allowed

    def test_different_file_path_allowed(self, gov):
        gov._session_risk.record_call("write_file", {"path": "log.txt"})
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_python", {"code": "x", "path": "data_process.py"}, ctx)
        assert d.allowed


# ═══════════════════════════════════════════════════════════════
# Test: GovernanceWrapper
# ═══════════════════════════════════════════════════════════════

class TestGovernanceWrapper:
    """GovernanceWrapper 集成测试。"""

    @pytest.fixture
    def wrapper(self):
        risk = SessionRiskContext(window_seconds=300.0)
        cb = SecurityCircuitBreaker(max_blocks=3, window_seconds=60.0)
        gov = SecurityGovernance(
            phase_restrictions_enabled=True,
            combo_detection_enabled=True,
            param_whitelist_enabled=True,
            session_risk=risk,
            param_validator=ParameterWhitelistValidator(),
        )
        return GovernanceWrapper(governance=gov, circuit_breaker=cb)

    @pytest.mark.asyncio
    async def test_returns_toolmessage_on_block(self, wrapper):
        ctx = ExecutionContext(phase="planning")
        tc = _FakeToolCall("write_file", {"path": "a.py"})
        result = await wrapper.execute_tool(tc, _FakeRegistry(), ctx)
        assert "[安全拦截]" in result.content
        assert "write_file" in result.content
        assert result.tool_call_id == "call_1"

    @pytest.mark.asyncio
    async def test_executes_normally_on_allow(self, wrapper):
        ctx = ExecutionContext(phase="executing")
        tc = _FakeToolCall("read_file", {"path": "a.py"})
        registry = _FakeRegistry()
        result = await wrapper.execute_tool(tc, registry, ctx)
        assert "executed read_file" in result.content
        assert len(registry.executed) == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_tripped_returns_block(self, wrapper):
        ctx = ExecutionContext(phase="planning")
        tc = _FakeToolCall("write_file", {"path": "a.py"})
        # Trip the breaker
        for _ in range(3):
            await wrapper.execute_tool(tc, _FakeRegistry(), ctx)
        # Now even safe tools should be blocked
        tc2 = _FakeToolCall("read_file", {"path": "b.txt"})
        result = await wrapper.execute_tool(tc2, _FakeRegistry(), ctx)
        assert "安全熔断" in result.content

    @pytest.mark.asyncio
    async def test_records_block_in_circuit_breaker(self, wrapper):
        ctx = ExecutionContext(phase="planning")
        tc = _FakeToolCall("write_file", {"path": "a.py"})
        await wrapper.execute_tool(tc, _FakeRegistry(), ctx)
        assert wrapper.circuit_breaker.block_count == 1

    @pytest.mark.asyncio
    async def test_records_call_in_risk_context_on_allow(self, wrapper):
        ctx = ExecutionContext(phase="executing")
        tc = _FakeToolCall("read_file", {"path": "a.py"})
        await wrapper.execute_tool(tc, _FakeRegistry(), ctx)
        names = wrapper._gov._session_risk.recent_tool_names()
        assert "read_file" in names

    def test_is_degraded_delegates_to_governance(self, wrapper):
        assert not wrapper.is_degraded
        wrapper._gov.disable_combo_detection()
        assert wrapper.is_degraded

    @pytest.mark.asyncio
    async def test_performance_degradation_mechanism(self):
        """验证性能降级机制的可达性：直接触发 disable_combo_detection。"""
        risk = SessionRiskContext()
        cb = SecurityCircuitBreaker()
        gov = SecurityGovernance(
            phase_restrictions_enabled=False,
            combo_detection_enabled=True,
            param_whitelist_enabled=False,
            session_risk=risk,
            degrade_threshold_ms=50.0,
        )
        wrapper = GovernanceWrapper(governance=gov, circuit_breaker=cb,
                                     enable_perf_monitor=True)
        assert not wrapper.is_degraded

        # 直接触发降级（模拟慢检查场景）
        gov.disable_combo_detection()
        assert wrapper.is_degraded

        # 降级后 combo 检测被跳过
        risk.record_call("write_file", {"path": "a.py"})
        ctx = ExecutionContext(phase="executing")
        tc = _FakeToolCall("execute_python", {"code": "x", "path": "a.py"})
        result = await wrapper.execute_tool(tc, _FakeRegistry(), ctx)
        assert "安全拦截" not in result.content  # combo detection skipped

    @pytest.mark.asyncio
    async def test_active_context_isolation(self, wrapper):
        """验证 ExecutionContext 的不可变性防止竞态。"""
        ctx1 = ExecutionContext(phase="executing", trace_id="trace1")
        ctx2 = ExecutionContext(phase="planning", trace_id="trace2")
        wrapper.set_active_context(ctx1)
        assert wrapper.get_active_context().phase == "executing"
        wrapper.set_active_context(ctx2)
        assert wrapper.get_active_context().phase == "planning"
        # ctx1 unchanged
        assert ctx1.phase == "executing"

    @pytest.mark.asyncio
    async def test_combo_detection_with_active_risk_context(self, wrapper):
        ctx = ExecutionContext(phase="executing")
        # Record a write
        wrapper._gov._session_risk.record_call("write_file", {"path": "a.py"})
        # Now execute_python should be blocked by combo detection
        tc = _FakeToolCall("execute_python", {"code": "print(1)", "path": "a.py"})
        result = await wrapper.execute_tool(tc, _FakeRegistry(), ctx)
        assert "[安全拦截]" in result.content
        assert "HIGH_RISK_COMBOS" in result.content


class TestGovernanceSessionReset:
    """v0.10.1: 跨会话安全状态隔离。"""

    @pytest.fixture
    def wrapper(self):
        risk = SessionRiskContext(window_seconds=300.0)
        cb = SecurityCircuitBreaker(max_blocks=3, window_seconds=60.0)
        gov = SecurityGovernance(
            phase_restrictions_enabled=True,
            combo_detection_enabled=True,
            session_risk=risk,
        )
        return GovernanceWrapper(governance=gov, circuit_breaker=cb)

    @pytest.mark.asyncio
    async def test_reset_clears_circuit_breaker(self, wrapper):
        """新会话重置后熔断器恢复 CLOSED。"""
        ctx = ExecutionContext(phase="planning")
        tc = _FakeToolCall("write_file", {"path": "a.py"})
        # 触发熔断
        for _ in range(3):
            await wrapper.execute_tool(tc, _FakeRegistry(), ctx)
        assert wrapper.circuit_breaker.state.value == "open"

        # 重置会话
        wrapper.reset_session()
        assert wrapper.circuit_breaker.state.value == "closed"
        assert wrapper.circuit_breaker.block_count == 0

    @pytest.mark.asyncio
    async def test_reset_clears_risk_context(self, wrapper):
        """新会话重置后风险上下文清空。"""
        ctx = ExecutionContext(phase="executing")
        tc = _FakeToolCall("read_file", {"path": "a.py"})
        await wrapper.execute_tool(tc, _FakeRegistry(), ctx)
        assert wrapper._gov._session_risk.window_size == 1

        wrapper.reset_session()
        assert wrapper._gov._session_risk.window_size == 0

    @pytest.mark.asyncio
    async def test_reset_clears_active_context(self, wrapper):
        """重置后活跃上下文恢复默认。"""
        wrapper.set_active_context(ExecutionContext(phase="executing", trace_id="xyz"))
        wrapper.reset_session()
        ctx = wrapper.get_active_context()
        assert ctx.phase is None
        assert ctx.trace_id == ""

    @pytest.mark.asyncio
    async def test_reset_prevents_cross_session_contamination(self, wrapper):
        """会话 A 的 BLOCK 不影响会话 B。"""
        ctx = ExecutionContext(phase="planning")
        tc = _FakeToolCall("write_file", {"path": "a.py"})

        # 会话 A：连续触发 2 次 BLOCK（差一次熔断）
        await wrapper.execute_tool(tc, _FakeRegistry(), ctx)
        await wrapper.execute_tool(tc, _FakeRegistry(), ctx)
        assert wrapper.circuit_breaker.block_count == 2

        # 重置 → 会话 B 开始
        wrapper.reset_session()
        assert wrapper.circuit_breaker.block_count == 0

        # 会话 B：正常操作不受影响
        ctx2 = ExecutionContext(phase="executing")
        tc2 = _FakeToolCall("read_file", {"path": "b.txt"})
        result = await wrapper.execute_tool(tc2, _FakeRegistry(), ctx2)
        assert "executed read_file" in result.content
