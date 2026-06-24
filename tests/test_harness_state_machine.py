"""
PEVR 状态机单元测试。

覆盖：
    - 状态转换正确性：合法/非法转换
    - 守卫函数拦截
    - 入口/出口动作执行顺序
    - 终端状态检测
    - 上下文管理
    - 转换历史记录
    - 纯逻辑可测（Mock 回调验证调用次数和参数）
"""

from unittest.mock import Mock, AsyncMock, call
import pytest

from pyagent.harness.context.state_machine import (
    PEVRStateMachine,
    PEVRState,
    PEVREvent,
    Transition,
    InvalidTransitionError,
    GuardRejectedError,
)


# ═══════════════════════════════════════════════════════════════
# 状态和事件枚举测试
# ═══════════════════════════════════════════════════════════════

class TestPEVRState:
    def test_terminal_states(self):
        assert PEVRState.COMPLETED.is_terminal is True
        assert PEVRState.FAILED.is_terminal is True

    def test_non_terminal_states(self):
        assert PEVRState.PLANNING.is_terminal is False
        assert PEVRState.EXECUTING.is_terminal is False
        assert PEVRState.VERIFYING.is_terminal is False
        assert PEVRState.REPAIRING.is_terminal is False

    def test_string_value(self):
        assert PEVRState.PLANNING.value == "planning"
        assert PEVRState.EXECUTING.value == "executing"


class TestPEVREvent:
    def test_event_values(self):
        assert PEVREvent.PLAN_DONE.value == "plan_done"
        assert PEVREvent.PLAN_FAILED.value == "plan_failed"
        assert PEVREvent.ALL_STEPS_DONE.value == "all_steps_done"
        assert PEVREvent.PERMISSION_VIOLATION.value == "permission_violation"
        assert PEVREvent.VERIFY_PASSED.value == "verify_passed"


# ═══════════════════════════════════════════════════════════════
# PEVRStateMachine 核心测试
# ═══════════════════════════════════════════════════════════════

class TestStateMachineBasic:
    """基础状态机功能测试。"""

    def test_initial_state(self):
        sm = PEVRStateMachine()
        assert sm.current_state == PEVRState.PLANNING
        assert not sm.is_terminal()

    def test_custom_initial_state(self):
        sm = PEVRStateMachine(initial_state=PEVRState.EXECUTING)
        assert sm.current_state == PEVRState.EXECUTING

    def test_reset(self):
        sm = PEVRStateMachine(initial_state=PEVRState.PLANNING)
        sm.context["foo"] = "bar"
        sm.reset()
        assert sm.current_state == PEVRState.PLANNING
        assert sm.context == {}
        assert sm.history == []

    def test_is_terminal(self):
        sm = PEVRStateMachine()
        assert not sm.is_terminal()
        sm.current_state = PEVRState.COMPLETED
        assert sm.is_terminal()
        sm.current_state = PEVRState.FAILED
        assert sm.is_terminal()


class TestTransitionRegistration:
    """转换注册测试。"""

    def test_add_valid_transition(self):
        sm = PEVRStateMachine()
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
        )
        assert sm.can_respond_to(PEVREvent.PLAN_DONE)

    def test_add_duplicate_transition_raises(self):
        sm = PEVRStateMachine()
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
        )
        with pytest.raises(ValueError, match="已注册"):
            sm.add_transition(
                PEVRState.PLANNING, PEVREvent.PLAN_DONE,
                to_state=PEVRState.FAILED,
            )

    def test_add_transitions_from_table(self):
        sm = PEVRStateMachine()
        table = [
            {
                "from_state": PEVRState.PLANNING,
                "event": PEVREvent.PLAN_DONE,
                "to_state": PEVRState.EXECUTING,
            },
            {
                "from_state": PEVRState.PLANNING,
                "event": PEVREvent.PLAN_FAILED,
                "to_state": PEVRState.FAILED,
            },
        ]
        sm.add_transitions_from_table(table)
        assert sm.can_respond_to(PEVREvent.PLAN_DONE)
        assert sm.can_respond_to(PEVREvent.PLAN_FAILED)

    def test_cannot_respond_to_unregistered_event(self):
        sm = PEVRStateMachine()
        assert not sm.can_respond_to(PEVREvent.VERIFY_PASSED)

    def test_get_available_events(self):
        sm = PEVRStateMachine()
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
        )
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_FAILED,
            to_state=PEVRState.FAILED,
        )
        events = sm.get_available_events()
        assert PEVREvent.PLAN_DONE in events
        assert PEVREvent.PLAN_FAILED in events
        assert PEVREvent.VERIFY_PASSED not in events


class TestStateTransition:
    """状态转换执行测试。"""

    @pytest.mark.asyncio
    async def test_simple_transition(self):
        sm = PEVRStateMachine()
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
        )
        result = await sm.step(PEVREvent.PLAN_DONE)
        assert result == PEVRState.EXECUTING
        assert sm.current_state == PEVRState.EXECUTING

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self):
        sm = PEVRStateMachine()
        with pytest.raises(InvalidTransitionError):
            await sm.step(PEVREvent.VERIFY_PASSED)

    @pytest.mark.asyncio
    async def test_transition_to_terminal(self):
        sm = PEVRStateMachine(initial_state=PEVRState.VERIFYING)
        sm.add_transition(
            PEVRState.VERIFYING, PEVREvent.VERIFY_PASSED,
            to_state=PEVRState.COMPLETED,
        )
        result = await sm.step(PEVREvent.VERIFY_PASSED)
        assert result == PEVRState.COMPLETED
        assert sm.is_terminal()

    @pytest.mark.asyncio
    async def test_context_update_during_transition(self):
        sm = PEVRStateMachine()
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
        )
        await sm.step(PEVREvent.PLAN_DONE, plan_id="p123", steps=5)
        assert sm.context["plan_id"] == "p123"
        assert sm.context["steps"] == 5
        assert sm.context["current_state"] == PEVRState.EXECUTING

    @pytest.mark.asyncio
    async def test_history_recorded(self):
        sm = PEVRStateMachine()
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
        )
        await sm.step(PEVREvent.PLAN_DONE)
        assert len(sm.history) == 1
        assert sm.history[0]["from"] == "planning"
        assert sm.history[0]["event"] == "plan_done"
        assert sm.history[0]["to"] == "executing"

    @pytest.mark.asyncio
    async def test_dispatch_alias(self):
        sm = PEVRStateMachine()
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
        )
        result = await sm.dispatch(PEVREvent.PLAN_DONE)
        assert result == PEVRState.EXECUTING


class TestGuards:
    """守卫函数测试。"""

    @pytest.mark.asyncio
    async def test_guard_allows(self):
        sm = PEVRStateMachine()
        passed = []
        guard = lambda ctx: passed.append(True) or True
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            guards=[guard],
        )
        result = await sm.step(PEVREvent.PLAN_DONE)
        assert result == PEVRState.EXECUTING
        assert len(passed) == 1

    @pytest.mark.asyncio
    async def test_guard_rejects(self):
        sm = PEVRStateMachine()
        guard = lambda ctx: False
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            guards=[guard],
        )
        with pytest.raises(GuardRejectedError):
            await sm.step(PEVREvent.PLAN_DONE)
        # 状态不应改变
        assert sm.current_state == PEVRState.PLANNING

    @pytest.mark.asyncio
    async def test_multiple_guards_all_must_pass(self):
        sm = PEVRStateMachine()
        guard1 = lambda ctx: True
        guard2 = lambda ctx: ctx.get("ready", False)
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            guards=[guard1, guard2],
        )
        # guard2 fails because ctx["ready"] is not set
        with pytest.raises(GuardRejectedError):
            await sm.step(PEVREvent.PLAN_DONE)

    @pytest.mark.asyncio
    async def test_guard_receives_context(self):
        sm = PEVRStateMachine()
        received = {}
        guard = lambda ctx: received.update(ctx) or True
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            guards=[guard],
        )
        await sm.step(PEVREvent.PLAN_DONE, my_key="my_value")
        assert received.get("my_key") == "my_value"
        assert received.get("current_state") == PEVRState.PLANNING

    @pytest.mark.asyncio
    async def test_guard_exception_wraps_to_guard_rejected(self):
        sm = PEVRStateMachine()
        def bad_guard(ctx):
            raise RuntimeError("boom")
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            guards=[bad_guard],
        )
        with pytest.raises(GuardRejectedError, match="boom"):
            await sm.step(PEVREvent.PLAN_DONE)


class TestCallbacks:
    """入口/出口动作回调测试。"""

    @pytest.mark.asyncio
    async def test_before_callback_called(self):
        sm = PEVRStateMachine()
        before_log = []
        async def before_fn(ctx):
            before_log.append(f"leaving {ctx['current_state'].value}")
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            before=[before_fn],
        )
        await sm.step(PEVREvent.PLAN_DONE)
        assert len(before_log) == 1
        assert "leaving planning" in before_log[0]

    @pytest.mark.asyncio
    async def test_after_callback_called(self):
        sm = PEVRStateMachine()
        after_log = []
        async def after_fn(ctx):
            after_log.append(f"entered {ctx['current_state'].value}")
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            after=[after_fn],
        )
        await sm.step(PEVREvent.PLAN_DONE)
        assert len(after_log) == 1
        assert "entered executing" in after_log[0]

    @pytest.mark.asyncio
    async def test_before_after_execution_order(self):
        sm = PEVRStateMachine()
        order = []
        async def before_fn(ctx):
            order.append("before")
        async def after_fn(ctx):
            order.append("after")
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            before=[before_fn],
            after=[after_fn],
        )
        await sm.step(PEVREvent.PLAN_DONE)
        assert order == ["before", "after"]

    @pytest.mark.asyncio
    async def test_multiple_callbacks_in_order(self):
        sm = PEVRStateMachine()
        calls = []
        async def cb1(ctx):
            calls.append("cb1")
        async def cb2(ctx):
            calls.append("cb2")
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            after=[cb1, cb2],
        )
        await sm.step(PEVREvent.PLAN_DONE)
        assert calls == ["cb1", "cb2"]

    @pytest.mark.asyncio
    async def test_sync_callback_in_after(self):
        sm = PEVRStateMachine()
        called = []
        def sync_cb(ctx):
            called.append(True)
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            after=[sync_cb],
        )
        await sm.step(PEVREvent.PLAN_DONE)
        assert len(called) == 1

    @pytest.mark.asyncio
    async def test_callback_receives_full_context(self):
        sm = PEVRStateMachine()
        received_kwargs = {}
        async def after_fn(ctx):
            received_kwargs.update(ctx)
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            after=[after_fn],
        )
        await sm.step(PEVREvent.PLAN_DONE, extra_data="hello")
        assert received_kwargs.get("extra_data") == "hello"
        # 状态已更新
        assert received_kwargs.get("current_state") == PEVRState.EXECUTING


class TestChainedTransitions:
    """状态机链式转换测试（回调中触发下一步）。"""

    @pytest.mark.asyncio
    async def test_full_pevr_chain(self):
        """验证完整 PEVR 链: PLANNING → EXECUTING → VERIFYING → COMPLETED。"""
        sm = PEVRStateMachine(initial_state=PEVRState.PLANNING)

        # 注入 _sm 引用到上下文，供回调使用
        sm.context["_sm"] = sm

        # 注册所有转换
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            after=[_chain_to_executing],
        )
        sm.add_transition(
            PEVRState.EXECUTING, PEVREvent.ALL_STEPS_DONE,
            to_state=PEVRState.VERIFYING,
            after=[_chain_to_verify],
        )
        sm.add_transition(
            PEVRState.VERIFYING, PEVREvent.VERIFY_PASSED,
            to_state=PEVRState.COMPLETED,
        )

        # 启动链
        await sm.step(PEVREvent.PLAN_DONE)

        # 验证走完全程
        assert sm.current_state == PEVRState.COMPLETED
        assert len(sm.history) == 3
        assert sm.history[0]["to"] == "executing"
        assert sm.history[1]["to"] == "verifying"
        assert sm.history[2]["to"] == "completed"

    @pytest.mark.asyncio
    async def test_chain_with_repair(self):
        """验证失败→修补→重试验收的链。"""
        sm = PEVRStateMachine(initial_state=PEVRState.VERIFYING)
        sm.context["_sm"] = sm

        sm.add_transition(
            PEVRState.VERIFYING, PEVREvent.VERIFY_FAILED,
            to_state=PEVRState.REPAIRING,
            after=[_chain_to_repair_done],
        )
        sm.add_transition(
            PEVRState.REPAIRING, PEVREvent.REPAIR_DONE,
            to_state=PEVRState.VERIFYING,
            after=[_chain_to_verify_pass],
        )
        sm.add_transition(
            PEVRState.VERIFYING, PEVREvent.VERIFY_PASSED,
            to_state=PEVRState.COMPLETED,
        )

        await sm.step(PEVREvent.VERIFY_FAILED)
        assert sm.current_state == PEVRState.COMPLETED
        assert len(sm.history) == 3


# ── 链式转换辅助回调 ──────────────────────────────

async def _chain_to_executing(ctx: dict):
    sm = ctx.get("_sm")
    if sm:
        await sm.step(PEVREvent.ALL_STEPS_DONE, _sm=sm)


async def _chain_to_verify(ctx: dict):
    sm = ctx.get("_sm")
    if sm:
        await sm.step(PEVREvent.VERIFY_PASSED, _sm=sm)


async def _chain_to_repair_done(ctx: dict):
    sm = ctx.get("_sm")
    if sm:
        await sm.step(PEVREvent.REPAIR_DONE, _sm=sm)


async def _chain_to_verify_pass(ctx: dict):
    sm = ctx.get("_sm")
    if sm:
        await sm.step(PEVREvent.VERIFY_PASSED, _sm=sm)


# ═══════════════════════════════════════════════════════════════
# Transition dataclass 测试
# ═══════════════════════════════════════════════════════════════

class TestTransitionDataclass:
    def test_default_fields(self):
        t = Transition(
            from_state=PEVRState.PLANNING,
            event=PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
        )
        assert t.guards == []
        assert t.before == []
        assert t.after == []

    def test_with_callbacks(self):
        guard = lambda ctx: True
        before = lambda ctx: None
        after = lambda ctx: None
        t = Transition(
            from_state=PEVRState.PLANNING,
            event=PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            guards=[guard],
            before=[before],
            after=[after],
        )
        assert len(t.guards) == 1
        assert len(t.before) == 1
        assert len(t.after) == 1
