"""
PEVR 状态机 — 枚举驱动的事件状态转换引擎。

设计原则：
    - 纯逻辑：不导入 Agent/LLM/Memory/IO，所有副作用通过回调注入
    - 声明式配置：转换表 dict 定义所有合法状态转移
    - 可独立单元测试：注入 Mock 回调验证调用次数和参数
    - 禁止手写 if-else 流转

状态流转图：
    PLANNING ──(plan_done)──▶ EXECUTING ──(all_steps_done)──▶ VERIFYING ──(passed)──▶ COMPLETED
        │                        │                               │
        └──(plan_failed)──▶ FAILED  ├──(permission_violation)──▶ REPAIRING ──(done)──▶ VERIFYING
                                     │                               │
                                     └──(fatal_error)──▶ FAILED ◀──(max_repairs)──│
                                                                                  │
                                                   VERIFYING ──(failed+max)──────┘

使用方式：
    sm = PEVRStateMachine(initial_state=PEVRState.PLANNING)
    sm.add_transition(
        PEVRState.PLANNING, PEVREvent.PLAN_DONE,
        to_state=PEVRState.EXECUTING,
        guards=[lambda ctx: ctx.get("plan") is not None],
        before=[lambda ctx: logger.info("离开 PLANNING")],
        after=[lambda ctx: logger.info("进入 EXECUTING")],
    )
    await sm.dispatch(PEVREvent.PLAN_DONE, plan=execution_plan)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
import logging

logger = logging.getLogger("pyagent.harness.state_machine")


# ── 状态和事件枚举 ────────────────────────────────────

class PEVRState(str, Enum):
    """PEVR 状态机状态节点。"""
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """是否为终端状态（COMPLETED 或 FAILED）。"""
        return self in (PEVRState.COMPLETED, PEVRState.FAILED)


class PEVREvent(str, Enum):
    """PEVR 状态机事件（驱动状态转换）。"""
    PLAN_DONE = "plan_done"
    PLAN_FAILED = "plan_failed"
    ALL_STEPS_DONE = "all_steps_done"
    PERMISSION_VIOLATION = "permission_violation"
    FATAL_ERROR = "fatal_error"
    VERIFY_PASSED = "verify_passed"
    VERIFY_FAILED = "verify_failed"
    REPAIR_DONE = "repair_done"
    REPAIR_FAILED = "repair_failed"


# ── Transition 定义 ───────────────────────────────────

@dataclass
class Transition:
    """
    状态转换定义。

    Attributes:
        from_state: 源状态。
        event: 触发事件。
        to_state: 目标状态。
        guards: 守卫函数列表，全部返回 True 才允许转换。
                签名: (context: dict) -> bool
        before: 出口动作列表（离开当前状态前执行）。
                签名: async (context: dict) -> None
        after: 入口动作列表（进入目标状态后执行）。
               签名: async (context: dict) -> None
    """
    from_state: PEVRState
    event: PEVREvent
    to_state: PEVRState
    guards: list[Callable[[dict], bool]] = field(default_factory=list)
    before: list[Callable] = field(default_factory=list)
    after: list[Callable] = field(default_factory=list)


# ── 异常 ──────────────────────────────────────────────

class InvalidTransitionError(Exception):
    """尝试执行不合法的状态转换时抛出。"""

    def __init__(self, from_state: PEVRState, event: PEVREvent):
        self.from_state = from_state
        self.event = event
        super().__init__(
            f"非法状态转换: {from_state.value} --({event.value})→ ? "
            f"（未定义从 {from_state.value} 响应 {event.value} 事件的转换）"
        )


class GuardRejectedError(Exception):
    """守卫函数拒绝转换时抛出。"""

    def __init__(self, from_state: PEVRState, event: PEVREvent, reason: str):
        self.from_state = from_state
        self.event = event
        self.reason = reason
        super().__init__(
            f"转换被守卫拒绝: {from_state.value} --({event.value})→ ? "
            f"原因: {reason}"
        )


# ── 状态机核心 ────────────────────────────────────────

class PEVRStateMachine:
    """
    PEVR 枚举驱动状态机。

    特性：
        - 枚举驱动转换表，不使用 if-else
        - 所有副作用通过回调函数注入
        - 守卫/出口/入口动作完全可测试
        - 支持同步和异步回调

    使用方式：
        sm = PEVRStateMachine(initial_state=PEVRState.PLANNING)

        # 注册转换
        sm.add_transition(
            PEVRState.PLANNING, PEVREvent.PLAN_DONE,
            to_state=PEVRState.EXECUTING,
            guards=[plan_is_valid],
            after=[on_enter_executing],
        )

        # 驱动状态机
        await sm.step(PEVREvent.PLAN_DONE, plan=execution_plan)
        print(sm.current_state)  # PEVRState.EXECUTING
    """

    def __init__(self, initial_state: PEVRState = PEVRState.PLANNING):
        """
        Args:
            initial_state: 初始状态，默认 PLANNING。
        """
        self._initial_state = initial_state
        self.current_state: PEVRState = initial_state
        self._transitions: dict[tuple[PEVRState, PEVREvent], Transition] = {}
        self.context: dict[str, Any] = {}
        self.history: list[dict] = []  # 状态转换历史

    # ── 转换注册 ────────────────────────────────────

    def add_transition(
        self,
        from_state: PEVRState,
        event: PEVREvent,
        to_state: PEVRState,
        guards: list[Callable[[dict], bool]] | None = None,
        before: list[Callable] | None = None,
        after: list[Callable] | None = None,
    ):
        """
        注册一条状态转换。

        Args:
            from_state: 源状态。
            event: 触发事件。
            to_state: 目标状态。
            guards: 守卫函数列表（同步，签名为 (ctx: dict) -> bool）。
            before: 出口动作列表（异步，签名为 async (ctx: dict) -> None）。
            after: 入口动作列表（异步，签名为 async (ctx: dict) -> None）。
        """
        key = (from_state, event)
        if key in self._transitions:
            raise ValueError(
                f"转换 {from_state.value} --({event.value})→ "
                f"{self._transitions[key].to_state.value} 已注册"
            )
        self._transitions[key] = Transition(
            from_state=from_state,
            event=event,
            to_state=to_state,
            guards=guards or [],
            before=before or [],
            after=after or [],
        )

    def add_transitions_from_table(
        self,
        table: list[dict],
    ):
        """
        从声明式转换表批量注册。

        Args:
            table: 转换配置列表，每项包含:
                from_state, event, to_state, guards?, before?, after?
        """
        for entry in table:
            self.add_transition(
                from_state=entry["from_state"],
                event=entry["event"],
                to_state=entry["to_state"],
                guards=entry.get("guards"),
                before=entry.get("before"),
                after=entry.get("after"),
            )

    # ── 状态查询 ────────────────────────────────────

    def is_terminal(self) -> bool:
        """当前是否处于终端状态。"""
        return self.current_state.is_terminal

    def get_available_events(self) -> list[PEVREvent]:
        """返回当前状态下可响应的所有事件。"""
        return [
            event for (state, event) in self._transitions
            if state == self.current_state
        ]

    def can_respond_to(self, event: PEVREvent) -> bool:
        """当前状态是否能响应指定事件。"""
        return (self.current_state, event) in self._transitions

    # ── 状态转换 ────────────────────────────────────

    async def step(self, event: PEVREvent, **kwargs) -> PEVRState:
        """
        触发一次状态转换。

        执行流程：
            1. 查找转换 (current_state, event)
            2. 执行所有 guards（同步，任一返回 False 则抛出 GuardRejectedError）
            3. 执行所有 before 回调（异步，出口动作）
            4. 更新 current_state
            5. 执行所有 after 回调（异步，入口动作）
            6. 记录转换历史

        Args:
            event: 触发事件。
            **kwargs: 合并到 context 的键值对。

        Returns:
            转换后的状态。

        Raises:
            InvalidTransitionError: 当前状态无此事件的转换定义。
            GuardRejectedError: 守卫函数拒绝转换。
        """
        key = (self.current_state, event)
        if key not in self._transitions:
            raise InvalidTransitionError(self.current_state, event)

        transition = self._transitions[key]

        # 合并 kwargs 到 context
        self.context.update(kwargs)
        self.context["current_state"] = self.current_state
        self.context["event"] = event

        # 1. 执行守卫（同步）
        for guard in transition.guards:
            try:
                result = guard(self.context)
                if not result:
                    raise GuardRejectedError(
                        self.current_state, event,
                        f"守卫 {guard.__name__} 返回 False"
                    )
            except GuardRejectedError:
                raise
            except Exception as e:
                raise GuardRejectedError(
                    self.current_state, event,
                    f"守卫 {getattr(guard, '__name__', str(guard))} 异常: {e}"
                )

        # 2. 执行出口动作（before）
        for action in transition.before:
            await self._invoke(action, self.context)

        # 3. 状态转换
        old_state = self.current_state
        self.current_state = transition.to_state
        self.context["current_state"] = self.current_state

        logger.debug(
            "[StateMachine] %s --(%s)--> %s",
            old_state.value, event.value, self.current_state.value,
        )

        # 4. 记录历史（必须在 after 回调之前，确保链式转换按时间顺序记录）
        self.history.append({
            "from": old_state.value,
            "event": event.value,
            "to": self.current_state.value,
        })

        # 5. 执行入口动作（after）
        for action in transition.after:
            await self._invoke(action, self.context)

        return self.current_state

    async def dispatch(self, event: PEVREvent, **kwargs) -> PEVRState:
        """
        与 step() 同义，语义化别名。驱动状态机前进。
        """
        return await self.step(event, **kwargs)

    def reset(self):
        """重置到初始状态，清空 context 和历史。"""
        self.current_state = self._initial_state
        self.context = {}
        self.history = []

    # ── 辅助 ──────────────────────────────────────

    @staticmethod
    async def _invoke(fn, ctx: dict):
        """调用回调函数，自动判断同步/异步。"""
        import asyncio
        if asyncio.iscoroutinefunction(fn):
            await fn(ctx)
        else:
            result = fn(ctx)
            if asyncio.iscoroutine(result):
                await result
