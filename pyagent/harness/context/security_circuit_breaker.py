"""
安全事件熔断器 — Layer 3 运行时保护（优化 3）。

基于滑动窗口的安全事件计数熔断器，采用经典三态 Circuit Breaker 模式：

    CLOSED  ──(block_count >= max_blocks)──▶ OPEN
    OPEN    ──(cooldown elapsed)──────────▶ HALF_OPEN
    HALF_OPEN ──(probe ALLOW)─────────────▶ CLOSED
    HALF_OPEN ──(probe BLOCK, backoff)────▶ OPEN

设计要点：
    - 与 repair_context.CircuitBreaker（硬计数器）不同，安全熔断用滑动窗口
    - Half-Open 状态防止"熔断→恢复→再熔断"抖动
    - 指数退避（backoff_factor）确保故障 LLM 不会无限重试
    - 线程安全：所有方法同步，由 GovernanceWrapper 层协调并发

使用方式：
    cb = SecurityCircuitBreaker(max_blocks=5, window_seconds=60.0)
    if cb.is_tripped():
        return ToolMessage("熔断")
    # ... tool executed, got BLOCK decision ...
    cb.record_block()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("pyagent.security.circuit_breaker")


class BreakerState(Enum):
    """熔断器状态。"""
    CLOSED = "closed"        # 正常：计数 BLOCK 事件
    OPEN = "open"            # 熔断：拒绝所有调用
    HALF_OPEN = "half_open"  # 半开：允许 1 次试探性调用


@dataclass
class _BlockRecord:
    """单次 BLOCK 事件记录。"""
    tool_name: str
    rule_id: str
    timestamp: float


class SecurityCircuitBreaker:
    """安全事件滑动窗口熔断器。

    Attributes:
        state: 当前状态（CLOSED / OPEN / HALF_OPEN）。
        block_count: 当前窗口内的 BLOCK 事件数。
    """

    def __init__(
        self,
        max_blocks: int = 5,
        window_seconds: float = 60.0,
        cooldown_seconds: float = 300.0,
        backoff_factor: float = 1.5,
        max_backoff_seconds: float = 3600.0,
    ):
        """
        Args:
            max_blocks: 窗口内触发熔断的 BLOCK 事件数阈值。
            window_seconds: 滑动窗口时长（秒）。
            cooldown_seconds: OPEN 状态冷却时间（秒）。每次从 HALF_OPEN 回到 OPEN
                              时乘以 backoff_factor。
            backoff_factor: 指数退避因子（>=1.0）。
            max_backoff_seconds: 冷却时间上限（秒）。
        """
        if max_blocks < 1:
            raise ValueError(f"max_blocks must be >= 1, got {max_blocks}")
        if backoff_factor < 1.0:
            raise ValueError(f"backoff_factor must be >= 1.0, got {backoff_factor}")

        self.max_blocks = max_blocks
        self.window_seconds = window_seconds
        self._original_cooldown = cooldown_seconds
        self._current_cooldown = cooldown_seconds
        self._backoff_factor = backoff_factor
        self._max_backoff = max_backoff_seconds

        self._blocks: deque[_BlockRecord] = deque()
        self._state = BreakerState.CLOSED
        self._state_changed_at: float = 0.0
        self._tripped_count: int = 0  # 累计熔断次数（监控用）
        self._half_open_probe_done: bool = False

    # ── 公共 API ──────────────────────────────────

    @property
    def state(self) -> BreakerState:
        """当前状态。"""
        self._evolve_state()
        return self._state

    def is_tripped(self) -> bool:
        """检查熔断器是否处于 OPEN 状态（拒绝所有调用）。

        注意：HALF_OPEN 状态返回 False（允许试探）。
        """
        self._evolve_state()
        return self._state == BreakerState.OPEN

    def record_block(self, tool_name: str = "", rule_id: str = ""):
        """记录一次 BLOCK 事件。

        在 CLOSED 状态：添加到滑动窗口。若窗口内事件数 >= max_blocks，
        转入 OPEN 状态。
        在 HALF_OPEN 状态：试探失败，转入 OPEN 并应用指数退避。
        """
        now = time.monotonic()
        self._evolve_state()

        if self._state == BreakerState.CLOSED:
            self._blocks.append(_BlockRecord(
                tool_name=tool_name, rule_id=rule_id, timestamp=now,
            ))
            self._evict_expired()
            if len(self._blocks) >= self.max_blocks:
                self._transition_to(BreakerState.OPEN, now)
                logger.warning(
                    "[SecurityCircuitBreaker] 熔断触发: %d 次 BLOCK / %.0fs → OPEN",
                    len(self._blocks), self.window_seconds,
                )

        elif self._state == BreakerState.HALF_OPEN:
            # 试探失败 → 回到 OPEN，增加冷却时间
            self._current_cooldown = min(
                self._current_cooldown * self._backoff_factor,
                self._max_backoff,
            )
            self._transition_to(BreakerState.OPEN, now)
            logger.warning(
                "[SecurityCircuitBreaker] HALF_OPEN 试探失败 → OPEN "
                "(cooldown=%.0fs, backoff=×%.1f)",
                self._current_cooldown, self._backoff_factor,
            )

        # OPEN 状态下不记录（已被 is_tripped() 拦截，不应到达此处）

    def record_allow(self):
        """记录一次 ALLOW 事件（仅在 HALF_OPEN 状态有意义）。

        HALF_OPEN 状态下试探成功 → 恢复 CLOSED。
        """
        now = time.monotonic()
        self._evolve_state()

        if self._state == BreakerState.HALF_OPEN:
            self.reset()
            logger.info(
                "[SecurityCircuitBreaker] HALF_OPEN 试探成功 → CLOSED（恢复正常）"
            )

    def reset(self):
        """手动重置熔断器到 CLOSED 状态。"""
        self._blocks.clear()
        self._state = BreakerState.CLOSED
        self._state_changed_at = time.monotonic()
        self._current_cooldown = self._original_cooldown
        self._half_open_probe_done = False

    # ── 查询属性 ──────────────────────────────────

    @property
    def block_count(self) -> int:
        """当前窗口内的 BLOCK 事件数量。"""
        self._evict_expired()
        return len(self._blocks)

    @property
    def tripped_count(self) -> int:
        """累计熔断次数（监控指标）。"""
        return self._tripped_count

    @property
    def current_cooldown(self) -> float:
        """当前冷却时间（可能因 backoff 而增长）。"""
        return self._current_cooldown

    # ── 内部方法 ──────────────────────────────────

    def _evolve_state(self):
        """根据时间和条件演化状态。"""
        now = time.monotonic()

        if self._state == BreakerState.CLOSED:
            self._evict_expired()

        elif self._state == BreakerState.OPEN:
            elapsed = now - self._state_changed_at
            if elapsed >= self._current_cooldown:
                self._transition_to(BreakerState.HALF_OPEN, now)
                logger.info(
                    "[SecurityCircuitBreaker] OPEN → HALF_OPEN (冷却 %.0fs 结束)",
                    self._current_cooldown,
                )

        # HALF_OPEN 不需要时间驱动的状态转换
        # 由 record_block() / record_allow() 驱动

    def _evict_expired(self):
        """移除窗口外的过期 BLOCK 记录。"""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        while self._blocks and self._blocks[0].timestamp < cutoff:
            self._blocks.popleft()

    def _transition_to(self, new_state: BreakerState, now: float):
        """执行状态转换。"""
        old = self._state
        self._state = new_state
        self._state_changed_at = now
        if new_state == BreakerState.OPEN:
            self._tripped_count += 1
        elif new_state == BreakerState.CLOSED:
            self._blocks.clear()
            self._current_cooldown = self._original_cooldown
        logger.debug(
            "[SecurityCircuitBreaker] %s → %s (blocks=%d, cooldown=%.0fs)",
            old.value, new_state.value, len(self._blocks), self._current_cooldown,
        )
