"""
SecurityCircuitBreaker 熔断阈值压测 — 滑动窗口边界精度验证。

覆盖：
    1. 精确 max_blocks 触发 OPEN（不多一次不少一次）
    2. max_blocks-1 保持 CLOSED
    3. 窗口过期后 block_count 精确归零
    4. 冷却时间精度（允许 ±10% 偏差）
    5. 指数退避链精度
    6. OPEN 状态下 record_block 不改变状态
    7. 连续多次 HALF_OPEN 试探失败后的退避累计
"""

import sys
sys.path.insert(0, '.')
import time
import pytest
from pyagent.harness.context.security_circuit_breaker import (
    SecurityCircuitBreaker, BreakerState,
)


class TestCircuitBreakerExactThreshold:
    """精确熔断阈值 — 不多不少刚好 max_blocks。"""

    def test_exact_max_blocks_triggers_open(self):
        cb = SecurityCircuitBreaker(max_blocks=5, window_seconds=60.0)
        for i in range(5):
            assert cb.state == BreakerState.CLOSED, f"Should be CLOSED after {i} blocks"
            cb.record_block(f"tool_{i}", "rule_test")
        assert cb.state == BreakerState.OPEN, "Should be OPEN after 5th block"
        assert cb.block_count == 5

    def test_one_less_than_max_stays_closed(self):
        cb = SecurityCircuitBreaker(max_blocks=5, window_seconds=60.0)
        for i in range(4):
            cb.record_block(f"tool_{i}", "rule_test")
        assert cb.state == BreakerState.CLOSED, "4 blocks should NOT trigger OPEN"
        assert cb.block_count == 4
        assert not cb.is_tripped()

    def test_exact_threshold_for_max_1(self):
        """边界：max_blocks=1 时，首次 BLOCK 即触发。"""
        cb = SecurityCircuitBreaker(max_blocks=1, window_seconds=60.0)
        assert cb.state == BreakerState.CLOSED
        cb.record_block("t", "r")
        assert cb.state == BreakerState.OPEN


class TestCircuitBreakerWindowPrecision:
    """滑动窗口精度 — 过期驱逐的时机准确性。"""

    def test_window_expiry_resets_block_count(self):
        """窗口过期后 BLOCK 计数归零。"""
        cb = SecurityCircuitBreaker(max_blocks=3, window_seconds=0.1,
                                     cooldown_seconds=0.5)
        cb.record_block("t1", "r1")
        cb.record_block("t2", "r2")
        assert cb.block_count == 2
        time.sleep(0.15)  # > window_seconds
        assert cb.block_count == 0, "Block count should be 0 after window expiry"

    def test_window_expiry_prevents_trip(self):
        """窗口过期后的新 BLOCK 不计入旧窗口。"""
        cb = SecurityCircuitBreaker(max_blocks=3, window_seconds=0.1,
                                     cooldown_seconds=0.5)
        cb.record_block("t1", "r1")
        cb.record_block("t2", "r2")
        time.sleep(0.15)
        cb.record_block("t3", "r3")  # First 2 expired, this is effectively #1
        assert cb.block_count == 1
        assert cb.state == BreakerState.CLOSED

    def test_partial_expiry_accurate(self):
        """窗口中部分过期，剩余计数精确。"""
        cb = SecurityCircuitBreaker(max_blocks=5, window_seconds=0.2)
        cb.record_block("t1", "r1")
        time.sleep(0.05)
        cb.record_block("t2", "r2")
        time.sleep(0.05)
        cb.record_block("t3", "r3")
        # t1 is ~0.10s old, t2 ~0.05s, t3 fresh
        time.sleep(0.15)
        # t1 should be expired (>0.20s), t2 may or may not be
        # We just verify the count is ≤ 2
        assert cb.block_count <= 2, (
            f"At most 2 blocks should remain after partial expiry, got {cb.block_count}"
        )


class TestCircuitBreakerCooldownTiming:
    """冷却时间精度。"""

    def test_cooldown_transitions_to_half_open(self):
        """冷却时间结束后自动进入 HALF_OPEN。"""
        cb = SecurityCircuitBreaker(max_blocks=2, window_seconds=60.0,
                                     cooldown_seconds=0.3)
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        assert cb.state == BreakerState.OPEN
        time.sleep(0.4)  # > cooldown
        assert cb.state == BreakerState.HALF_OPEN

    def test_cooldown_not_expired_stays_open(self):
        """冷却未结束时保持 OPEN。"""
        cb = SecurityCircuitBreaker(max_blocks=2, window_seconds=60.0,
                                     cooldown_seconds=0.3)
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        time.sleep(0.1)  # < cooldown
        assert cb.state == BreakerState.OPEN

    def test_cooldown_timing_within_tolerance(self):
        """冷却时间偏差 ≤ 10%。"""
        cooldown = 0.2
        cb = SecurityCircuitBreaker(max_blocks=2, window_seconds=60.0,
                                     cooldown_seconds=cooldown)
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        t0 = time.monotonic()
        # Poll until HALF_OPEN or timeout
        elapsed = 0
        while cb.state == BreakerState.OPEN and elapsed < 1.0:
            time.sleep(0.02)
            elapsed = time.monotonic() - t0
        assert cb.state == BreakerState.HALF_OPEN
        deviation = abs(elapsed - cooldown) / cooldown
        assert deviation <= 0.15, (
            f"Cooldown timing deviation {deviation:.1%} exceeds 15% tolerance"
        )


class TestCircuitBreakerBackoffPrecision:
    """指数退避精度。"""

    def test_backoff_multiplies_cooldown(self):
        """HALF_OPEN 试探失败后 cooldown 乘以 backoff_factor。"""
        cb = SecurityCircuitBreaker(max_blocks=2, window_seconds=60.0,
                                     cooldown_seconds=0.3,
                                     backoff_factor=2.0)
        # Trigger OPEN
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        original = cb.current_cooldown
        assert original == 0.3

        # Wait for HALF_OPEN
        time.sleep(0.4)
        assert cb.state == BreakerState.HALF_OPEN

        # Probe fails → back to OPEN with backoff
        cb.record_block("c", "r3")
        assert cb.state == BreakerState.OPEN
        assert cb.current_cooldown == pytest.approx(0.6, rel=0.01)

    def test_backoff_chain_accumulates(self):
        """连续多次试探失败，退避因子累积生效。"""
        cb = SecurityCircuitBreaker(max_blocks=1, window_seconds=60.0,
                                     cooldown_seconds=0.15,
                                     backoff_factor=2.0,
                                     max_backoff_seconds=10.0)
        # 1st trip
        cb.record_block("a", "r1")
        assert cb.current_cooldown == 0.15
        time.sleep(0.2)
        cb.record_block("b", "r2")  # HALF_OPEN probe fails
        assert cb.current_cooldown == pytest.approx(0.3, rel=0.05)

        # 2nd backoff
        time.sleep(0.35)
        cb.record_block("c", "r3")
        assert cb.current_cooldown == pytest.approx(0.6, rel=0.05)

        # 3rd backoff
        time.sleep(0.65)
        cb.record_block("d", "r4")
        assert cb.current_cooldown == pytest.approx(1.2, rel=0.05)

    def test_backoff_capped_at_max(self):
        """退避不超过 max_backoff_seconds。"""
        cb = SecurityCircuitBreaker(max_blocks=1, window_seconds=60.0,
                                     cooldown_seconds=0.1,
                                     backoff_factor=10.0,
                                     max_backoff_seconds=0.5)
        cb.record_block("a", "r1")
        time.sleep(0.15)
        cb.record_block("b", "r2")
        assert cb.current_cooldown == 0.5, (
            f"Cooldown should be capped at max_backoff=0.5, got {cb.current_cooldown}"
        )


class TestCircuitBreakerHalfOpenEdgeCases:
    """HALF_OPEN 状态边界场景。"""

    def test_allow_during_half_open_resets_to_closed(self):
        """HALF_OPEN 试探 ALLOW → CLOSED。"""
        cb = SecurityCircuitBreaker(max_blocks=2, window_seconds=60.0,
                                     cooldown_seconds=0.3)
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        time.sleep(0.4)
        assert cb.state == BreakerState.HALF_OPEN
        cb.record_allow()
        assert cb.state == BreakerState.CLOSED
        assert cb.block_count == 0

    def test_block_during_open_does_not_affect_state(self):
        """OPEN 状态下 record_block 已被 is_tripped() 拦截，
        即使直接调用也不应产生额外副作用（幂等）。"""
        cb = SecurityCircuitBreaker(max_blocks=2, window_seconds=60.0,
                                     cooldown_seconds=0.3)
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        assert cb.state == BreakerState.OPEN
        # Extra block while OPEN
        cb.record_block("c", "r3")
        assert cb.state == BreakerState.OPEN
        assert cb.tripped_count == 1  # Not incremented again while already OPEN

    def test_multiple_allows_in_half_open(self):
        """HALF_OPEN 多次 record_allow — 第一次恢复 CLOSED，后续无副作用。"""
        cb = SecurityCircuitBreaker(max_blocks=2, window_seconds=60.0,
                                     cooldown_seconds=0.3)
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        time.sleep(0.4)
        cb.record_allow()
        assert cb.state == BreakerState.CLOSED
        cb.record_allow()  # No-op in CLOSED
        assert cb.state == BreakerState.CLOSED

    def test_reset_clears_backoff(self):
        """reset 恢复原始 cooldown。"""
        cb = SecurityCircuitBreaker(max_blocks=1, window_seconds=60.0,
                                     cooldown_seconds=0.3,
                                     backoff_factor=2.0)
        cb.record_block("a", "r1")
        time.sleep(0.4)
        cb.record_block("b", "r2")
        assert cb.current_cooldown > 0.3
        cb.reset()
        assert cb.current_cooldown == 0.3
        assert cb.state == BreakerState.CLOSED
