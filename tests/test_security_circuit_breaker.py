"""
SecurityCircuitBreaker 三态熔断器测试。

覆盖：
    - CLOSED → OPEN → HALF_OPEN → CLOSED 全状态转换
    - HALF_OPEN → OPEN 指数退避
    - 滑动窗口过期驱逐
    - 手动 reset
"""

import sys
sys.path.insert(0, '.')
import time
import pytest
from pyagent.harness.context.security_circuit_breaker import (
    SecurityCircuitBreaker, BreakerState,
)


class TestSecurityCircuitBreaker:
    """熔断器状态机行为。"""

    @pytest.fixture
    def cb(self):
        return SecurityCircuitBreaker(max_blocks=3, window_seconds=60.0,
                                       cooldown_seconds=0.5, backoff_factor=1.5,
                                       max_backoff_seconds=3600.0)

    def test_not_tripped_initially(self, cb):
        assert cb.state == BreakerState.CLOSED
        assert not cb.is_tripped()
        assert cb.block_count == 0
        assert cb.tripped_count == 0

    def test_stays_closed_below_threshold(self, cb):
        cb.record_block("tool_a", "rule_1")
        cb.record_block("tool_b", "rule_2")
        assert cb.state == BreakerState.CLOSED
        assert cb.block_count == 2

    def test_tripped_after_max_blocks(self, cb):
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        cb.record_block("c", "r3")
        assert cb.state == BreakerState.OPEN
        assert cb.is_tripped()
        assert cb.tripped_count == 1

    def test_open_to_half_open_after_cooldown(self, cb):
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        cb.record_block("c", "r3")
        assert cb.state == BreakerState.OPEN
        time.sleep(0.6)
        assert cb.state == BreakerState.HALF_OPEN
        assert not cb.is_tripped()

    def test_half_open_allow_returns_to_closed(self, cb):
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        cb.record_block("c", "r3")
        time.sleep(0.6)
        assert cb.state == BreakerState.HALF_OPEN
        cb.record_allow()
        assert cb.state == BreakerState.CLOSED
        assert cb.block_count == 0

    def test_half_open_block_back_to_open_with_backoff(self, cb):
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        cb.record_block("c", "r3")
        original_cooldown = cb.current_cooldown
        time.sleep(0.6)
        assert cb.state == BreakerState.HALF_OPEN
        cb.record_block("d", "r4")
        assert cb.state == BreakerState.OPEN
        assert cb.current_cooldown == original_cooldown * 1.5

    def test_backoff_capped_at_max(self, cb):
        cb._max_backoff = 1.0
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        cb.record_block("c", "r3")
        time.sleep(0.6)
        cb.record_block("d", "r4")
        assert cb.current_cooldown <= cb._max_backoff

    def test_manual_reset(self, cb):
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        cb.record_block("c", "r3")
        assert cb.state == BreakerState.OPEN
        cb.reset()
        assert cb.state == BreakerState.CLOSED
        assert cb.block_count == 0
        assert cb.tripped_count == 1  # reset doesn't clear tripped_count

    def test_expired_blocks_dont_trip(self, cb):
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        time.sleep(0.15)  # wait more than window (we need to make window tiny)
        # Use a short-window breaker for this test
        cb2 = SecurityCircuitBreaker(max_blocks=2, window_seconds=0.05,
                                      cooldown_seconds=0.5)
        cb2.record_block("a", "r1")
        time.sleep(0.1)
        assert cb2.block_count == 0
        assert cb2.state == BreakerState.CLOSED

    def test_rejects_invalid_max_blocks(self):
        with pytest.raises(ValueError, match="max_blocks"):
            SecurityCircuitBreaker(max_blocks=0)
        with pytest.raises(ValueError, match="backoff_factor"):
            SecurityCircuitBreaker(max_blocks=3, backoff_factor=0.5)

    def test_is_tripped_only_true_in_open(self, cb):
        assert not cb.is_tripped()
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        cb.record_block("c", "r3")
        assert cb.is_tripped()
        time.sleep(0.6)
        assert cb.state == BreakerState.HALF_OPEN
        assert not cb.is_tripped()  # HALF_OPEN allows probes

    def test_multiple_open_transitions_increment_count(self, cb):
        cb.record_block("a", "r1")
        cb.record_block("b", "r2")
        cb.record_block("c", "r3")
        assert cb.tripped_count == 1
        time.sleep(0.6)
        cb.record_block("d", "r4")
        assert cb.tripped_count == 2
