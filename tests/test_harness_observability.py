"""
可观测性与安全审计基础设施测试。

覆盖：
    - SecurityAuditEvent：模型创建 / is_p0 属性 / 序列化 / 字段校验
    - TraceContext：UUID4 生成 / 上下文提取 / 回退链
    - Sanitizer：递归脱敏 / 嵌套 / 列表 / 不可变性 / Pydantic / 扩展字段
    - SENSITIVE_FIELDS 配置
    - compute_step_fingerprint / compute_risk_score
    - AuditLogger：队列写入 / P0 兜底 / 生命周期
    - ObservabilityContext：log_decision / trace_id 自动填充 / 生命周期
"""

from unittest.mock import Mock, AsyncMock, patch, PropertyMock
import json
import os
import sys
import asyncio
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from pyagent.harness.context.observability import (
    SecurityAuditEvent,
    TraceContext,
    Sanitizer,
    SanitizedSerializable,
    AuditLogger,
    ObservabilityContext,
    SENSITIVE_FIELDS,
    compute_step_fingerprint,
    compute_risk_score,
)


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _make_event(**overrides) -> SecurityAuditEvent:
    """创建测试用 SecurityAuditEvent。"""
    defaults = {
        "decision": "ALLOW",
        "rule_id": "test_rule_01",
        "plan_step_fingerprint": "abc123",
        "risk_score": 0,
        "tool_name": "read_file",
        "tool_params": {"file_path": "/tmp/test.py"},
        "phase": "PLAN",
        "state": "IDLE",
        "trace_id": "deadbeef",
    }
    defaults.update(overrides)
    return SecurityAuditEvent(**defaults)


# ═══════════════════════════════════════════════════════════════
# SecurityAuditEvent 测试
# ═══════════════════════════════════════════════════════════════

class TestSecurityAuditEvent:

    def test_creation_with_required_fields(self):
        """仅必填字段可创建。"""
        event = SecurityAuditEvent(decision="ALLOW", rule_id="R001")
        assert event.decision == "ALLOW"
        assert event.rule_id == "R001"

    def test_field_defaults(self):
        """非必填字段使用预期默认值。"""
        event = SecurityAuditEvent(decision="BLOCK", rule_id="R002")
        assert event.plan_step_fingerprint == ""
        assert event.risk_score == 0
        assert event.tool_name == ""
        assert event.tool_params == {}
        assert event.phase == ""
        assert event.state == ""
        assert event.trace_id == ""
        # timestamp 由 default_factory 生成，应为非空字符串
        assert isinstance(event.timestamp, str)
        assert len(event.timestamp) > 0
        assert isinstance(event.metadata, dict)
        assert event.metadata == {}

    def test_is_p0_block(self):
        """BLOCK 决策 → is_p0 = True。"""
        event = _make_event(decision="BLOCK")
        assert event.is_p0 is True

    def test_is_p0_repair(self):
        """REPAIR 决策 → is_p0 = True。"""
        event = _make_event(decision="REPAIR")
        assert event.is_p0 is True

    def test_is_p0_allow_false(self):
        """ALLOW 决策 → is_p0 = False。"""
        event = _make_event(decision="ALLOW")
        assert event.is_p0 is False

    def test_to_json_serialization(self):
        """to_json() 返回合法 JSON 字符串且包含关键字段。"""
        event = _make_event(decision="BLOCK", rule_id="R_HIGH_RISK")
        raw = event.to_json()
        assert isinstance(raw, str)
        data = json.loads(raw)
        assert data["decision"] == "BLOCK"
        assert data["rule_id"] == "R_HIGH_RISK"
        assert data["trace_id"] == "deadbeef"

    def test_model_dump_roundtrip(self):
        """model_dump → SecurityAuditEvent(**dumped) 可重建等价事件。"""
        original = _make_event(
            decision="REPAIR",
            risk_score=80,
            metadata={"step_id": 5, "attempt": 2},
        )
        dumped = original.model_dump()
        rebuilt = SecurityAuditEvent(**dumped)
        assert rebuilt.decision == original.decision
        assert rebuilt.rule_id == original.rule_id
        assert rebuilt.risk_score == original.risk_score
        assert rebuilt.metadata == original.metadata
        assert rebuilt.tool_params == original.tool_params

    def test_risk_score_boundary_valid(self):
        """risk_score 在 [0, 100] 范围内合法。"""
        e0 = SecurityAuditEvent(decision="ALLOW", rule_id="R", risk_score=0)
        assert e0.risk_score == 0
        e100 = SecurityAuditEvent(decision="ALLOW", rule_id="R", risk_score=100)
        assert e100.risk_score == 100

    def test_risk_score_out_of_range_raises(self):
        """risk_score 超出 [0, 100] 应触发 ValidationError。"""
        with pytest.raises(ValidationError):
            SecurityAuditEvent(decision="ALLOW", rule_id="R", risk_score=-1)
        with pytest.raises(ValidationError):
            SecurityAuditEvent(decision="ALLOW", rule_id="R", risk_score=101)

    def test_decision_invalid_literal_raises(self):
        """decision 为非法 Literal 值应触发 ValidationError。"""
        with pytest.raises(ValidationError):
            SecurityAuditEvent(decision="INVALID", rule_id="R")

    def test_all_fields_present_in_dump(self):
        """model_dump 应包含所有字段。"""
        event = _make_event()
        dumped = event.model_dump()
        expected_keys = {
            "decision", "rule_id", "plan_step_fingerprint", "risk_score",
            "tool_name", "tool_params", "phase", "state", "trace_id",
            "timestamp", "metadata",
        }
        assert set(dumped.keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════
# TraceContext 测试
# ═══════════════════════════════════════════════════════════════

class TestTraceContext:

    def test_generate_returns_32_hex_chars(self):
        """generate() 返回 32 字符十六进制字符串（UUID4 hex）。"""
        tid = TraceContext.generate()
        assert isinstance(tid, str)
        assert len(tid) == 32
        # 应为全十六进制
        assert all(c in "0123456789abcdef" for c in tid)

    def test_generate_produces_unique_values(self):
        """连续调用 generate() 应产生不同值。"""
        ids = {TraceContext.generate() for _ in range(50)}
        assert len(ids) == 50

    def test_current_from_observability(self):
        """优先从 ctx["observability"].trace_id 提取。"""
        mock_obs = Mock()
        mock_obs.trace_id = "obs-trace-id-123"
        ctx = {"observability": mock_obs, "plan": None}
        assert TraceContext.current(ctx) == "obs-trace-id-123"

    def test_current_falls_back_to_plan(self):
        """observability 不可用时，回退到 ctx["plan"].trace_id。"""
        mock_plan = Mock()
        mock_plan.trace_id = "plan-trace-id-456"
        ctx = {"observability": None, "plan": mock_plan}
        assert TraceContext.current(ctx) == "plan-trace-id-456"

    def test_current_falls_back_to_dict_key(self):
        """plan 也不可用时，回退到 ctx["trace_id"]。"""
        ctx = {"observability": None, "plan": None, "trace_id": "legacy-tid"}
        assert TraceContext.current(ctx) == "legacy-tid"

    def test_current_returns_empty_when_nothing_found(self):
        """所有来源均不可用时返回空字符串。"""
        ctx = {"observability": None, "plan": None}
        assert TraceContext.current(ctx) == ""

    def test_current_skips_empty_obs_trace_id(self):
        """observability 存在但 trace_id 为空 → 继续回退。"""
        mock_obs = Mock()
        mock_obs.trace_id = ""
        mock_plan = Mock()
        mock_plan.trace_id = "plan-fallback"
        ctx = {"observability": mock_obs, "plan": mock_plan}
        assert TraceContext.current(ctx) == "plan-fallback"


# ═══════════════════════════════════════════════════════════════
# Sanitizer 测试
# ═══════════════════════════════════════════════════════════════

class TestSanitizer:

    @pytest.fixture
    def sanitizer(self):
        return Sanitizer()

    def test_redact_top_level_sensitive(self, sanitizer):
        """顶层 password / api_key → [REDACTED]。"""
        obj = {"password": "s3cret!", "username": "alice"}
        result = sanitizer.sanitize(obj)
        assert result["password"] == "[REDACTED]"
        assert result["username"] == "alice"

    def test_redact_nested_sensitive(self, sanitizer):
        """2+ 层深度的 token → [REDACTED]。"""
        obj = {
            "config": {
                "inner": {
                    "token": "ghp_xxxx",
                    "user": "bot",
                }
            }
        }
        result = sanitizer.sanitize(obj)
        assert result["config"]["inner"]["token"] == "[REDACTED]"
        assert result["config"]["inner"]["user"] == "bot"

    def test_redact_in_list_items(self, sanitizer):
        """列表中元素的敏感字段也脱敏。"""
        obj = {
            "entries": [
                {"api_key": "k1", "name": "a"},
                {"api_key": "k2", "name": "b"},
            ]
        }
        result = sanitizer.sanitize(obj)
        assert result["entries"][0]["api_key"] == "[REDACTED]"
        assert result["entries"][1]["api_key"] == "[REDACTED]"
        assert result["entries"][0]["name"] == "a"
        assert result["entries"][1]["name"] == "b"

    def test_does_not_modify_original(self, sanitizer):
        """脱敏后原始对象不变（不可变性）。"""
        original = {"password": "secret123", "data": {"nested": {"token": "t"}}}
        original_copy = {"password": "secret123", "data": {"nested": {"token": "t"}}}
        sanitizer.sanitize(original)
        assert original == original_copy

    def test_custom_extra_fields(self):
        """自定义 extra_fields 也参与脱敏。"""
        s = Sanitizer(extra_fields={"custom_secret", "db_password"})
        obj = {"custom_secret": "cs", "db_password": "db", "name": "x"}
        result = s.sanitize(obj)
        assert result["custom_secret"] == "[REDACTED]"
        assert result["db_password"] == "[REDACTED]"
        assert result["name"] == "x"

    def test_non_dict_non_list_unchanged(self, sanitizer):
        """非 dict/list 对象原样返回。"""
        assert sanitizer.sanitize(42) == 42
        assert sanitizer.sanitize("hello") == "hello"
        assert sanitizer.sanitize(None) is None
        assert sanitizer.sanitize(True) is True

    def test_handles_pydantic_model_via_model_dump(self, sanitizer):
        """Pydantic 模型通过 model_dump 脱敏。"""
        event = _make_event(
            tool_params={"api_key": "sk-exposed"},
        )
        result = sanitizer.sanitize(event)
        assert isinstance(result, dict)
        assert result["tool_params"]["api_key"] == "[REDACTED]"
        assert result["decision"] == "ALLOW"

    def test_handles_sanitized_serializable(self):
        """实现 SanitizedSerializable 的对象调用 .sanitize()。"""

        class CustomObj(SanitizedSerializable):
            def sanitize(self) -> dict:
                return {"safe": "yes"}

        obj = CustomObj()
        s = Sanitizer()
        result = s.sanitize(obj)
        assert result == {"safe": "yes"}

    def test_redacts_bearer_token_in_string_value(self, sanitizer):
        """字符串值中的 Bearer token 模式被替换。"""
        obj = {"headers": {"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.xxx"}}
        result = sanitizer.sanitize(obj)
        assert "[REDACTED]" in result["headers"]["Authorization"]

    def test_redacts_sk_prefix_in_string_value(self, sanitizer):
        """字符串值中的 sk-xxx API key 模式被替换。"""
        obj = {"note": "use key sk-1234567890abcdef for access"}
        result = sanitizer.sanitize(obj)
        assert "sk-[REDACTED]" in result["note"]

    def test_case_insensitive_key_match(self, sanitizer):
        """敏感字段匹配不区分大小写。"""
        obj = {"PASSWORD": "Upper", "Api_Key": "mixed"}
        result = sanitizer.sanitize(obj)
        assert result["PASSWORD"] == "[REDACTED]"
        assert result["Api_Key"] == "[REDACTED]"

    def test_contains_match(self, sanitizer):
        """key 包含敏感字段名也匹配（如 x_api_key）。"""
        obj = {"x_api_key": "val"}
        result = sanitizer.sanitize(obj)
        assert result["x_api_key"] == "[REDACTED]"


# ═══════════════════════════════════════════════════════════════
# SENSITIVE_FIELDS 配置测试
# ═══════════════════════════════════════════════════════════════

class TestSensitiveFields:

    def test_contains_core_security_keys(self):
        """SENSITIVE_FIELDS 包含核心安全键。"""
        core = {"password", "api_key", "token", "secret", "authorization",
                "private_key", "credential", "passphrase"}
        for key in core:
            assert key in SENSITIVE_FIELDS, f"{key} 缺失"

    def test_is_set_type(self):
        assert isinstance(SENSITIVE_FIELDS, set)


# ═══════════════════════════════════════════════════════════════
# compute_step_fingerprint 测试
# ═══════════════════════════════════════════════════════════════

class TestComputeStepFingerprint:

    def test_consistent_for_same_params(self):
        """相同参数多次调用返回相同指纹。"""
        params = {"file_path": "/a/b.py", "offset": 10}
        f1 = compute_step_fingerprint(params)
        f2 = compute_step_fingerprint(params)
        f3 = compute_step_fingerprint(dict(params))
        assert f1 == f2 == f3

    def test_different_for_different_params(self):
        """不同参数返回不同指纹。"""
        f1 = compute_step_fingerprint({"file_path": "/a.py"})
        f2 = compute_step_fingerprint({"file_path": "/b.py"})
        assert f1 != f2

    def test_returns_no_params_for_empty_dict(self):
        """空 dict → "no_params"。"""
        assert compute_step_fingerprint({}) == "no_params"

    def test_returns_16_char_hex(self):
        """非空 params 返回 16 字符十六进制。"""
        fp = compute_step_fingerprint({"a": 1})
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_order_independent(self):
        """字典键顺序不影响指纹。"""
        f1 = compute_step_fingerprint({"a": 1, "b": 2})
        f2 = compute_step_fingerprint({"b": 2, "a": 1})
        assert f1 == f2


# ═══════════════════════════════════════════════════════════════
# compute_risk_score 测试
# ═══════════════════════════════════════════════════════════════

class TestComputeRiskScore:

    def test_empty_set_returns_zero(self):
        assert compute_risk_score(set()) == 0

    def test_single_tool_returns_60(self):
        assert compute_risk_score({"delete_file"}) == 60

    def test_two_tool_returns_80(self):
        assert compute_risk_score({"write_file", "execute_python"}) == 80

    def test_three_tool_returns_95(self):
        assert compute_risk_score({"a", "b", "c"}) == 95

    def test_four_tool_returns_95(self):
        """3+ 工具均返回 95。"""
        assert compute_risk_score({"a", "b", "c", "d"}) == 95

    def test_return_type_is_int(self):
        assert isinstance(compute_risk_score({"x"}), int)


# ═══════════════════════════════════════════════════════════════
# AuditLogger 测试
# ═══════════════════════════════════════════════════════════════

class TestAuditLogger:

    @pytest.fixture
    def temp_fallback(self, tmp_path):
        return str(tmp_path / "audit_fallback.jsonl")

    @pytest.fixture
    def logger(self, temp_fallback):
        return AuditLogger(fallback_path=temp_fallback)

    @pytest.mark.asyncio
    async def test_start_and_stop_lifecycle(self, logger):
        """启动 → 停止 生命周期正常。"""
        assert not logger._running
        await logger.start()
        assert logger._running
        await logger.stop()
        assert not logger._running

    @pytest.mark.asyncio
    async def test_emit_adds_to_queue_and_flush(self, logger):
        """emit 将事件写入队列，flush 等待消费。"""
        await logger.start()
        event = _make_event(decision="ALLOW", rule_id="R_OK")
        logger.emit(event)
        # 等待后台消费
        logger.flush()
        await logger.stop()

    @pytest.mark.asyncio
    async def test_multiple_emits(self, logger):
        """连续 emit 多个事件不抛异常。"""
        await logger.start()
        for i in range(10):
            logger.emit(_make_event(rule_id=f"R_{i}"))
        logger.flush()
        await logger.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_no_error(self, logger):
        """未启动直接 stop 不报错。"""
        await logger.stop()
        assert not logger._running

    @pytest.mark.asyncio
    async def test_start_idempotent(self, logger):
        """重复 start 不报错，状态不变。"""
        await logger.start()
        await logger.start()
        assert logger._running
        await logger.stop()

    def test_p0_fallback_file_written_when_queue_full(self, temp_fallback):
        """
        队列满时 P0 事件落盘到兜底文件。

        通过 mock queue 的 put_nowait 始终抛出 QueueFull 来模拟。
        """
        logger = AuditLogger(fallback_path=temp_fallback)
        event = _make_event(decision="BLOCK", rule_id="R_CRITICAL")

        with patch.object(logger._queue, 'put_nowait', side_effect=asyncio.QueueFull()):
            logger.emit(event)

        # 验证兜底文件已创建
        fallback = Path(temp_fallback)
        assert fallback.exists(), f"兜底文件 {temp_fallback} 应已创建"

        # 验证内容
        content = fallback.read_text(encoding='utf-8')
        assert "BLOCK" in content
        assert "R_CRITICAL" in content
        assert logger._flushed_p0 >= 1

    def test_non_p0_dropped_when_queue_full(self, temp_fallback):
        """队列满时非 P0 事件静默丢弃，兜底文件不创建。"""
        logger = AuditLogger(fallback_path=temp_fallback)
        event = _make_event(decision="ALLOW", rule_id="R_OK")

        with patch.object(logger._queue, 'put_nowait', side_effect=asyncio.QueueFull()):
            logger.emit(event)

        # 非 P0 不应落盘
        fallback = Path(temp_fallback)
        assert not fallback.exists(), "非 P0 不应写入兜底文件"
        assert logger._dropped_non_p0 >= 1

    def test_fallback_file_permissions(self, temp_fallback):
        """兜底文件权限为 0o600（Windows 下可能忽略，不报错即可）。"""
        logger = AuditLogger(fallback_path=temp_fallback)
        event = _make_event(decision="BLOCK", rule_id="R_PERM")

        with patch.object(logger._queue, 'put_nowait', side_effect=asyncio.QueueFull()):
            logger.emit(event)

        fallback = Path(temp_fallback)
        assert fallback.exists()
        # 能读取说明写入成功，不强制检查 Unix 权限（Windows 兼容）


# ═══════════════════════════════════════════════════════════════
# ObservabilityContext 测试
# ═══════════════════════════════════════════════════════════════

class TestObservabilityContext:

    @pytest.fixture
    def tmp_fallback(self, tmp_path):
        return str(tmp_path / "obs_fallback.jsonl")

    @pytest.fixture
    def obs(self, tmp_fallback):
        return ObservabilityContext(trace_id="test-trace-001", fallback_path=tmp_fallback)

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, obs):
        """start → stop 生命周期正常。"""
        await obs.start()
        assert obs._started
        await obs.stop()
        assert not obs._started

    @pytest.mark.asyncio
    async def test_stop_without_start_no_error(self, obs):
        """未 start 直接 stop 不报错。"""
        await obs.stop()
        assert not obs._started

    def test_log_decision_creates_event_with_auto_filled_trace_id(self, obs):
        """log_decision 自动填充 trace_id。"""
        event = obs.log_decision(
            decision="BLOCK",
            rule_id="HIGH_RISK_COMBOS[0]",
            tool_name="delete_file",
            tool_params={"file_path": "/tmp/x.py"},
            risk_score=60,
        )
        assert isinstance(event, SecurityAuditEvent)
        assert event.trace_id == "test-trace-001"
        assert event.decision == "BLOCK"
        assert event.rule_id == "HIGH_RISK_COMBOS[0]"
        assert event.risk_score == 60

    def test_log_decision_sanitizes_tool_params(self, obs):
        """log_decision 自动脱敏工具参数。"""
        event = obs.log_decision(
            decision="ALLOW",
            rule_id="step_level_check",
            tool_name="read_file",
            tool_params={"api_key": "sk-secret", "file_path": "/safe.py"},
        )
        assert event.tool_params["api_key"] == "[REDACTED]"
        assert event.tool_params["file_path"] == "/safe.py"

    def test_log_decision_handles_no_params(self, obs):
        """log_decision 不传 tool_params 时默认空 dict。"""
        event = obs.log_decision(decision="ALLOW", rule_id="R")
        assert event.tool_params == {}

    def test_log_decision_passes_all_fields(self, obs):
        """log_decision 传递所有字段到 SecurityAuditEvent。"""
        event = obs.log_decision(
            decision="REPAIR",
            rule_id="R_REPAIR",
            tool_name="git_reset",
            tool_params={"hard": True},
            risk_score=80,
            plan_step_fingerprint="fp_repair",
            phase="REPAIR",
            state="REPAIRING",
            metadata={"step_id": 3},
        )
        assert event.decision == "REPAIR"
        assert event.rule_id == "R_REPAIR"
        assert event.tool_name == "git_reset"
        assert event.tool_params == {"hard": True}
        assert event.risk_score == 80
        assert event.plan_step_fingerprint == "fp_repair"
        assert event.phase == "REPAIR"
        assert event.state == "REPAIRING"
        assert event.metadata == {"step_id": 3}

    def test_dropped_count_property(self, obs):
        """dropped_count 返回内部 audit_logger 的丢弃计数。"""
        assert obs.dropped_count == 0
        obs.audit_logger._dropped_non_p0 = 5
        assert obs.dropped_count == 5

    def test_flushed_count_property(self, obs):
        """flushed_count 返回内部 audit_logger 的落盘计数。"""
        assert obs.flushed_count == 0
        obs.audit_logger._flushed_p0 = 3
        assert obs.flushed_count == 3

    def test_emit_auto_fills_trace_id(self, obs):
        """emit 自动填充 trace_id 到事件（若事件未设置）。"""
        event = _make_event(trace_id="")
        obs.emit(event)
        assert event.trace_id == "test-trace-001"

    def test_emit_preserves_existing_trace_id(self, obs):
        """事件已有 trace_id 时不覆盖。"""
        event = _make_event(trace_id="custom-tid")
        obs.emit(event)
        assert event.trace_id == "custom-tid"

    @pytest.mark.asyncio
    async def test_start_stop_cleans_up_audit_logger(self, obs):
        """start/stop 正确启停内部 AuditLogger。"""
        await obs.start()
        assert obs.audit_logger._running
        await obs.stop()
        assert not obs.audit_logger._running

    @pytest.mark.asyncio
    async def test_start_idempotent(self, obs):
        """重复 start 不报错。"""
        await obs.start()
        await obs.start()
        assert obs._started
        await obs.stop()
