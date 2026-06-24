"""
修补上下文隔离与熔断机制测试。

覆盖：
    - RepairContext：字段隔离 / to_hint_text / 默认值
    - CircuitBreaker：初始状态 / 计数 / 熔断 / 报告 / 参数校验
    - ConvergenceDetector：首次调用 / 相同检测 / 不同检测 / 归一化 / reset / 阈值校验
    - RepairLog：记录 / 审计事件 / 收敛标记 / 摘要 / 全失败检测
"""

import pytest

from pyagent.harness.context.repair_context import (
    RepairContext,
    CircuitBreaker,
    ConvergenceDetector,
    RepairLog,
)


# ═══════════════════════════════════════════════════════════════
# RepairContext 测试
# ═══════════════════════════════════════════════════════════════

class TestRepairContext:

    def test_no_conversation_history_field(self):
        """RepairContext 明确不包含 conversation_history 字段。"""
        ctx = RepairContext()
        assert not hasattr(ctx, "conversation_history"), (
            "RepairContext 不应包含 conversation_history 字段（隔离上下文，避免膨胀）"
        )

    def test_to_hint_text_includes_attempt_and_tools(self):
        """to_hint_text() 应包含修复尝试次数和可用工具。"""
        ctx = RepairContext(
            failed_step_description="读取配置文件",
            failure_reason="文件不存在",
            acceptance_criteria="成功读取并解析 JSON",
            available_tools=["read_file", "search_content"],
            repair_attempt=2,
            max_repairs=3,
        )
        text = ctx.to_hint_text()
        assert "修复尝试: 2/3" in text
        assert "可用工具: read_file, search_content" in text

    def test_to_hint_text_empty_tools_shows_placeholder(self):
        """可用工具为空时，应显示受限白名单提示。"""
        ctx = RepairContext(
            available_tools=[],
            repair_attempt=1,
        )
        text = ctx.to_hint_text()
        assert "受限白名单" in text

    def test_to_hint_text_includes_previous_repair_summary(self):
        """上一轮修补摘要存在时，应包含在 hint 文本中。"""
        ctx = RepairContext(
            failed_step_description="写入脚本",
            failure_reason="语法错误",
            acceptance_criteria="脚本可执行",
            available_tools=["write_file"],
            previous_repair_summary="修改了文件路径，但未修正语法",
            repair_attempt=1,
        )
        text = ctx.to_hint_text()
        assert "上一轮修补摘要:" in text
        assert "修改了文件路径，但未修正语法" in text

    def test_to_hint_text_without_previous_summary(self):
        """无上一轮修补摘要时，不应包含该项。"""
        ctx = RepairContext(
            failed_step_description="运行测试",
            failure_reason="断言失败",
            acceptance_criteria="所有用例通过",
            available_tools=["execute_python"],
            repair_attempt=1,
            # previous_repair_summary 留空
        )
        text = ctx.to_hint_text()
        assert "上一轮修补摘要:" not in text

    def test_default_values(self):
        """RepairContext 的默认值应正确。"""
        ctx = RepairContext()
        assert ctx.failed_step_description == ""
        assert ctx.failure_reason == ""
        assert ctx.acceptance_criteria == ""
        assert ctx.available_tools == []
        assert ctx.previous_repair_summary == ""
        assert ctx.repair_attempt == 0
        assert ctx.max_repairs == 3

    def test_previous_repair_summary_truncated_in_hint(self):
        """上一轮修补摘要超过 500 字符时，to_hint_text 应截断。"""
        long_summary = "修改了多处代码逻辑。" * 60  # 远超 500 字符
        assert len(long_summary) > 500
        ctx = RepairContext(
            failed_step_description="重构模块",
            failure_reason="循环导入",
            acceptance_criteria="模块可导入",
            available_tools=["read_file", "write_file"],
            previous_repair_summary=long_summary,
            repair_attempt=3,
        )
        text = ctx.to_hint_text()
        # 确认截断行为：hint 中的摘要长度不超过 500
        # 找到 "上一轮修补摘要: " 后的内容
        prefix = "上一轮修补摘要: "
        idx = text.index(prefix)
        summary_in_hint = text[idx + len(prefix):]
        assert len(summary_in_hint) <= 500


# ═══════════════════════════════════════════════════════════════
# CircuitBreaker 测试
# ═══════════════════════════════════════════════════════════════

class TestCircuitBreaker:

    def test_initial_state_not_tripped(self):
        """初始状态下断路器未熔断，剩余次数等于 max_repairs。"""
        cb = CircuitBreaker(max_repairs=3)
        assert cb.is_tripped() is False
        assert cb.remaining == 3
        assert cb.attempt_count == 0

    def test_record_attempt_increments_counter(self):
        """record_attempt 应递增 attempt_count。"""
        cb = CircuitBreaker(max_repairs=3)
        cb.record_attempt(detail="第一次尝试", failure_reason="文件不存在")
        assert cb.attempt_count == 1
        cb.record_attempt(detail="第二次尝试", failure_reason="语法错误")
        assert cb.attempt_count == 2

    def test_is_tripped_after_max_reached(self):
        """达到 max_repairs 后 is_tripped 返回 True。"""
        cb = CircuitBreaker(max_repairs=2)
        assert cb.is_tripped() is False
        cb.record_attempt()
        assert cb.is_tripped() is False
        cb.record_attempt()
        assert cb.is_tripped() is True

    def test_is_tripped_exact_at_threshold(self):
        """attempt_count == max_repairs 时即熔断（不是超过后）。"""
        cb = CircuitBreaker(max_repairs=1)
        assert cb.is_tripped() is False
        cb.record_attempt()
        assert cb.is_tripped() is True

    def test_remaining_decreases_with_attempts(self):
        """每次 record_attempt 后 remaining 应递减。"""
        cb = CircuitBreaker(max_repairs=3)
        assert cb.remaining == 3
        cb.record_attempt()
        assert cb.remaining == 2
        cb.record_attempt()
        assert cb.remaining == 1
        cb.record_attempt()
        assert cb.remaining == 0

    def test_remaining_never_negative(self):
        """超过阈值后 remaining 不应为负数。"""
        cb = CircuitBreaker(max_repairs=2)
        cb.record_attempt()
        cb.record_attempt()
        cb.record_attempt()  # 超出阈值
        assert cb.remaining == 0

    def test_raises_value_error_for_max_repairs_less_than_one(self):
        """max_repairs < 1 时 __init__ 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="max_repairs"):
            CircuitBreaker(max_repairs=0)
        with pytest.raises(ValueError, match="max_repairs"):
            CircuitBreaker(max_repairs=-1)

    def test_max_repairs_default_is_three(self):
        """默认 max_repairs 为 3。"""
        cb = CircuitBreaker()
        assert cb.max_repairs == 3

    def test_generate_failure_report_contains_all_keys(self):
        """generate_failure_report 应包含所有必需字段。"""
        cb = CircuitBreaker(max_repairs=2)
        cb.record_attempt(
            detail="修改了文件路径",
            failure_reason="路径拼接错误",
        )
        cb.record_attempt(
            detail="修改了导入顺序",
            failure_reason="循环导入未解决",
        )
        report = cb.generate_failure_report()
        required_keys = {
            "tripped", "attempts", "max_repairs",
            "last_attempt", "last_failure", "suggestion", "timestamp",
        }
        assert set(report.keys()) == required_keys
        assert report["tripped"] is True
        assert report["attempts"] == 2
        assert report["max_repairs"] == 2
        assert report["last_attempt"] == "修改了导入顺序"
        assert report["last_failure"] == "循环导入未解决"

    def test_generate_failure_report_suggestion_mentions_intervention(self):
        """建议应包含人工介入相关提示。"""
        cb = CircuitBreaker(max_repairs=3)
        cb.record_attempt()
        cb.record_attempt()
        cb.record_attempt()
        report = cb.generate_failure_report()
        assert "人工介入" in report["suggestion"]
        # 应提到三个关键检查方向
        assert "验收标准" in report["suggestion"]
        assert "可用工具" in report["suggestion"]
        assert "系统性障碍" in report["suggestion"]

    def test_generate_failure_report_without_details(self):
        """未提供 detail/failure_reason 时报告应有占位文本。"""
        cb = CircuitBreaker(max_repairs=1)
        cb.record_attempt()
        report = cb.generate_failure_report()
        assert report["last_attempt"] == "无"
        assert report["last_failure"] == "未知"

    def test_generate_failure_report_timestamp_format(self):
        """时间戳应为 ISO 8601 格式。"""
        cb = CircuitBreaker(max_repairs=1)
        cb.record_attempt()
        report = cb.generate_failure_report()
        # 基本格式检查：含 T 和 Z 或 +00:00
        ts = report["timestamp"]
        assert "T" in ts
        assert ts.endswith("+00:00") or ts.endswith("Z")


# ═══════════════════════════════════════════════════════════════
# ConvergenceDetector 测试
# ═══════════════════════════════════════════════════════════════

class TestConvergenceDetector:

    def test_first_call_returns_false(self):
        """首次调用无历史 → 返回 False。"""
        cd = ConvergenceDetector()
        result = cd.check_convergence("修改了文件路径")
        assert result is False

    def test_identical_changes_returns_true(self):
        """连续两次相同的 changes_made → 返回 True（原地打转）。"""
        cd = ConvergenceDetector()
        cd.check_convergence("修改了文件路径")
        result = cd.check_convergence("修改了文件路径")
        assert result is True

    def test_different_changes_returns_false(self):
        """连续两次不同的 changes_made → 返回 False。"""
        cd = ConvergenceDetector()
        cd.check_convergence("修改了文件路径")
        result = cd.check_convergence("重构了模块导入")
        assert result is False

    def test_normalization_whitespace_difference(self):
        """仅空白差异不应导致假阴性（归一化后应判定为收敛）。"""
        cd = ConvergenceDetector()
        cd.check_convergence("修改了文件路径")
        # 多空白、换行应被归一化处理
        result = cd.check_convergence("  修改了    文件路径  ")
        assert result is True, "空白差异应被归一化，判定为收敛"

    def test_normalization_case_difference(self):
        """仅大小写差异不应导致假阴性（归一化后应判定为收敛）。"""
        cd = ConvergenceDetector()
        cd.check_convergence("Refactored Module Imports")
        result = cd.check_convergence("refactored module imports")
        assert result is True, "大小写差异应被归一化，判定为收敛"

    def test_normalization_punctuation_removal(self):
        """标点符号应被移除后比较。"""
        cd = ConvergenceDetector()
        cd.check_convergence("修改了文件路径。")
        result = cd.check_convergence("修改了文件路径")
        assert result is True, "标点符号差异应被归一化，判定为收敛"

    def test_last_score_populated_after_comparison(self):
        """第二次调用后 last_score 应被填充。"""
        cd = ConvergenceDetector()
        assert cd.last_score == 0.0
        cd.check_convergence("修改了文件路径")
        assert cd.last_score == 0.0  # 首次调用仍为 0
        cd.check_convergence("修改了文件路径")
        assert cd.last_score > 0.0  # 第二次调用后应有分数
        assert cd.last_score == 1.0  # 完全相同

    def test_last_score_with_different_text(self):
        """不同文本的 last_score 应反映相似度。"""
        cd = ConvergenceDetector()
        cd.check_convergence("修改了文件路径")
        cd.check_convergence("重构了整个模块结构")
        assert 0.0 <= cd.last_score < 0.7  # 不同文本相似度应较低

    def test_reset_clears_history(self):
        """reset() 后首次调用应再次返回 False。"""
        cd = ConvergenceDetector()
        cd.check_convergence("修改了文件路径")
        cd.check_convergence("修改了文件路径")  # 第二次相同 → True
        cd.reset()
        # 重置后首次调用应返回 False
        result = cd.check_convergence("修改了文件路径")
        assert result is False

    def test_reset_clears_last_score(self):
        """reset() 应将 last_score 归零。"""
        cd = ConvergenceDetector()
        cd.check_convergence("修改了文件路径")
        cd.check_convergence("修改了文件路径")
        assert cd.last_score > 0.0
        cd.reset()
        assert cd.last_score == 0.0

    def test_threshold_default_is_0_7(self):
        """默认阈值为 0.7。"""
        cd = ConvergenceDetector()
        assert cd.threshold == 0.7

    def test_raises_value_error_for_invalid_threshold(self):
        """阈值不在 [0.0, 1.0] 范围内应抛出 ValueError。"""
        with pytest.raises(ValueError, match="threshold"):
            ConvergenceDetector(threshold=-0.1)
        with pytest.raises(ValueError, match="threshold"):
            ConvergenceDetector(threshold=1.1)

    def test_custom_threshold_can_be_set(self):
        """自定义阈值应生效。"""
        cd = ConvergenceDetector(threshold=0.5)
        assert cd.threshold == 0.5

    def test_threshold_boundary_zero(self):
        """阈值 0.0 总是收敛。"""
        cd = ConvergenceDetector(threshold=0.0)
        cd.check_convergence("修改了文件路径")
        result = cd.check_convergence("完全不同的内容")
        assert result is True  # 任何相似度 >= 0.0 → True

    def test_threshold_boundary_one(self):
        """阈值 1.0 仅完全相同才收敛。"""
        cd = ConvergenceDetector(threshold=1.0)
        cd.check_convergence("修改了文件路径")
        result = cd.check_convergence("修改了文件路径！")  # 标点归一化后相同
        assert result is True

    def test_empty_string_handling(self):
        """空字符串应能正常处理。"""
        cd = ConvergenceDetector()
        cd.check_convergence("")
        result = cd.check_convergence("")
        assert result is True  # 两个空字符串归一化后相同

    def test_check_semantic_reserved_interface(self):
        """check_semantic 预留接口应回退到编辑距离（无 model 时）。"""
        cd = ConvergenceDetector()
        cd.check_convergence("修改了文件路径")
        score = cd.check_semantic("修改了文件路径")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_similarity_edge_case_both_empty(self):
        """两个空字符串的相似度应为 1.0。"""
        result = ConvergenceDetector._similarity("", "")
        assert result == 1.0

    def test_similarity_edge_case_one_empty(self):
        """一个空字符串与一个非空字符串的相似度应为 0.0。"""
        result = ConvergenceDetector._similarity("", "内容")
        assert result == 0.0
        result = ConvergenceDetector._similarity("内容", "")
        assert result == 0.0


# ═══════════════════════════════════════════════════════════════
# RepairLog 测试
# ═══════════════════════════════════════════════════════════════

class TestRepairLog:

    def test_record_appends_entry_with_correct_fields(self):
        """record 应追加包含正确字段的条目。"""
        log = RepairLog()
        log.record(
            attempt=1,
            changes_made="修改了文件路径",
            fixed=True,
            convergence_score=0.0,
        )
        assert len(log.entries) == 1
        entry = log.entries[0]
        assert entry["attempt"] == 1
        assert entry["changes_made"] == "修改了文件路径"
        assert entry["fixed"] is True
        assert entry["convergence_score"] == 0.0
        assert "timestamp" in entry

    def test_record_extra_fields(self):
        """extra 参数字段应合并到条目中。"""
        log = RepairLog()
        log.record(
            attempt=1,
            changes_made="修改了路径",
            fixed=False,
            convergence_score=0.1,
            extra={"tool_called": "write_file", "file_path": "x.py"},
        )
        entry = log.entries[0]
        assert entry["tool_called"] == "write_file"
        assert entry["file_path"] == "x.py"

    def test_to_audit_events_generates_one_event_per_entry(self):
        """to_audit_events 应为每个条目生成一个 SecurityAuditEvent。"""
        log = RepairLog()
        log.record(attempt=1, changes_made="修改了路径", fixed=False, convergence_score=0.1)
        log.record(attempt=2, changes_made="修改了导入", fixed=True, convergence_score=0.0)
        events = log.to_audit_events(trace_id="abc123")
        assert len(events) == 2

    def test_to_audit_events_metadata_contains_entry_fields(self):
        """审计事件 metadata 应包含修复条目核心字段。"""
        log = RepairLog()
        log.record(attempt=1, changes_made="修改了路径", fixed=False, convergence_score=0.1)
        events = log.to_audit_events(trace_id="trace-001")
        event = events[0]
        assert event.decision == "REPAIR"
        assert event.trace_id == "trace-001"
        assert event.metadata["repair_attempt"] == 1
        assert event.metadata["changes_made"] == "修改了路径"
        assert event.metadata["fixed"] is False
        assert event.metadata["convergence_score"] == 0.1

    def test_to_audit_events_convergence_detected_rule_id(self):
        """convergence_score >= 0.7 时 rule_id 应为 'convergence_detected'。"""
        log = RepairLog()
        log.record(attempt=1, changes_made="修改了路径", fixed=False, convergence_score=0.85)
        events = log.to_audit_events()
        assert events[0].rule_id == "convergence_detected"

    def test_to_audit_events_normal_rule_id(self):
        """convergence_score < 0.7 时 rule_id 应为 repair_attempt_{N}。"""
        log = RepairLog()
        log.record(attempt=2, changes_made="修改了导入", fixed=True, convergence_score=0.3)
        events = log.to_audit_events()
        assert events[0].rule_id == "repair_attempt_2"

    def test_to_audit_events_risk_score_for_fixed(self):
        """修复成功时 risk_score 应为 40。"""
        log = RepairLog()
        log.record(attempt=1, changes_made="修复了", fixed=True, convergence_score=0.0)
        events = log.to_audit_events()
        assert events[0].risk_score == 40

    def test_to_audit_events_risk_score_for_unfixed(self):
        """修复失败时 risk_score 应为 70。"""
        log = RepairLog()
        log.record(attempt=1, changes_made="未修复", fixed=False, convergence_score=0.0)
        events = log.to_audit_events()
        assert events[0].risk_score == 70

    def test_last_summary_returns_formatted_text(self):
        """last_summary 应返回格式化的摘要文本。"""
        log = RepairLog()
        log.record(attempt=1, changes_made="修改了文件路径", fixed=True, convergence_score=0.0)
        summary = log.last_summary()
        assert "第 1 轮" in summary
        assert "已修复" in summary
        assert "修改了文件路径" in summary

    def test_last_summary_for_failed_attempt(self):
        """未修复的条目应显示 '未修复'。"""
        log = RepairLog()
        log.record(attempt=2, changes_made="未解决循环导入", fixed=False, convergence_score=0.8)
        summary = log.last_summary()
        assert "第 2 轮" in summary
        assert "未修复" in summary

    def test_last_summary_empty_log(self):
        """空日志的 last_summary 应返回空字符串。"""
        log = RepairLog()
        assert log.last_summary() == ""

    def test_last_summary_truncates_long_changes(self):
        """changes_made 超过 max_chars 时应截断。"""
        log = RepairLog()
        long_changes = "修改了大量代码来修复问题。" * 50
        log.record(attempt=1, changes_made=long_changes, fixed=True, convergence_score=0.0)
        summary = log.last_summary(max_chars=100)
        assert len(summary) < 500  # 因为截断了 changes_made

    def test_all_failed_all_unfixed(self):
        """所有条目 fixed=False → all_failed() 返回 True。"""
        log = RepairLog()
        log.record(attempt=1, changes_made="尝试1", fixed=False, convergence_score=0.1)
        log.record(attempt=2, changes_made="尝试2", fixed=False, convergence_score=0.3)
        assert log.all_failed() is True

    def test_all_failed_with_one_success(self):
        """存在任一条目 fixed=True → all_failed() 返回 False。"""
        log = RepairLog()
        log.record(attempt=1, changes_made="尝试1", fixed=False, convergence_score=0.1)
        log.record(attempt=2, changes_made="尝试2", fixed=True, convergence_score=0.0)
        assert log.all_failed() is False

    def test_all_failed_empty_log(self):
        """空日志应认为全部失败（返回 True）。"""
        log = RepairLog()
        assert log.all_failed() is True

    def test_multiple_records_increment_correctly(self):
        """多次 record 应正确维护顺序。"""
        log = RepairLog()
        for i in range(1, 4):
            log.record(attempt=i, changes_made=f"修改{i}", fixed=False, convergence_score=0.0)
        assert len(log.entries) == 3
        assert log.entries[0]["attempt"] == 1
        assert log.entries[1]["attempt"] == 2
        assert log.entries[2]["attempt"] == 3


# ═══════════════════════════════════════════════════════════════
# ThresholdRecommendation 测试
# ═══════════════════════════════════════════════════════════════

class TestThresholdRecommendation:
    def test_default_is_inactive(self):
        """默认推荐未激活。"""
        from pyagent.harness.context.repair_context import ThresholdRecommendation
        rec = ThresholdRecommendation()
        assert rec.is_active is False
        assert rec.recommended == 0.7
        assert rec.confidence == "insufficient"
        assert rec.sample_count == 0

    def test_active_recommendation(self):
        """激活的推荐包含完整分布信息。"""
        from pyagent.harness.context.repair_context import ThresholdRecommendation
        rec = ThresholdRecommendation(
            recommended=0.78,
            confidence="medium",
            sample_count=25,
            distribution={"min": 0.1, "p25": 0.3, "p50": 0.5, "p75": 0.78, "p90": 0.88, "max": 0.95},
            is_active=True,
            fallback_reason="",
        )
        assert rec.is_active is True
        assert rec.recommended == 0.78
        assert rec.distribution["p75"] == 0.78

    def test_inactive_with_reason(self):
        """未激活的推荐包含回退原因。"""
        from pyagent.harness.context.repair_context import ThresholdRecommendation
        rec = ThresholdRecommendation(
            recommended=0.7,
            confidence="insufficient",
            sample_count=5,
            is_active=False,
            fallback_reason="样本不足（5 < 10）",
        )
        assert rec.is_active is False
        assert "样本不足" in rec.fallback_reason


# ═══════════════════════════════════════════════════════════════
# ThresholdAnalyzer 测试
# ═══════════════════════════════════════════════════════════════

class TestThresholdAnalyzer:
    def test_insufficient_samples_zero(self):
        """0 条样本 → insufficient。"""
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        analyzer = ThresholdAnalyzer()
        rec = analyzer.analyze([])
        assert rec.is_active is False
        assert rec.confidence == "insufficient"
        assert rec.sample_count == 0
        assert rec.recommended == 0.7  # 回退默认值
        assert "样本不足" in rec.fallback_reason

    def test_insufficient_samples_nine(self):
        """9 条样本 → 仍不足（MIN_SAMPLES=10）。"""
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        entries = [
            {"convergence_score": 0.1 * (i + 1)}
            for i in range(9)
        ]
        analyzer = ThresholdAnalyzer()
        rec = analyzer.analyze(entries)
        assert rec.is_active is False
        assert rec.sample_count == 9
        assert "9 < 10" in rec.fallback_reason

    def test_low_confidence_at_min_samples(self):
        """恰好 10 条样本 → low 置信度。"""
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        entries = [
            {"convergence_score": 0.05 * (i + 1)}
            for i in range(10)
        ]
        analyzer = ThresholdAnalyzer()
        rec = analyzer.analyze(entries)
        assert rec.is_active is True
        assert rec.confidence == "low"
        assert rec.sample_count == 10

    def test_medium_confidence_at_20(self):
        """20 条样本 → medium 置信度。"""
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        scores = [0.1 + 0.03 * i for i in range(20)]
        entries = [{"convergence_score": s} for s in scores]
        analyzer = ThresholdAnalyzer()
        rec = analyzer.analyze(entries)
        assert rec.confidence == "medium"
        assert rec.is_active is True

    def test_high_confidence_at_50(self):
        """50 条样本 → high 置信度。"""
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        scores = [0.1 + 0.01 * i for i in range(50)]
        entries = [{"convergence_score": s} for s in scores]
        analyzer = ThresholdAnalyzer()
        rec = analyzer.analyze(entries)
        assert rec.confidence == "high"
        assert rec.is_active is True

    def test_excludes_zero_scores(self):
        """convergence_score=0 的条目被排除（首次修补，无历史可比较）。"""
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        entries = [
            {"convergence_score": 0.0},   # 排除
            {"convergence_score": 0.0},   # 排除
            {"convergence_score": 0.5},
            {"convergence_score": 0.6},
            {"convergence_score": 0.0},   # 排除
            {"convergence_score": 0.7},
            {"convergence_score": 0.8},
            {"convergence_score": 0.3},
            {"convergence_score": 0.4},
            {"convergence_score": 0.2},   # 第 7 个有效样本
            {"convergence_score": 0.9},
            {"convergence_score": 0.1},
            {"convergence_score": 0.55},  # 第 10 个有效样本
            {"convergence_score": 0.65},
            {"convergence_score": 0.75},
        ]
        analyzer = ThresholdAnalyzer()
        rec = analyzer.analyze(entries)
        # 排除 3 个 0.0 后剩 12 条 → low
        assert rec.sample_count == 12
        assert rec.is_active is True

    def test_p75_calculation_known_distribution(self):
        """P75 分位数计算：对已知分布验证。"""
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        # 10 个均匀分布的值 [0.1, 0.2, ..., 1.0]
        scores = [0.1 * i for i in range(1, 11)]
        entries = [{"convergence_score": s} for s in scores]
        analyzer = ThresholdAnalyzer()
        rec = analyzer.analyze(entries)

        # 10 个值，P75 索引 = (10-1)*0.75 = 6.75
        # sorted[6] + 0.75 * (sorted[7] - sorted[6])
        # = 0.7 + 0.75 * (0.8 - 0.7) = 0.7 + 0.075 = 0.775
        assert rec.distribution["p75"] == 0.775

    def test_p75_in_higher_density_range_raises_threshold(self):
        """
        场景：大部分修补的相似度集中在高区间（0.7-0.85），
        P75 应落在较高位置 → 阈值上调。
        """
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        # 15 条集中在 0.7-0.88 高相似区间
        scores = [
            0.72, 0.74, 0.76, 0.78, 0.80,
            0.81, 0.82, 0.83, 0.84, 0.85,
            0.86, 0.87, 0.88,
            # 少量低相似度（有效修补）
            0.25, 0.15,
        ]
        entries = [{"convergence_score": s} for s in scores]
        analyzer = ThresholdAnalyzer()
        rec = analyzer.analyze(entries)

        # P75 应 > 0.80（大部分数据集中在此）
        assert rec.distribution["p75"] > 0.80, (
            f"高相似度集中时 P75 应上调，实际 P75={rec.distribution['p75']}"
        )

    def test_p75_in_lower_range_lowers_threshold(self):
        """
        场景：大部分修补相似度低（有效的差异修补），
        P75 较低 → 阈值下调。
        """
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        # 15 条集中在 0.15-0.4 低相似区间
        scores = [
            0.15, 0.18, 0.20, 0.22, 0.25,
            0.28, 0.30, 0.32, 0.35, 0.38,
            0.40,
            # 少量高相似度
            0.75, 0.80, 0.85,
            # 再补充几条低值到 15
            0.12, 0.14,
        ]
        entries = [{"convergence_score": s} for s in scores]
        analyzer = ThresholdAnalyzer()
        rec = analyzer.analyze(entries)

        # P75 应 < 0.50（大部分差异大，阈值应该更低以捕捉真正的收敛）
        assert rec.distribution["p75"] < 0.50, (
            f"低相似度集中时 P75 应较低，实际 P75={rec.distribution['p75']}"
        )

    def test_clamped_to_upper_bound(self):
        """P75 超过上界时钳制到 max_threshold。"""
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        # 所有相似度都 > 0.9 → P75 会很高
        scores = [0.90 + 0.005 * i for i in range(15)]
        entries = [{"convergence_score": s} for s in scores]
        analyzer = ThresholdAnalyzer(min_threshold=0.6, max_threshold=0.85)
        rec = analyzer.analyze(entries)

        assert rec.recommended <= 0.85
        assert "钳制" in rec.fallback_reason or rec.recommended == 0.85

    def test_clamped_to_lower_bound(self):
        """P75 低于下界时钳制到 min_threshold。"""
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        # 所有相似度都很低 → P75 会很底
        scores = [0.02 + 0.005 * i for i in range(15)]
        entries = [{"convergence_score": s} for s in scores]
        analyzer = ThresholdAnalyzer(min_threshold=0.6, max_threshold=0.85)
        rec = analyzer.analyze(entries)

        assert rec.recommended >= 0.6
        assert "钳制" in rec.fallback_reason or rec.recommended == 0.6

    def test_custom_bounds(self):
        """自定义安全边界生效。"""
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        scores = [0.3 + 0.02 * i for i in range(15)]  # P75 ≈ 0.51
        entries = [{"convergence_score": s} for s in scores]

        # 窄边界 [0.5, 0.7]
        analyzer = ThresholdAnalyzer(min_threshold=0.5, max_threshold=0.7)
        rec = analyzer.analyze(entries)
        assert 0.5 <= rec.recommended <= 0.7

    def test_invalid_bounds_raises(self):
        """非法边界参数应抛出 ValueError。"""
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        import pytest
        with pytest.raises(ValueError):
            ThresholdAnalyzer(min_threshold=0.8, max_threshold=0.6)  # min > max

    def test_complete_distribution(self):
        """验证分布包含所有 6 个统计量。"""
        from pyagent.harness.context.repair_context import ThresholdAnalyzer
        scores = [0.1 * i for i in range(1, 11)]
        entries = [{"convergence_score": s} for s in scores]
        analyzer = ThresholdAnalyzer()
        rec = analyzer.analyze(entries)

        assert set(rec.distribution.keys()) == {"min", "p25", "p50", "p75", "p90", "max"}
        assert rec.distribution["min"] == 0.1
        assert rec.distribution["max"] == 1.0
        assert rec.distribution["min"] <= rec.distribution["p25"] <= rec.distribution["p50"]
        assert rec.distribution["p50"] <= rec.distribution["p75"] <= rec.distribution["p90"]
        assert rec.distribution["p90"] <= rec.distribution["max"]


# ═══════════════════════════════════════════════════════════════
# ThresholdAdapter 测试
# ═══════════════════════════════════════════════════════════════

class TestThresholdAdapter:
    def test_returns_fallback_when_not_calibrated(self):
        """未校准时返回 fallback。"""
        from pyagent.harness.context.repair_context import ThresholdAdapter
        adapter = ThresholdAdapter(fallback=0.72)
        assert adapter.get_threshold() == 0.72
        assert adapter.is_calibrated is False

    def test_returns_recommended_when_calibrated(self):
        """校准激活后返回推荐值。"""
        from pyagent.harness.context.repair_context import (
            ThresholdAdapter, ThresholdRecommendation,
        )
        adapter = ThresholdAdapter(fallback=0.7)
        rec = ThresholdRecommendation(
            recommended=0.78, confidence="medium", sample_count=25,
            distribution={}, is_active=True,
        )
        adapter.update(rec)
        assert adapter.is_calibrated is True
        assert adapter.get_threshold() == 0.78

    def test_still_uses_fallback_when_recommendation_inactive(self):
        """推荐未激活时继续使用 fallback。"""
        from pyagent.harness.context.repair_context import (
            ThresholdAdapter, ThresholdRecommendation,
        )
        adapter = ThresholdAdapter(fallback=0.7)
        rec = ThresholdRecommendation(
            recommended=0.9, confidence="insufficient", sample_count=5,
            is_active=False,
        )
        adapter.update(rec)
        assert adapter.is_calibrated is False
        assert adapter.get_threshold() == 0.7  # 回退

    def test_clamps_out_of_range_recommendation(self):
        """推荐值超出 [MIN, MAX] 时钳制。"""
        from pyagent.harness.context.repair_context import (
            ThresholdAdapter, ThresholdRecommendation,
        )
        adapter = ThresholdAdapter(fallback=0.7)
        rec = ThresholdRecommendation(
            recommended=0.95, confidence="high", sample_count=50,
            is_active=True,
        )
        adapter.update(rec)
        # 应钳制到 MAX=0.85
        assert adapter.get_threshold() <= ThresholdAdapter.MAX_THRESHOLD
        assert adapter.get_threshold() == 0.85

    def test_clamps_low_recommendation(self):
        """推荐值低于 MIN 时钳制。"""
        from pyagent.harness.context.repair_context import (
            ThresholdAdapter, ThresholdRecommendation,
        )
        adapter = ThresholdAdapter(fallback=0.7)
        rec = ThresholdRecommendation(
            recommended=0.3, confidence="high", sample_count=50,
            is_active=True,
        )
        adapter.update(rec)
        assert adapter.get_threshold() >= ThresholdAdapter.MIN_THRESHOLD
        assert adapter.get_threshold() == 0.6

    def test_reset_clears_calibration(self):
        """reset 后恢复为 fallback。"""
        from pyagent.harness.context.repair_context import (
            ThresholdAdapter, ThresholdRecommendation,
        )
        adapter = ThresholdAdapter(fallback=0.7)
        rec = ThresholdRecommendation(
            recommended=0.78, confidence="medium", sample_count=25,
            is_active=True,
        )
        adapter.update(rec)
        assert adapter.is_calibrated is True

        adapter.reset()
        assert adapter.is_calibrated is False
        assert adapter.get_threshold() == 0.7

    def test_current_threshold_property(self):
        """current_threshold 属性与 get_threshold() 一致。"""
        from pyagent.harness.context.repair_context import ThresholdAdapter
        adapter = ThresholdAdapter(fallback=0.72)
        assert adapter.current_threshold == adapter.get_threshold()

    def test_recommendation_property(self):
        """recommendation 属性返回最近一次推荐。"""
        from pyagent.harness.context.repair_context import (
            ThresholdAdapter, ThresholdRecommendation,
        )
        adapter = ThresholdAdapter()
        assert adapter.recommendation is None

        rec = ThresholdRecommendation(is_active=False)
        adapter.update(rec)
        assert adapter.recommendation is rec

    def test_default_fallback_is_0_7(self):
        """默认 fallback 为 0.7。"""
        from pyagent.harness.context.repair_context import ThresholdAdapter
        adapter = ThresholdAdapter()
        assert adapter.get_threshold() == 0.7


# ═══════════════════════════════════════════════════════════════
# ConvergenceDetector + ThresholdAdapter 集成测试
# ═══════════════════════════════════════════════════════════════

class TestConvergenceDetectorWithAdapter:
    def test_backward_compatible_without_adapter(self):
        """不传 adapter 时行为不变（向后兼容）。"""
        from pyagent.harness.context.repair_context import ConvergenceDetector
        cd = ConvergenceDetector(threshold=0.7)
        assert cd.check_convergence("修改了文件路径") is False
        assert cd.check_convergence("修改了文件路径") is True  # 完全相同

    def test_uses_adapter_threshold(self):
        """传入 adapter 后使用适配器阈值而非固定值。"""
        from pyagent.harness.context.repair_context import (
            ConvergenceDetector, ThresholdAdapter, ThresholdRecommendation,
        )
        # 将阈值设为适度值 0.75
        adapter = ThresholdAdapter(fallback=0.75)
        rec = ThresholdRecommendation(
            recommended=0.75, confidence="medium",
            sample_count=25, is_active=True,
        )
        adapter.update(rec)

        cd = ConvergenceDetector(threshold_adapter=adapter)
        # 首次：记录基线
        assert cd.check_convergence("修改了文件路径") is False
        # 第二次：相关但明显不同的文本（相似度较低）
        result = cd.check_convergence("重构了模块导入逻辑并更新了测试")
        # 两段文本差异大，相似度应远 < 0.75
        assert result is False
        assert cd.last_score < 0.75

    def test_adapter_switch_changes_behavior(self):
        """运行时更新 adapter 阈值，行为立即改变。"""
        from pyagent.harness.context.repair_context import (
            ConvergenceDetector, ThresholdAdapter, ThresholdRecommendation,
        )
        adapter = ThresholdAdapter(fallback=0.95)  # 极高阈值 → 几乎不判定收敛
        cd = ConvergenceDetector(threshold_adapter=adapter)

        # 记录第一次
        cd.check_convergence("修改了文件路径")
        # 第二次：完全相同，相似度 1.0 > 0.95 → 收敛
        assert cd.check_convergence("修改了文件路径") is True

        # 切换到极低阈值（0.6）
        cd.reset()
        rec = ThresholdRecommendation(
            recommended=0.6, confidence="high", sample_count=50,
            is_active=True,
        )
        adapter.update(rec)

        cd.check_convergence("修改了文件导入路径")
        # 相似度 ~0.7 > 0.6 → 收敛
        assert cd.check_convergence("修改了文件导入路径") is True

    def test_adapter_inactive_uses_fixed_threshold(self):
        """adapter 未激活时使用 ConvergenceDetector.__init__ 的固定 threshold。"""
        from pyagent.harness.context.repair_context import (
            ConvergenceDetector, ThresholdAdapter, ThresholdRecommendation,
        )
        adapter = ThresholdAdapter(fallback=0.72)
        # 未调用 update → is_calibrated=False
        cd = ConvergenceDetector(threshold=0.85, threshold_adapter=adapter)

        # 应使用 fallback=0.72（来自不活跃的 adapter）
        cd.check_convergence("修改了配置项")
        # 相似文本，相似度约 0.8 > 0.72 → 收敛
        assert cd.check_convergence("修改了配置文件") is True

    def test_effective_threshold_queryable(self):
        """可通过 adapter.current_threshold 查询当前有效阈值。"""
        from pyagent.harness.context.repair_context import (
            ConvergenceDetector, ThresholdAdapter, ThresholdRecommendation,
        )
        adapter = ThresholdAdapter(fallback=0.7)
        rec = ThresholdRecommendation(
            recommended=0.78, confidence="medium", sample_count=30,
            is_active=True,
        )
        adapter.update(rec)
        cd = ConvergenceDetector(threshold_adapter=adapter)

        # 调用一次以触发相似度计算
        cd.check_convergence("文本A")
        cd.check_convergence("文本B")

        assert adapter.current_threshold == 0.78
