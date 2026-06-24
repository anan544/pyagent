"""
规则推荐引擎测试。

覆盖：
    - 数据模型：FingerprintHotspot / RuleRecommendation / RecommendationReport
    - RuleRecommender.analyze()：空事件 / 无 BLOCK / 低于阈值 / 单规则 / 多规则
    - 指纹聚类：热点计算 / 最小次数过滤
    - 事故关联：全部关联抑制 / 部分关联标记 / 无事故推荐
    - 边界：空指纹 / 逗号分隔工具名 / 时间窗口
    - 持久化：save_report / load_report / delete_report
    - 集成：与 AuditLogReader 协作
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from pyagent.harness.context.audit_reader import AuditLogReader
from pyagent.harness.context.observability import SecurityAuditEvent
from pyagent.harness.context.rule_recommender import (
    RuleRecommender,
    RuleRecommendation,
    RecommendationReport,
    FingerprintHotspot,
    DEFAULT_MIN_BLOCK_COUNT,
)


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _make_event(**overrides) -> SecurityAuditEvent:
    """创建测试用 SecurityAuditEvent。"""
    defaults = {
        "decision": "ALLOW",
        "rule_id": "step_level_check",
        "plan_step_fingerprint": "abc123",
        "risk_score": 0,
        "tool_name": "read_file",
        "tool_params": {"file_path": "/tmp/test.py"},
        "phase": "executing",
        "state": "EXECUTING",
        "trace_id": "trace-001",
        "timestamp": "2026-06-20T10:00:00+00:00",
    }
    defaults.update(overrides)
    return SecurityAuditEvent(**defaults)


def _make_block_events(
    rule_id: str,
    count: int,
    fingerprint: str = "abc123",
    tool_name: str = "write_file, execute_python",
    trace_id: str = "trace-001",
    risk_score: int = 80,
) -> list[SecurityAuditEvent]:
    """批量创建同质 BLOCK 事件。"""
    events = []
    for i in range(count):
        events.append(_make_event(
            decision="BLOCK",
            rule_id=rule_id,
            plan_step_fingerprint=fingerprint,
            tool_name=tool_name,
            trace_id=f"{trace_id}-{i}",
            risk_score=risk_score,
        ))
    return events


def _temp_json_path() -> str:
    """创建临时 JSON 文件路径。"""
    fd, path = tempfile.mkstemp(suffix='.json', prefix='recommender_test_')
    os.close(fd)
    return path


# ═══════════════════════════════════════════════════════════════
# 数据模型测试
# ═══════════════════════════════════════════════════════════════

class TestFingerprintHotspot:
    def test_creation_with_fields(self):
        """FingerprintHotspot 正常创建。"""
        hs = FingerprintHotspot(
            fingerprint="abc123def456",
            count=8,
            tool_params_example={"file_path": "/tmp/test.py"},
        )
        assert hs.fingerprint == "abc123def456"
        assert hs.count == 8
        assert hs.tool_params_example == {"file_path": "/tmp/test.py"}

    def test_default_values(self):
        """FingerprintHotspot 默认值正确。"""
        hs = FingerprintHotspot(fingerprint="abc", count=3)
        assert hs.tool_params_example == {}


class TestRuleRecommendation:
    def test_creation_with_fields(self):
        """RuleRecommendation 正常创建。"""
        rec = RuleRecommendation(
            rule_id="HIGH_RISK_COMBOS[0]",
            combo=["execute_python", "write_file"],
            block_count=15,
            unique_fingerprints=4,
            top_fingerprints=[
                FingerprintHotspot(fingerprint="fp1", count=8),
                FingerprintHotspot(fingerprint="fp2", count=4),
            ],
            avg_risk_score=80.0,
            has_related_incidents=False,
            reason="测试理由",
        )
        assert rec.rule_id == "HIGH_RISK_COMBOS[0]"
        assert rec.block_count == 15
        assert rec.recommendation == "review_for_whitelist"

    def test_default_values(self):
        """RuleRecommendation 默认值。"""
        rec = RuleRecommendation(rule_id="test_rule", block_count=5)
        assert rec.combo == []
        assert rec.unique_fingerprints == 0
        assert rec.top_fingerprints == []
        assert rec.avg_risk_score == 0.0
        assert rec.has_related_incidents is False
        assert rec.recommendation == "review_for_whitelist"


class TestRecommendationReport:
    def test_creation_with_recommendations(self):
        """RecommendationReport 正常创建，含推荐列表。"""
        rec = RuleRecommendation(rule_id="r1", block_count=10)
        report = RecommendationReport(
            source_file="test.jsonl",
            total_events_analyzed=200,
            total_block_events=50,
            recommendations=[rec],
        )
        assert report.total_events_analyzed == 200
        assert report.total_block_events == 50
        assert len(report.recommendations) == 1
        assert report.generated_at != ""

    def test_default_values(self):
        """RecommendationReport 默认值。"""
        report = RecommendationReport()
        assert report.total_events_analyzed == 0
        assert report.total_block_events == 0
        assert report.recommendations == []
        assert report.source_file == ""
        assert report.analysis_window == {}


# ═══════════════════════════════════════════════════════════════
# RuleRecommender.analyze() 测试
# ═══════════════════════════════════════════════════════════════

class TestRuleRecommenderAnalyze:
    """analyze() 方法测试。"""

    def test_empty_events_returns_empty_report(self):
        """空事件列表 → 空报告。"""
        recommender = RuleRecommender()
        report = recommender.analyze([])
        assert report.total_events_analyzed == 0
        assert report.total_block_events == 0
        assert report.recommendations == []

    def test_no_block_events(self):
        """全部为 ALLOW 事件 → 无推荐。"""
        events = [
            _make_event(decision="ALLOW"),
            _make_event(decision="ALLOW", trace_id="trace-002"),
        ]
        recommender = RuleRecommender(min_block_count=1)
        report = recommender.analyze(events)
        assert report.total_events_analyzed == 2
        assert report.total_block_events == 0
        assert report.recommendations == []

    def test_below_min_block_count(self):
        """BLOCK 次数低于阈值 → 不生成推荐。"""
        events = [
            _make_event(decision="BLOCK", rule_id="HIGH_RISK_COMBOS[0]",
                        tool_name="write_file, execute_python"),
            _make_event(decision="BLOCK", rule_id="HIGH_RISK_COMBOS[0]",
                        tool_name="write_file, execute_python"),
            # 仅 2 次 BLOCK，低于默认阈值 5
        ]
        recommender = RuleRecommender(min_block_count=5)
        report = recommender.analyze(events)
        assert report.recommendations == []

    def test_single_rule_exceeds_threshold(self):
        """单规则 BLOCK 次数超过阈值 → 生成 1 条推荐。"""
        events = _make_block_events(
            rule_id="HIGH_RISK_COMBOS[0]",
            count=7,
            fingerprint="abc123",
            tool_name="write_file, execute_python",
        )
        recommender = RuleRecommender(min_block_count=5)
        report = recommender.analyze(events)
        assert report.total_block_events == 7
        assert len(report.recommendations) == 1
        rec = report.recommendations[0]
        assert rec.rule_id == "HIGH_RISK_COMBOS[0]"
        assert rec.block_count == 7
        assert "write_file" in rec.combo
        assert "execute_python" in rec.combo
        assert rec.has_related_incidents is False

    def test_multiple_rules_sorted_by_block_count(self):
        """多条规则 → 按 block_count 降序排列。"""
        events = []
        events += _make_block_events("HIGH_RISK_COMBOS[0]", count=10, fingerprint="fp_a")
        events += _make_block_events("HIGH_RISK_COMBOS[3]", count=15, fingerprint="fp_b",
                                     tool_name="execute_python, execute_command")
        events += _make_block_events("HIGH_RISK_COMBOS[2]", count=6, fingerprint="fp_c",
                                     tool_name="delete_file")

        recommender = RuleRecommender(min_block_count=5)
        report = recommender.analyze(events)

        assert len(report.recommendations) == 3
        # 验证排序
        assert report.recommendations[0].rule_id == "HIGH_RISK_COMBOS[3]"
        assert report.recommendations[0].block_count == 15
        assert report.recommendations[1].rule_id == "HIGH_RISK_COMBOS[0]"
        assert report.recommendations[1].block_count == 10
        assert report.recommendations[2].rule_id == "HIGH_RISK_COMBOS[2]"
        assert report.recommendations[2].block_count == 6

    def test_fingerprint_clustering(self):
        """指纹聚类正确 → 热点按次数降序，低频指纹被过滤。"""
        events = []
        # 指纹 fp_a 出现 8 次
        events += _make_block_events("HIGH_RISK_COMBOS[0]", count=8, fingerprint="fp_a")
        # 指纹 fp_b 出现 5 次
        events += _make_block_events("HIGH_RISK_COMBOS[0]", count=5, fingerprint="fp_b")
        # 指纹 fp_c 出现 2 次（低于 min_fingerprint_count=3，不进入热点）
        events += _make_block_events("HIGH_RISK_COMBOS[0]", count=2, fingerprint="fp_c")

        recommender = RuleRecommender(min_block_count=5, min_fingerprint_count=3)
        report = recommender.analyze(events)

        assert len(report.recommendations) == 1
        rec = report.recommendations[0]
        assert rec.unique_fingerprints == 3  # 总共有 3 个独立指纹
        assert len(rec.top_fingerprints) == 2  # 仅 fp_a(8) 和 fp_b(5) 进入热点

        # 排序验证：fp_a 在前
        assert rec.top_fingerprints[0].fingerprint == "fp_a"
        assert rec.top_fingerprints[0].count == 8
        assert rec.top_fingerprints[1].fingerprint == "fp_b"
        assert rec.top_fingerprints[1].count == 5

    def test_all_blocks_have_incidents_suppresses_recommendation(self):
        """全部 BLOCK 都关联安全事故 → 不生成推荐。"""
        # 所有 BLOCK 共享同一 trace_id，且该 trace_id 有 REPAIR 事件
        shared_trace = "shared-incident-trace"
        events = []
        for i in range(6):
            events.append(_make_event(
                decision="BLOCK",
                rule_id="HIGH_RISK_COMBOS[0]",
                tool_name="write_file, execute_python",
                plan_step_fingerprint="fp_a",
                trace_id=shared_trace,
                risk_score=80,
            ))
        # 同一 trace_id 的 REPAIR 事件
        events.append(_make_event(
            decision="REPAIR", rule_id="step_level_check",
            trace_id=shared_trace, tool_name="write_file",
        ))

        recommender = RuleRecommender(min_block_count=5)
        report = recommender.analyze(events)

        # 6 次 BLOCK 全部关联 REPAIR → 抑制推荐
        assert report.recommendations == []

    def test_partial_incidents_generates_recommendation(self):
        """部分 BLOCK 关联事故 → 生成推荐但标记 has_related_incidents=True。"""
        events = []
        # 5 次 BLOCK 关联事故（共享 trace_id = "bad-1"）
        for i in range(5):
            events.append(_make_event(
                decision="BLOCK",
                rule_id="HIGH_RISK_COMBOS[0]",
                plan_step_fingerprint="fp_a",
                tool_name="write_file, execute_python",
                trace_id="bad-1",
                risk_score=80,
            ))
        # 5 次 BLOCK 无事故（各自独立 trace_id）
        for i in range(5):
            events.append(_make_event(
                decision="BLOCK",
                rule_id="HIGH_RISK_COMBOS[0]",
                plan_step_fingerprint="fp_b",
                tool_name="write_file, execute_python",
                trace_id=f"safe-{i}",
                risk_score=80,
            ))
        # REPAIR 事件（仅关联 bad-1）
        events.append(_make_event(
            decision="REPAIR", rule_id="step_level_check",
            trace_id="bad-1", tool_name="write_file",
        ))

        recommender = RuleRecommender(min_block_count=5)
        report = recommender.analyze(events)

        assert len(report.recommendations) == 1
        rec = report.recommendations[0]
        assert rec.block_count == 10  # 全部 10 次
        assert rec.has_related_incidents is True

    def test_no_fingerprint_blocks(self):
        """BLOCK 事件无 plan_step_fingerprint → 统一归为 'no_params'。"""
        events = []
        for i in range(6):
            events.append(_make_event(
                decision="BLOCK",
                rule_id="HIGH_RISK_COMBOS[2]",
                plan_step_fingerprint="",  # 空指纹
                tool_name="delete_file",
                trace_id=f"trace-{i}",
            ))

        recommender = RuleRecommender(min_block_count=5)
        report = recommender.analyze(events)

        assert len(report.recommendations) == 1
        rec = report.recommendations[0]
        assert rec.unique_fingerprints == 1
        assert rec.top_fingerprints[0].fingerprint == "no_params"

    def test_mixed_tool_names_in_combo(self):
        """逗号分隔的 tool_name 正确拆解为 combo。"""
        events = _make_block_events(
            "HIGH_RISK_COMBOS[0]", count=6,
            tool_name="write_file, execute_python, execute_command",
        )

        recommender = RuleRecommender(min_block_count=5)
        report = recommender.analyze(events)

        assert len(report.recommendations) == 1
        combo = report.recommendations[0].combo
        assert "write_file" in combo
        assert "execute_python" in combo
        assert "execute_command" in combo

    def test_analysis_window_extracted(self):
        """分析窗口时间范围正确提取。"""
        events = [
            _make_event(
                decision="BLOCK", rule_id="HIGH_RISK_COMBOS[0]",
                timestamp="2026-06-20T08:00:00+00:00",
                tool_name="write_file",
            ),
            _make_event(
                decision="BLOCK", rule_id="HIGH_RISK_COMBOS[0]",
                timestamp="2026-06-20T18:00:00+00:00",
                tool_name="write_file",
            ),
            _make_event(
                decision="BLOCK", rule_id="HIGH_RISK_COMBOS[0]",
                timestamp="2026-06-20T12:00:00+00:00",
                tool_name="write_file",
            ),
            _make_event(
                decision="BLOCK", rule_id="HIGH_RISK_COMBOS[0]",
                timestamp="2026-06-20T22:00:00+00:00",
                tool_name="write_file",
            ),
            _make_event(
                decision="BLOCK", rule_id="HIGH_RISK_COMBOS[0]",
                timestamp="2026-06-20T15:00:00+00:00",
                tool_name="write_file",
            ),
        ]

        recommender = RuleRecommender(min_block_count=5)
        report = recommender.analyze(events)

        assert report.analysis_window["start"] == "2026-06-20T08:00:00+00:00"
        assert report.analysis_window["end"] == "2026-06-20T22:00:00+00:00"

    def test_avg_risk_score_calculated(self):
        """平均风险评分正确计算。"""
        events = []
        for i in range(5):
            events.append(_make_event(
                decision="BLOCK",
                rule_id="HIGH_RISK_COMBOS[0]",
                tool_name="write_file, execute_python",
                risk_score=80 if i % 2 == 0 else 60,
                trace_id=f"trace-{i}",
            ))

        recommender = RuleRecommender(min_block_count=5)
        report = recommender.analyze(events)

        # 3×80 + 2×60 = 360, avg = 72.0
        assert report.recommendations[0].avg_risk_score == 72.0

    def test_source_file_recorded(self):
        """source_file 正确记录到报告。"""
        events = _make_block_events("HIGH_RISK_COMBOS[0]", count=6)

        recommender = RuleRecommender()
        report = recommender.analyze(events, source_file=".claude/audit.jsonl")

        assert report.source_file == ".claude/audit.jsonl"

    def test_single_tool_in_combo(self):
        """单工具高危规则 → combo 仅含一个工具。"""
        events = _make_block_events(
            "HIGH_RISK_COMBOS[2]", count=6,
            tool_name="delete_file",
        )

        recommender = RuleRecommender(min_block_count=5)
        report = recommender.analyze(events)

        assert report.recommendations[0].combo == ["delete_file"]


# ═══════════════════════════════════════════════════════════════
# 持久化测试
# ═══════════════════════════════════════════════════════════════

class TestRuleRecommenderPersistence:
    """save_report / load_report / delete_report 测试。"""

    def test_save_and_load_roundtrip(self):
        """保存后加载 → 数据一致。"""
        path = _temp_json_path()
        try:
            rec = RuleRecommendation(
                rule_id="HIGH_RISK_COMBOS[0]",
                combo=["write_file", "execute_python"],
                block_count=12,
                unique_fingerprints=3,
                top_fingerprints=[
                    FingerprintHotspot(
                        fingerprint="fp_a", count=7,
                        tool_params_example={"file_path": "/tmp/x.py"},
                    ),
                ],
                avg_risk_score=78.5,
                has_related_incidents=True,
                reason="测试推荐理由",
            )
            report = RecommendationReport(
                source_file="test.jsonl",
                analysis_window={
                    "start": "2026-06-20T08:00:00+00:00",
                    "end": "2026-06-20T18:00:00+00:00",
                },
                total_events_analyzed=100,
                total_block_events=30,
                recommendations=[rec],
            )

            assert RuleRecommender.save_report(report, path) is True

            loaded = RuleRecommender.load_report(path)
            assert loaded is not None
            assert loaded.total_events_analyzed == 100
            assert loaded.total_block_events == 30
            assert len(loaded.recommendations) == 1
            lr = loaded.recommendations[0]
            assert lr.rule_id == "HIGH_RISK_COMBOS[0]"
            assert lr.block_count == 12
            assert lr.combo == ["write_file", "execute_python"]
            assert lr.avg_risk_score == 78.5
            assert lr.has_related_incidents is True
            assert len(lr.top_fingerprints) == 1
            assert lr.top_fingerprints[0].fingerprint == "fp_a"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_missing_file(self):
        """加载不存在的文件 → None。"""
        result = RuleRecommender.load_report("/nonexistent/path/report.json")
        assert result is None

    def test_load_corrupt_file(self):
        """加载损坏的 JSON 文件 → None。"""
        path = _temp_json_path()
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("这不是 JSON { invalid")
            result = RuleRecommender.load_report(path)
            assert result is None
        finally:
            Path(path).unlink(missing_ok=True)

    def test_delete_report(self):
        """删除报告文件 → True。"""
        path = _temp_json_path()
        try:
            report = RecommendationReport()
            RuleRecommender.save_report(report, path)
            assert Path(path).exists()
            assert RuleRecommender.delete_report(path) is True
            assert not Path(path).exists()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_delete_nonexistent(self):
        """删除不存在的文件 → True（幂等）。"""
        result = RuleRecommender.delete_report("/nonexistent/path/report.json")
        assert result is True

    def test_save_creates_parent_dirs(self):
        """保存时自动创建父目录。"""
        base = tempfile.mkdtemp(prefix='recommender_')
        try:
            path = os.path.join(base, "subdir", "nested", "report.json")
            report = RecommendationReport()
            assert RuleRecommender.save_report(report, path) is True
            assert Path(path).exists()
        finally:
            import shutil
            shutil.rmtree(base, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 集成测试：与 AuditLogReader 协作
# ═══════════════════════════════════════════════════════════════

class TestIntegrationWithAuditReader:
    """RuleRecommender + AuditLogReader 集成测试。"""

    def test_analyze_from_jsonl_file(self):
        """从 JSONL 文件读取事件 → 分析 → 生成推荐。"""
        fd, jsonl_path = tempfile.mkstemp(suffix='.jsonl', prefix='audit_int_')
        os.close(fd)

        try:
            # 写入混合审计事件到 JSONL
            events = []
            # BLOCK 事件（超过阈值）
            for i in range(8):
                events.append(_make_event(
                    decision="BLOCK",
                    rule_id="HIGH_RISK_COMBOS[0]",
                    tool_name="write_file, execute_python",
                    plan_step_fingerprint=f"fp_{i % 3}",  # 3 个不同指纹
                    trace_id=f"trace-{i}",
                    risk_score=80,
                ))
            # 一些 ALLOW 事件
            for i in range(5):
                events.append(_make_event(
                    decision="ALLOW",
                    rule_id="step_level_check",
                    tool_name="read_file",
                    trace_id=f"allow-{i}",
                ))
            # 一个 REPAIR 事件（安全事故）
            events.append(_make_event(
                decision="REPAIR",
                rule_id="step_level_check",
                tool_name="write_file",
                trace_id="trace-0",  # 关联 BLOCK trace-0
            ))

            # 写入 JSONL
            import json as json_mod
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for ev in events:
                    f.write(ev.to_json() + '\n')

            # 使用 AuditLogReader 读取
            reader = AuditLogReader(jsonl_path)
            all_events = reader.read_all()

            # 使用 RuleRecommender 分析
            recommender = RuleRecommender(min_block_count=5, min_fingerprint_count=2)
            report = recommender.analyze(all_events, source_file=jsonl_path)

            assert report.total_events_analyzed == 14
            assert report.total_block_events == 8
            assert len(report.recommendations) == 1

            rec = report.recommendations[0]
            assert rec.rule_id == "HIGH_RISK_COMBOS[0]"
            assert rec.block_count == 8
            assert rec.unique_fingerprints == 3
            assert rec.has_related_incidents is True  # trace-0 有 REPAIR
            assert len(rec.top_fingerprints) >= 1

        finally:
            Path(jsonl_path).unlink(missing_ok=True)

    def test_realistic_scenario_multiple_rules(self):
        """真实场景：多条规则触发，部分抑制。"""
        events = []
        # 规则 0: write_file + execute_python — 12 次 BLOCK，无事故
        events += _make_block_events("HIGH_RISK_COMBOS[0]", count=12,
                                     fingerprint="fp_code_gen", trace_id="safe-a")
        # 规则 1: write_file + execute_command — 7 次 BLOCK，全部关联事故
        for i in range(7):
            events.append(_make_event(
                decision="BLOCK", rule_id="HIGH_RISK_COMBOS[1]",
                tool_name="write_file, execute_command",
                plan_step_fingerprint="fp_danger",
                trace_id="bad-trace",
                risk_score=80,
            ))
        events.append(_make_event(
            decision="REPAIR", rule_id="step_level_check",
            trace_id="bad-trace", tool_name="execute_command",
        ))
        # 规则 2: delete_file — 3 次 BLOCK，低于阈值
        events += _make_block_events("HIGH_RISK_COMBOS[2]", count=3,
                                     fingerprint="fp_del", tool_name="delete_file",
                                     trace_id="safe-c")

        recommender = RuleRecommender(min_block_count=5, min_fingerprint_count=2)
        report = recommender.analyze(events)

        # 规则 0 → 推荐（无事故）
        # 规则 1 → 抑制（全部关联事故）
        # 规则 2 → 不进入分析（低于阈值）
        assert len(report.recommendations) == 1
        assert report.recommendations[0].rule_id == "HIGH_RISK_COMBOS[0]"
        assert report.recommendations[0].block_count == 12
        assert report.recommendations[0].has_related_incidents is False

    def test_recommendation_reason_contains_key_info(self):
        """推荐理由包含关键信息。"""
        events = _make_block_events("HIGH_RISK_COMBOS[3]", count=10,
                                    fingerprint="fp_test",
                                    tool_name="execute_python, execute_command")

        recommender = RuleRecommender(min_block_count=5)
        report = recommender.analyze(events)

        rec = report.recommendations[0]
        assert "10 次 BLOCK" in rec.reason
        assert "无关联安全事故" in rec.reason
        assert "人工审查" in rec.reason
