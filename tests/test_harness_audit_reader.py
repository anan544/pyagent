"""
审计日志读取器测试。

覆盖：
    - LogCursor：创建 / 序列化 / 反序列化 / 默认值
    - AuditLogReader.read_all()：空文件 / 正常文件 / 文件不存在
    - AuditLogReader.read_since()：首次读取 / 增量读取 / 无新数据
    - 防御性解析：损坏行跳过 / 尾部不完整行 / 空行
    - 游标持久化：save_cursor / load_cursor / delete_cursor
    - 过滤器：按 decision / trace_id / rule_id / time / risk_score / tool / phase
    - 统计聚合：分布计数 / P0 计数 / top_blocked_tools
    - 文件轮转检测：截断时游标重置
    - 边界情况：空行混入 / 合法损坏行混合
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from pyagent.harness.context.audit_reader import (
    AuditLogReader,
    LogCursor,
)
from pyagent.harness.context.observability import (
    SecurityAuditEvent,
    Sanitizer,
)


# ═══════════════════════════════════════════════════════════════
# 辅助
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
    }
    defaults.update(overrides)
    return SecurityAuditEvent(**defaults)


def _write_jsonl(path: str, events: list[dict | SecurityAuditEvent]) -> None:
    """将事件列表写入 JSONL 文件。"""
    with open(path, 'w', encoding='utf-8') as f:
        for ev in events:
            if isinstance(ev, SecurityAuditEvent):
                f.write(ev.to_json() + '\n')
            elif isinstance(ev, dict):
                f.write(json.dumps(ev) + '\n')
            elif isinstance(ev, str):
                f.write(ev + '\n')


def _temp_jsonl_path() -> str:
    """创建临时 JSONL 文件路径。"""
    fd, path = tempfile.mkstemp(suffix='.jsonl', prefix='audit_test_')
    os.close(fd)
    return path


# ═══════════════════════════════════════════════════════════════
# LogCursor 测试
# ═══════════════════════════════════════════════════════════════

class TestLogCursor:
    def test_default_values(self):
        """LogCursor 默认值正确。"""
        cursor = LogCursor()
        assert cursor.file_path == ""
        assert cursor.last_offset == 0
        assert cursor.last_inode == 0
        assert cursor.last_position == 0
        assert cursor.checksum == ""

    def test_to_dict_and_from_dict_roundtrip(self):
        """to_dict → from_dict 往返一致。"""
        cursor = LogCursor(
            file_path="/tmp/audit.jsonl",
            last_offset=4096,
            last_inode=12345,
            last_position=100,
            checksum="sha256:abc",
        )
        d = cursor.to_dict()
        restored = LogCursor.from_dict(d)
        assert restored.file_path == cursor.file_path
        assert restored.last_offset == cursor.last_offset
        assert restored.last_inode == cursor.last_inode
        assert restored.last_position == cursor.last_position
        assert restored.checksum == cursor.checksum

    def test_from_dict_partial(self):
        """from_dict 处理不完整字段。"""
        restored = LogCursor.from_dict({"file_path": "/x.jsonl"})
        assert restored.file_path == "/x.jsonl"
        assert restored.last_offset == 0

    def test_from_dict_empty(self):
        """from_dict 处理空 dict。"""
        restored = LogCursor.from_dict({})
        assert restored.file_path == ""
        assert restored.last_offset == 0


# ═══════════════════════════════════════════════════════════════
# AuditLogReader.read_all() 测试
# ═══════════════════════════════════════════════════════════════

class TestReadAll:
    def test_empty_file_returns_empty_list(self):
        """空文件返回空列表。"""
        path = _temp_jsonl_path()
        try:
            Path(path).write_text("", encoding='utf-8')
            reader = AuditLogReader(path)
            events = reader.read_all()
            assert events == []
        finally:
            os.unlink(path)

    def test_file_not_found_returns_empty_list(self):
        """文件不存在返回空列表。"""
        reader = AuditLogReader("/nonexistent/path/audit.jsonl")
        events = reader.read_all()
        assert events == []

    def test_reads_all_events(self):
        """读取文件中所有有效事件。"""
        path = _temp_jsonl_path()
        try:
            events_in = [
                _make_event(decision="ALLOW", rule_id="R001"),
                _make_event(decision="BLOCK", rule_id="R002", risk_score=80),
                _make_event(decision="REPAIR", rule_id="R003", risk_score=40),
            ]
            _write_jsonl(path, events_in)

            reader = AuditLogReader(path)
            events_out = reader.read_all()
            assert len(events_out) == 3
            assert events_out[0].decision == "ALLOW"
            assert events_out[1].decision == "BLOCK"
            assert events_out[2].decision == "REPAIR"
        finally:
            os.unlink(path)

    def test_events_are_securityauditevent_instances(self):
        """返回的是 SecurityAuditEvent 实例。"""
        path = _temp_jsonl_path()
        try:
            _write_jsonl(path, [_make_event()])
            reader = AuditLogReader(path)
            events = reader.read_all()
            assert len(events) == 1
            assert isinstance(events[0], SecurityAuditEvent)
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════
# AuditLogReader.read_since() 增量读取测试
# ═══════════════════════════════════════════════════════════════

class TestReadSince:
    def test_initial_read_from_beginning(self):
        """无游标时从文件开头读取。"""
        path = _temp_jsonl_path()
        try:
            _write_jsonl(path, [
                _make_event(rule_id="R001"),
                _make_event(rule_id="R002"),
            ])
            reader = AuditLogReader(path)
            cursor = LogCursor(file_path=path)
            events = list(reader.read_since(cursor))

            assert len(events) == 2
            assert cursor.last_position == 2
            assert cursor.last_offset > 0
        finally:
            os.unlink(path)

    def test_incremental_read_only_new_lines(self):
        """增量读取仅返回新增行。"""
        path = _temp_jsonl_path()
        try:
            # 第一批：2 条事件
            _write_jsonl(path, [
                _make_event(rule_id="R001"),
                _make_event(rule_id="R002"),
            ])

            reader = AuditLogReader(path)
            cursor = LogCursor(file_path=path)
            batch1 = list(reader.read_since(cursor))
            assert len(batch1) == 2
            offset_after_batch1 = cursor.last_offset

            # 第二批：追加 2 条事件
            with open(path, 'a', encoding='utf-8') as f:
                f.write(_make_event(rule_id="R003").to_json() + '\n')
                f.write(_make_event(rule_id="R004").to_json() + '\n')

            batch2 = list(reader.read_since(cursor))
            assert len(batch2) == 2
            assert batch2[0].rule_id == "R003"
            assert batch2[1].rule_id == "R004"
            assert cursor.last_offset > offset_after_batch1
        finally:
            os.unlink(path)

    def test_no_new_data_returns_empty(self):
        """无新数据时返回空生成器。"""
        path = _temp_jsonl_path()
        try:
            _write_jsonl(path, [_make_event(rule_id="R001")])

            reader = AuditLogReader(path)
            cursor = LogCursor(file_path=path)
            list(reader.read_since(cursor))  # 读完
            old_offset = cursor.last_offset

            batch2 = list(reader.read_since(cursor))
            assert batch2 == []
            assert cursor.last_offset == old_offset  # 游标不变
        finally:
            os.unlink(path)

    def test_cursor_unchanged_when_no_complete_line(self):
        """文件有内容但尾部不完整时，游标不更新。"""
        path = _temp_jsonl_path()
        try:
            _write_jsonl(path, [_make_event(rule_id="R001")])

            reader = AuditLogReader(path)
            cursor = LogCursor(file_path=path)
            list(reader.read_since(cursor))
            offset_after = cursor.last_offset

            # 追加不完整行（无 \\n）
            with open(path, 'a', encoding='utf-8') as f:
                f.write('{"decision":"ALLOW","rule_id":"R002"')  # 不完整且无换行

            batch2 = list(reader.read_since(cursor))
            assert batch2 == []  # 不完整行不被消费
            assert cursor.last_offset == offset_after  # 游标不变
        finally:
            os.unlink(path)

    def test_batch_size(self):
        """batch_size 控制批次大小。"""
        path = _temp_jsonl_path()
        try:
            events = [_make_event(rule_id=f"R{i:03d}") for i in range(10)]
            _write_jsonl(path, events)

            reader = AuditLogReader(path)
            cursor = LogCursor(file_path=path)

            # batch_size=3: 应分批 yield
            batches = []
            for event in reader.read_since(cursor, batch_size=3):
                batches.append(event)

            # 3 + 3 + 3 + 1 = 4 批
            assert len(batches) == 10
        finally:
            os.unlink(path)

    def test_appended_complete_line_after_incomplete(self):
        """不完整行补齐后，下次读取可消费。"""
        path = _temp_jsonl_path()
        try:
            _write_jsonl(path, [_make_event(rule_id="R001")])

            reader = AuditLogReader(path)
            cursor = LogCursor(file_path=path)
            list(reader.read_since(cursor))

            # 写入不完整行
            with open(path, 'a', encoding='utf-8') as f:
                f.write('{"decision":"ALLOW","rule_id":"R002","plan_step_fingerprint":"","risk_score":0}\n')
                # 注意：这行缺少部分字段但 JSON 有效
                f.write('{"decision":"ALLOW"')  # 不完整

            # 第一次读取：只有完整行
            batch1 = list(reader.read_since(cursor))
            assert len(batch1) == 1
            assert batch1[0].rule_id == "R002"

            # 补全不完整行（继续写入）
            with open(path, 'a', encoding='utf-8') as f:
                f.write(',"rule_id":"R003","plan_step_fingerprint":"","risk_score":0}\n')

            batch2 = list(reader.read_since(cursor))
            assert len(batch2) == 1
            assert batch2[0].rule_id == "R003"
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════
# 防御性解析测试
# ═══════════════════════════════════════════════════════════════

class TestDefensiveParsing:
    def test_corrupt_line_skipped_and_recorded(self):
        """损坏行被跳过并记录到 corrupt_lines。"""
        path = _temp_jsonl_path()
        try:
            lines = [
                _make_event(rule_id="R001").to_json(),
                "this is not json at all",
                _make_event(rule_id="R003").to_json(),
            ]
            _write_jsonl(path, lines)

            reader = AuditLogReader(path)
            events = reader.read_all()

            assert len(events) == 2
            assert events[0].rule_id == "R001"
            assert events[1].rule_id == "R003"

            # 损坏行记录
            corrupt = reader.corrupt_lines
            assert len(corrupt) == 1
            assert corrupt[0]["position"] == 2
            assert "JSONDecodeError" in corrupt[0]["error"]
        finally:
            os.unlink(path)

    def test_multiple_corrupt_lines_all_skipped(self):
        """多条损坏行全部跳过，有效行正常返回。"""
        path = _temp_jsonl_path()
        try:
            lines = [
                _make_event(rule_id="R001").to_json(),
                "{invalid json {{",
                _make_event(rule_id="R004").to_json(),
                "also not json",
                _make_event(rule_id="R006").to_json(),
            ]
            _write_jsonl(path, lines)

            reader = AuditLogReader(path)
            events = reader.read_all()

            assert len(events) == 3
            # 2 条损坏行被记录
            assert len(reader.corrupt_lines) == 2
            assert reader.corrupt_lines[0]["position"] == 2
            assert reader.corrupt_lines[1]["position"] == 4
        finally:
            os.unlink(path)

    def test_corrupt_lines_reset_on_new_read(self):
        """每次 read_since 调用重置 corrupt_lines。"""
        path = _temp_jsonl_path()
        try:
            _write_jsonl(path, [
                "not json",
                _make_event(rule_id="R002").to_json(),
            ])

            reader = AuditLogReader(path)
            list(reader.read_all())
            assert len(reader.corrupt_lines) == 1

            # 第二次读取（无新数据）
            cursor = LogCursor(file_path=path)
            cursor.last_offset = os.path.getsize(path)
            list(reader.read_since(cursor))
            assert len(reader.corrupt_lines) == 0  # 已重置
        finally:
            os.unlink(path)

    def test_empty_lines_skipped_silently(self):
        """空行被静默跳过，不记录为损坏。"""
        path = _temp_jsonl_path()
        try:
            lines = [
                "",
                _make_event(rule_id="R001").to_json(),
                "",
                "",
                _make_event(rule_id="R002").to_json(),
            ]
            _write_jsonl(path, lines)

            reader = AuditLogReader(path)
            events = reader.read_all()

            assert len(events) == 2
            assert len(reader.corrupt_lines) == 0
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════
# 游标持久化测试
# ═══════════════════════════════════════════════════════════════

class TestCursorPersistence:
    def test_save_and_load_roundtrip(self):
        """save_cursor → load_cursor 往返一致。"""
        cursor_path = _temp_jsonl_path().replace('.jsonl', '_cursor.json')
        try:
            cursor = LogCursor(
                file_path="/tmp/audit.jsonl",
                last_offset=8192,
                last_inode=54321,
                last_position=200,
                checksum="sha256:def",
            )
            AuditLogReader.save_cursor(cursor, cursor_path)
            assert os.path.exists(cursor_path)

            restored = AuditLogReader.load_cursor(cursor_path)
            assert restored is not None
            assert restored.file_path == cursor.file_path
            assert restored.last_offset == cursor.last_offset
            assert restored.last_inode == cursor.last_inode
            assert restored.last_position == cursor.last_position
        finally:
            if os.path.exists(cursor_path):
                os.unlink(cursor_path)

    def test_load_nonexistent_returns_none(self):
        """不存在的游标文件返回 None。"""
        cursor = AuditLogReader.load_cursor("/nonexistent/cursor.json")
        assert cursor is None

    def test_load_corrupt_cursor_returns_none(self):
        """损坏的游标文件返回 None。"""
        cursor_path = _temp_jsonl_path().replace('.jsonl', '_cursor.json')
        try:
            with open(cursor_path, 'w', encoding='utf-8') as f:
                f.write("not valid json {{{")
            cursor = AuditLogReader.load_cursor(cursor_path)
            assert cursor is None
        finally:
            if os.path.exists(cursor_path):
                os.unlink(cursor_path)

    def test_delete_removes_file(self):
        """delete_cursor 删除文件。"""
        cursor_path = _temp_jsonl_path().replace('.jsonl', '_cursor.json')
        try:
            AuditLogReader.save_cursor(
                LogCursor(file_path="/tmp/x.jsonl"), cursor_path,
            )
            assert os.path.exists(cursor_path)
            ok = AuditLogReader.delete_cursor(cursor_path)
            assert ok
            assert not os.path.exists(cursor_path)
        finally:
            if os.path.exists(cursor_path):
                os.unlink(cursor_path)

    def test_delete_nonexistent_returns_true(self):
        """删除不存在的游标文件返回 True。"""
        ok = AuditLogReader.delete_cursor("/nonexistent/cursor.json")
        assert ok

    def test_save_creates_parent_dirs(self):
        """save_cursor 自动创建父目录。"""
        tmpdir = tempfile.mkdtemp(prefix='audit_test_dir_')
        cursor_path = os.path.join(tmpdir, 'sub', 'deep', 'cursor.json')
        try:
            AuditLogReader.save_cursor(
                LogCursor(file_path="/tmp/x.jsonl"), cursor_path,
            )
            assert os.path.exists(cursor_path)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# 过滤器测试
# ═══════════════════════════════════════════════════════════════

class TestFilter:
    EVENTS = [
        _make_event(decision="ALLOW", rule_id="R001", risk_score=0,
                     tool_name="read_file", phase="executing",
                     trace_id="trace-aaa"),
        _make_event(decision="BLOCK", rule_id="R002", risk_score=80,
                     tool_name="write_file", phase="planning",
                     trace_id="trace-bbb"),
        _make_event(decision="BLOCK", rule_id="R001", risk_score=60,
                     tool_name="execute_python", phase="executing",
                     trace_id="trace-aaa"),
        _make_event(decision="REPAIR", rule_id="R003", risk_score=40,
                     tool_name="write_file", phase="repairing",
                     trace_id="trace-ccc"),
        _make_event(decision="ALLOW", rule_id="R004", risk_score=0,
                     tool_name="search_content", phase="verifying",
                     trace_id="trace-bbb"),
    ]

    def test_filter_by_decision_single(self):
        """按单个 decision 过滤。"""
        result = list(AuditLogReader.filter(
            iter(self.EVENTS), decision="BLOCK"
        ))
        assert len(result) == 2
        assert all(e.decision == "BLOCK" for e in result)

    def test_filter_by_decision_list(self):
        """按 decision 列表过滤。"""
        result = list(AuditLogReader.filter(
            iter(self.EVENTS), decision=["BLOCK", "REPAIR"]
        ))
        assert len(result) == 3
        decisions = {e.decision for e in result}
        assert decisions == {"BLOCK", "REPAIR"}

    def test_filter_by_decision_set(self):
        """按 decision set 过滤。"""
        result = list(AuditLogReader.filter(
            iter(self.EVENTS), decision={"ALLOW"}
        ))
        assert len(result) == 2
        assert all(e.decision == "ALLOW" for e in result)

    def test_filter_by_trace_id(self):
        """按 trace_id 过滤。"""
        result = list(AuditLogReader.filter(
            iter(self.EVENTS), trace_id="trace-aaa"
        ))
        assert len(result) == 2
        assert result[0].rule_id == "R001" and result[0].decision == "ALLOW"
        assert result[1].rule_id == "R001" and result[1].decision == "BLOCK"

    def test_filter_by_rule_id_single(self):
        """按单个 rule_id 过滤。"""
        result = list(AuditLogReader.filter(
            iter(self.EVENTS), rule_id="R001"
        ))
        assert len(result) == 2

    def test_filter_by_rule_id_list(self):
        """按 rule_id 列表过滤。"""
        result = list(AuditLogReader.filter(
            iter(self.EVENTS), rule_id=["R001", "R003"]
        ))
        assert len(result) == 3

    def test_filter_by_min_risk_score(self):
        """按最低风险评分过滤。"""
        result = list(AuditLogReader.filter(
            iter(self.EVENTS), min_risk_score=60
        ))
        assert len(result) == 2
        assert all(e.risk_score >= 60 for e in result)

    def test_filter_by_max_risk_score(self):
        """按最高风险评分过滤。"""
        result = list(AuditLogReader.filter(
            iter(self.EVENTS), max_risk_score=0
        ))
        assert len(result) == 2
        assert all(e.risk_score <= 0 for e in result)

    def test_filter_by_risk_score_range(self):
        """按风险评分范围过滤。"""
        result = list(AuditLogReader.filter(
            iter(self.EVENTS), min_risk_score=40, max_risk_score=60
        ))
        assert len(result) == 2
        assert all(40 <= e.risk_score <= 60 for e in result)

    def test_filter_by_tool_name(self):
        """按工具名称过滤。"""
        result = list(AuditLogReader.filter(
            iter(self.EVENTS), tool_name="write_file"
        ))
        assert len(result) == 2
        assert all(e.tool_name == "write_file" for e in result)

    def test_filter_by_phase(self):
        """按阶段过滤。"""
        result = list(AuditLogReader.filter(
            iter(self.EVENTS), phase="executing"
        ))
        assert len(result) == 2
        assert all(e.phase == "executing" for e in result)

    def test_filter_by_time_range(self):
        """按时间窗口过滤。"""
        events = [
            _make_event(rule_id="R001", timestamp="2025-01-01T00:00:00Z"),
            _make_event(rule_id="R002", timestamp="2025-06-15T12:00:00Z"),
            _make_event(rule_id="R003", timestamp="2025-12-31T23:59:59Z"),
        ]
        result = list(AuditLogReader.filter(
            iter(events),
            start_time="2025-06-01T00:00:00Z",
            end_time="2025-07-01T00:00:00Z",
        ))
        assert len(result) == 1
        assert result[0].rule_id == "R002"

    def test_filter_combined_conditions(self):
        """组合条件 AND 过滤。"""
        result = list(AuditLogReader.filter(
            iter(self.EVENTS),
            decision="BLOCK",
            min_risk_score=70,
            tool_name="write_file",
        ))
        assert len(result) == 1
        assert result[0].rule_id == "R002"
        assert result[0].risk_score == 80

    def test_filter_no_match_returns_empty(self):
        """无匹配条件返回空生成器。"""
        result = list(AuditLogReader.filter(
            iter(self.EVENTS), decision="ALLOW", min_risk_score=80
        ))
        assert result == []

    def test_filter_preserves_generator_laziness(self):
        """过滤器是惰性的，不消耗生成器。"""
        gen = iter(self.EVENTS)
        filtered = AuditLogReader.filter(gen, decision="ALLOW")

        # 在遍历前，生成器未被消耗
        first = next(filtered)
        assert first.decision == "ALLOW"

        # 剩余事件仍在生成器中
        remaining = list(gen)  # gen 已被 next(filtered) 消耗了 2 个元素（ALLOW R001 后的）
        # filter 跳过了 BLOCK，所以 gen 前进到 BLOCK R002 后继续
        assert len(remaining) >= 3  # 还有 R002 BLOCK, R001 BLOCK, REPAIR, ALLOW...


# ═══════════════════════════════════════════════════════════════
# 统计聚合测试
# ═══════════════════════════════════════════════════════════════

class TestGetStatistics:
    EVENTS = [
        _make_event(decision="ALLOW", rule_id="R001", risk_score=0,
                     tool_name="read_file", phase="executing"),
        _make_event(decision="BLOCK", rule_id="R002", risk_score=80,
                     tool_name="write_file", phase="planning"),
        _make_event(decision="BLOCK", rule_id="R002", risk_score=60,
                     tool_name="write_file", phase="executing"),
        _make_event(decision="REPAIR", rule_id="R003", risk_score=40,
                     tool_name="execute_python", phase="repairing"),
        _make_event(decision="ALLOW", rule_id="R001", risk_score=0,
                     tool_name="read_file", phase="executing"),
    ]

    def test_total_events(self):
        """统计总事件数。"""
        stats = AuditLogReader.get_statistics(iter(self.EVENTS))
        assert stats["total_events"] == 5

    def test_by_decision(self):
        """按决策类型分布。"""
        stats = AuditLogReader.get_statistics(iter(self.EVENTS))
        assert stats["by_decision"] == {
            "ALLOW": 2, "BLOCK": 2, "REPAIR": 1,
        }

    def test_by_rule_id(self):
        """按规则 ID 分布。"""
        stats = AuditLogReader.get_statistics(iter(self.EVENTS))
        assert stats["by_rule_id"]["R001"] == 2
        assert stats["by_rule_id"]["R002"] == 2
        assert stats["by_rule_id"]["R003"] == 1

    def test_by_tool(self):
        """按工具分布。"""
        stats = AuditLogReader.get_statistics(iter(self.EVENTS))
        assert stats["by_tool"]["read_file"] == 2
        assert stats["by_tool"]["write_file"] == 2
        assert stats["by_tool"]["execute_python"] == 1

    def test_by_phase(self):
        """按阶段分布。"""
        stats = AuditLogReader.get_statistics(iter(self.EVENTS))
        assert stats["by_phase"]["executing"] == 3
        assert stats["by_phase"]["planning"] == 1
        assert stats["by_phase"]["repairing"] == 1

    def test_avg_and_max_risk_score(self):
        """平均和最高风险评分。"""
        stats = AuditLogReader.get_statistics(iter(self.EVENTS))
        assert stats["avg_risk_score"] == 36.0  # (0+80+60+40+0) / 5
        assert stats["max_risk_score"] == 80

    def test_p0_count(self):
        """P0 事件计数（BLOCK + REPAIR）。"""
        stats = AuditLogReader.get_statistics(iter(self.EVENTS))
        assert stats["p0_count"] == 3  # 2 BLOCK + 1 REPAIR

    def test_top_blocked_tools(self):
        """BLOCK 最多的工具排名。"""
        stats = AuditLogReader.get_statistics(iter(self.EVENTS))
        top = stats["top_blocked_tools"]
        assert len(top) == 1  # 只有 write_file 被 BLOCK
        assert top[0][0] == "write_file"
        assert top[0][1] == 2

    def test_time_range(self):
        """时间范围统计。"""
        events = [
            _make_event(timestamp="2025-03-01T00:00:00Z"),
            _make_event(timestamp="2025-06-15T12:00:00Z"),
            _make_event(timestamp="2025-01-01T00:00:00Z"),
        ]
        stats = AuditLogReader.get_statistics(iter(events))
        assert stats["time_range"]["start"] == "2025-01-01T00:00:00Z"
        assert stats["time_range"]["end"] == "2025-06-15T12:00:00Z"

    def test_empty_input(self):
        """空输入返回零值统计。"""
        stats = AuditLogReader.get_statistics(iter([]))
        assert stats["total_events"] == 0
        assert stats["avg_risk_score"] == 0.0
        assert stats["max_risk_score"] == 0
        assert stats["p0_count"] == 0
        assert stats["top_blocked_tools"] == []


# ═══════════════════════════════════════════════════════════════
# 文件轮转检测测试
# ═══════════════════════════════════════════════════════════════

class TestFileRotation:
    def test_truncation_resets_cursor(self):
        """文件被截断时游标重置为 0。"""
        path = _temp_jsonl_path()
        try:
            # 先写 3 条事件
            _write_jsonl(path, [
                _make_event(rule_id="R001"),
                _make_event(rule_id="R002"),
                _make_event(rule_id="R003"),
            ])

            reader = AuditLogReader(path)
            cursor = LogCursor(file_path=path)
            list(reader.read_since(cursor))
            assert cursor.last_offset > 0
            assert cursor.last_position == 3

            # 截断文件：重写为仅 1 条事件（文件更小）
            _write_jsonl(path, [_make_event(rule_id="R-short")])

            # 再次读取，游标应检测到截断并重置，读到新事件
            events = list(reader.read_since(cursor))
            assert len(events) == 1, (
                f"截断后应重新读取文件，预期 1 条事件，实际 {len(events)}"
            )
            assert events[0].rule_id == "R-short"
        finally:
            os.unlink(path)

    def test_count_new_events(self):
        """count() 返回新增事件数。"""
        path = _temp_jsonl_path()
        try:
            _write_jsonl(path, [
                _make_event(rule_id="R001"),
                _make_event(rule_id="R002"),
                _make_event(rule_id="R003"),
            ])

            reader = AuditLogReader(path)
            assert reader.count() == 3

            cursor = LogCursor(file_path=path)
            _ = list(reader.read_since(cursor))

            # 追加事件
            with open(path, 'a', encoding='utf-8') as f:
                f.write(_make_event(rule_id="R004").to_json() + '\n')

            assert reader.count(cursor) == 1
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════
# 集成：read_all → filter → get_statistics 链式调用
# ═══════════════════════════════════════════════════════════════

class TestChainedUsage:
    def test_read_filter_stats_chain(self):
        """端到端：读取 → 过滤 → 统计。"""
        path = _temp_jsonl_path()
        try:
            _write_jsonl(path, [
                _make_event(decision="ALLOW", rule_id="R001", risk_score=0,
                             tool_name="read_file"),
                _make_event(decision="BLOCK", rule_id="R002", risk_score=80,
                             tool_name="write_file"),
                _make_event(decision="BLOCK", rule_id="R002", risk_score=60,
                             tool_name="write_file"),
                _make_event(decision="REPAIR", rule_id="R003", risk_score=40,
                             tool_name="execute_python"),
                _make_event(decision="ALLOW", rule_id="R001", risk_score=0,
                             tool_name="search_content"),
            ])

            reader = AuditLogReader(path)
            all_events = reader.read_all()

            # 过滤出 P0 事件（BLOCK + REPAIR）
            p0_events = AuditLogReader.filter(
                all_events, decision=["BLOCK", "REPAIR"]
            )

            # 统计
            stats = AuditLogReader.get_statistics(p0_events)
            assert stats["total_events"] == 3
            assert stats["p0_count"] == 3
            assert stats["by_decision"]["BLOCK"] == 2
            assert stats["by_decision"]["REPAIR"] == 1
        finally:
            os.unlink(path)

    def test_incremental_with_filter(self):
        """增量读取 + 过滤器。"""
        path = _temp_jsonl_path()
        try:
            _write_jsonl(path, [
                _make_event(decision="ALLOW", rule_id="R001", risk_score=0),
                _make_event(decision="BLOCK", rule_id="R002", risk_score=80),
            ])

            reader = AuditLogReader(path)
            cursor = LogCursor(file_path=path)

            blocked = list(AuditLogReader.filter(
                reader.read_since(cursor),
                decision="BLOCK",
            ))
            assert len(blocked) == 1
            assert blocked[0].rule_id == "R002"
            assert cursor.last_position == 2
        finally:
            os.unlink(path)
