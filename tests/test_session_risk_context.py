"""
SessionRiskContext 滑动窗口测试。

覆盖：
    - 记录/查询/过期驱逐
    - params_summary 字段提取（file_path, command, code 等）
    - 窗口容量限制 + reset
"""

import sys
sys.path.insert(0, '.')
import time
import pytest
from pyagent.harness.context.session_risk_context import (
    SessionRiskContext, _extract_params_summary,
)


class TestParamsSummaryExtraction:
    """参数摘要提取逻辑。"""

    def test_extract_file_path_from_write_file(self):
        summary = _extract_params_summary("write_file", {"path": "a.py", "content": "x"})
        assert summary.get("path") == "a.py"

    def test_extract_file_path_alias_from_read_file(self):
        summary = _extract_params_summary("read_file", {"file_path": "b.txt"})
        assert summary.get("file_path") == "b.txt"  # read_file extracts both "path" and "file_path"
        summary2 = _extract_params_summary("read_file", {"path": "b.txt"})
        assert summary2.get("path") == "b.txt"

    def test_extract_command_from_execute_command(self):
        summary = _extract_params_summary("execute_command", {"command": "pytest -v"})
        assert summary.get("command") == "pytest -v"

    def test_extract_code_truncated(self):
        summary = _extract_params_summary("execute_python", {"code": "x" * 200})
        assert len(summary.get("code", "")) <= 80

    def test_empty_params_returns_empty(self):
        summary = _extract_params_summary("write_file", None)
        assert summary == {}

    def test_unknown_tool_returns_empty(self):
        summary = _extract_params_summary("unknown_tool", {"a": 1})
        assert summary == {}

    def test_search_content_extracts_pattern(self):
        summary = _extract_params_summary("search_content", {"pattern": "TODO", "path": "src/"})
        assert summary.get("pattern") == "TODO"


class TestSessionRiskContext:
    """滑动窗口行为测试。"""

    @pytest.fixture
    def ctx(self):
        return SessionRiskContext(window_seconds=300.0, max_records=50)

    def test_empty_context_returns_empty(self, ctx):
        assert ctx.recent_tool_names() == []
        assert ctx.recent_records() == []
        assert ctx.window_size == 0

    def test_record_call_appears_in_recent(self, ctx):
        ctx.record_call("write_file", {"path": "a.py"})
        assert ctx.window_size == 1
        names = ctx.recent_tool_names()
        assert "write_file" in names

    def test_include_current_adds_candidate(self, ctx):
        ctx.record_call("write_file", {"path": "a.py"})
        names = ctx.recent_tool_names(include_current="execute_python")
        assert set(names) == {"write_file", "execute_python"}

    def test_unique_names_deduplicated(self, ctx):
        ctx.record_call("write_file", {"path": "a.py"})
        ctx.record_call("write_file", {"path": "b.py"})
        names = ctx.recent_tool_names()
        assert names == ["write_file"]

    def test_recent_records_has_params_summary(self, ctx):
        ctx.record_call("write_file", {"path": "a.py"})
        records = ctx.recent_records()
        assert len(records) == 1
        assert records[0]["tool_name"] == "write_file"
        assert records[0]["params_summary"].get("path") == "a.py"

    def test_expired_records_removed(self, ctx):
        ctx._window_s = 0.01  # 10ms window
        ctx.record_call("write_file", {"path": "a.py"})
        time.sleep(0.02)
        assert ctx.window_size == 0

    def test_max_records_evicts_oldest(self, ctx):
        ctx._max_records = 3
        ctx.record_call("a", {"path": "1"})
        ctx.record_call("b", {"path": "2"})
        ctx.record_call("c", {"path": "3"})
        ctx.record_call("d", {"path": "4"})
        assert ctx.window_size == 3
        names = ctx.recent_tool_names()
        assert "a" not in names  # oldest evicted

    def test_reset_clears_all(self, ctx):
        ctx.record_call("write_file", {"path": "1"})
        ctx.record_call("execute_python", {"code": "2"})
        ctx.reset()
        assert ctx.window_size == 0

    def test_record_call_without_params(self, ctx):
        ctx.record_call("read_file")
        records = ctx.recent_records()
        assert records[0]["params_summary"] == {}
