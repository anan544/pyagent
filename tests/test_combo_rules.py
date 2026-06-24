"""
ComboRuleEngine 字段级精炼规则测试。

覆盖：
    - match_on 匹配/不匹配判断
    - 无精炼规则时维持快路径（返回 None）
    - 字段不存在时的降级行为
    - within_window 范围限制
"""

import sys
sys.path.insert(0, '.')
import pytest
from pyagent.harness.context.combo_rules import ComboRule, ComboRuleEngine


def _make_record(tool_name: str, **params_summary):
    return {
        "tool_name": tool_name,
        "params_summary": params_summary,
        "timestamp": 1000.0,
    }


class TestComboRuleEngine:
    """字段级精炼匹配。"""

    @pytest.fixture
    def engine(self):
        rules = [
            ComboRule(
                name="write_then_exec_same_file",
                sequence=["write_file", "execute_python"],
                match_on="file_path",
                within_window=5,
            ),
        ]
        return ComboRuleEngine(rules)

    def test_match_on_same_file_path_returns_rule_name(self, engine):
        result = engine.match(
            combo={"write_file", "execute_python"},
            current_tool="execute_python",
            current_params={"code": "print(1)", "path": "a.py"},
            recent_records=[
                _make_record("write_file", path="a.py"),
                _make_record("read_file", path="b.txt"),
            ],
        )
        assert result == "write_then_exec_same_file"

    def test_match_on_different_file_path_returns_false(self, engine):
        result = engine.match(
            combo={"write_file", "execute_python"},
            current_tool="execute_python",
            current_params={"code": "print(1)", "file_path": "data_process.py"},
            recent_records=[
                _make_record("write_file", path="log.txt"),
            ],
        )
        assert result is False

    def test_no_rules_returns_none(self):
        engine = ComboRuleEngine([])
        result = engine.match(
            combo={"write_file", "execute_python"},
            current_tool="write_file",
            current_params={"path": "a.py"},
            recent_records=[],
        )
        assert result is None

    def test_unknown_combo_returns_none(self, engine):
        result = engine.match(
            combo={"read_file", "search_content"},
            current_tool="read_file",
            current_params={"path": "a.py"},
            recent_records=[],
        )
        assert result is None

    def test_rule_without_match_on_returns_name(self):
        rules = [
            ComboRule(
                name="any_write_exec",
                sequence=["write_file", "execute_python"],
                match_on=None,
            ),
        ]
        engine = ComboRuleEngine(rules)
        result = engine.match(
            combo={"write_file", "execute_python"},
            current_tool="execute_python",
            current_params={"code": "x"},
            recent_records=[_make_record("write_file", path="unrelated.txt")],
        )
        assert result == "any_write_exec"

    def test_current_missing_match_on_field_returns_none(self, engine):
        result = engine.match(
            combo={"write_file", "execute_python"},
            current_tool="execute_python",
            current_params={"code": "x"},  # no file_path
            recent_records=[_make_record("write_file", path="a.py")],
        )
        assert result is None

    def test_within_window_limits_search(self):
        rules = [
            ComboRule(
                name="strict_window",
                sequence=["write_file", "execute_python"],
                match_on="file_path",
                within_window=1,
            ),
        ]
        engine = ComboRuleEngine(rules)
        # Only the last 1 record is searched — the matching write_file is too old
        result = engine.match(
            combo={"write_file", "execute_python"},
            current_tool="execute_python",
            current_params={"path": "a.py"},
            recent_records=[
                _make_record("write_file", path="a.py"),  # index 0, outside window=1
                _make_record("read_file", path="other.txt"),  # index 1, in window
            ],
        )
        assert result is False

    def test_superset_combo_matches(self):
        rules = [
            ComboRule(
                name="write_exec_cmd",
                sequence=["write_file", "execute_python"],
                match_on="file_path",
            ),
        ]
        engine = ComboRuleEngine(rules)
        # combo is a superset of rule.sequence
        result = engine.match(
            combo={"write_file", "execute_python", "execute_command"},
            current_tool="execute_python",
            current_params={"path": "x.py"},
            recent_records=[_make_record("write_file", path="x.py")],
        )
        assert result == "write_exec_cmd"

    def test_rule_count(self, engine):
        assert engine.rule_count == 1


class TestComboRuleModel:
    """ComboRule Pydantic 模型校验。"""

    def test_valid_rule(self):
        rule = ComboRule(name="test", sequence=["a", "b"], match_on="file_path")
        assert rule.name == "test"
        assert rule.sequence == ["a", "b"]
        assert rule.match_on == "file_path"
        assert rule.within_window == 5

    def test_sequence_min_length(self):
        with pytest.raises(Exception):
            ComboRule(name="bad", sequence=["single"])

    def test_serialize_roundtrip(self):
        rule = ComboRule(name="r", sequence=["x", "y"], match_on="path", within_window=3)
        data = rule.model_dump()
        assert data["name"] == "r"
        restored = ComboRule(**data)
        assert restored == rule
