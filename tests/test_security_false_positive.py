"""
安全治理误报测试 — 负向验证：「不该拦的绝对不能拦」。

覆盖 10 个边界场景：
    1. read_file 读取含 "delete" 关键词的日志文件
    2. execute_python 代码中包含 os.system 字符串字面量
    3. write_file + execute_python 操作不同文件（字段级精炼应放行）
    4. write_file + search_content（search 不在任何高危组合中）
    5. 独立的 execute_python 无前置 write_file
    6. delete_file 在 EXECUTING 阶段（阶段限制允许）
    7. 合法文件名 "a..b.txt"（非路径穿越）
    8. 命令包含 "rm" 子串（如 "python form_reader.py"）
    9. write_file 过期后 execute_python（窗口外不触发 combo）
    10. execute_command 使用允许的前缀 "git push"
"""

import sys
sys.path.insert(0, '.')
import time
import pytest
from pyagent.harness.context.security_governance import (
    SecurityDecision, ExecutionContext, SecurityGovernance,
)
from pyagent.harness.context.session_risk_context import SessionRiskContext
from pyagent.harness.context.parameter_validator import ParameterWhitelistValidator
from pyagent.harness.context.combo_rules import ComboRule, ComboRuleEngine


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _make_gov(**kwargs):
    """创建 SecurityGovernance 实例，默认关闭所有检查。"""
    params = dict(
        phase_restrictions_enabled=False,
        combo_detection_enabled=False,
        param_whitelist_enabled=False,
    )
    params.update(kwargs)
    return SecurityGovernance(**params)


class TestComboFalsePositives:
    """组合风险检测 — 不该拦的场景。"""

    @pytest.fixture
    def risk(self):
        return SessionRiskContext(window_seconds=300.0)

    def test_read_file_with_delete_in_filename_not_blocked(self, risk):
        """场景 1: read_file 读取含 delete 关键词的文件，不触发 HIGH_RISK_COMBOS
        因为 read_file 不在 delete_file 组合中。"""
        gov = _make_gov(combo_detection_enabled=True, session_risk=risk)
        risk.record_call("read_file", {"path": "logs/delete_user_2024.log"})
        ctx = ExecutionContext(phase="executing")
        d = gov.check("search_content", {"pattern": "ERROR"}, ctx)
        assert d.allowed, f"read_file + search_content should NOT be blocked: {d.reason}"

    def test_execute_python_with_os_system_string_not_blocked(self, risk):
        """场景 2: execute_python 执行包含 os.system 字符串的文档解析代码，
        参数校验器不应将 Python 代码当作 shell 命令检查。"""
        gov = _make_gov(
            combo_detection_enabled=True,
            param_whitelist_enabled=True,
            session_risk=risk,
            param_validator=ParameterWhitelistValidator(),
        )
        ctx = ExecutionContext(phase="executing")
        # execute_python 的参数校验走 _validate_path，不是 _validate_command
        d = gov.check("execute_python", {
            "code": '# 解析用户脚本\nif "os.system" in content:\n    warn_user()',
            "path": "analyzer.py",
        }, ctx)
        assert d.allowed, f"execute_python with os.system string should NOT be blocked: {d.reason}"

    def test_different_files_not_blocked_by_refinement(self, risk):
        """场景 3: write_file("log.txt") + execute_python("data_process.py")
        操作不同文件，字段级精炼应放行。"""
        rules = [
            ComboRule(name="w_e_same", sequence=["write_file", "execute_python"],
                      match_on="file_path"),
        ]
        engine = ComboRuleEngine(rules)
        gov = _make_gov(
            combo_detection_enabled=True,
            session_risk=risk,
            combo_rule_engine=engine,
        )
        risk.record_call("write_file", {"path": "log.txt"})
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_python", {"code": "x", "path": "data_process.py"}, ctx)
        assert d.allowed, (
            f"write_file(log.txt) + execute_python(data_process.py) different files"
            f" should NOT be blocked by refinement: {d.reason}"
        )

    def test_write_file_plus_search_content_not_blocked(self, risk):
        """场景 4: write_file + search_content — search 不在 HIGH_RISK_COMBOS 中。"""
        gov = _make_gov(combo_detection_enabled=True, session_risk=risk)
        risk.record_call("write_file", {"path": "notes.md"})
        ctx = ExecutionContext(phase="executing")
        d = gov.check("search_content", {"pattern": "TODO"}, ctx)
        assert d.allowed, (
            f"write_file + search_content should NOT match any combo: {d.reason}"
        )

    def test_standalone_execute_python_not_blocked(self, risk):
        """场景 5: 独立的 execute_python 无前置 write_file，不触发组合。"""
        gov = _make_gov(combo_detection_enabled=True, session_risk=risk)
        # Session is empty — no prior write_file
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_python", {"code": "print(1+1)"}, ctx)
        assert d.allowed, (
            f"Standalone execute_python should NOT be blocked: {d.reason}"
        )

    def test_delete_file_in_executing_phase_not_blocked(self):
        """场景 6: delete_file 在 EXECUTING 阶段，阶段限制允许。"""
        gov = _make_gov(phase_restrictions_enabled=True)
        ctx = ExecutionContext(phase="executing")
        d = gov.check("delete_file", {"path": "temp.txt"}, ctx)
        assert d.allowed, (
            f"delete_file in EXECUTING phase should NOT be blocked by phase restriction: {d.reason}"
        )

    def test_expired_write_not_triggering_combo(self, risk):
        """场景 9: write_file 在滑动窗口外过期后，execute_python 不触发组合。"""
        risk._window_s = 0.05  # 50ms
        risk.record_call("write_file", {"path": "old.py"})
        time.sleep(0.1)  # Wait for expiry
        gov = _make_gov(combo_detection_enabled=True, session_risk=risk)
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_python", {"code": "x"}, ctx)
        assert d.allowed, (
            f"Expired write_file should NOT trigger combo with execute_python: {d.reason}"
        )

    def test_read_file_then_search_content_in_executing_allowed(self, risk):
        """read_file + search_content 不在 HIGH_RISK_COMBOS 中（两者都是只读工具）。
        delete_file 本身是单工具高危组合，此测试改为两个安全工具的组合。"""
        gov = _make_gov(
            combo_detection_enabled=True,
            phase_restrictions_enabled=True,
            session_risk=risk,
        )
        risk.record_call("read_file", {"path": "a.txt"})
        ctx = ExecutionContext(phase="executing")
        d = gov.check("search_content", {"pattern": "TODO"}, ctx)
        assert d.allowed, (
            f"read_file + search_content is not a HIGH_RISK_COMBO: {d.reason}"
        )


class TestParamValidatorFalsePositives:
    """参数白名单校验 — 不该拦的场景。"""

    @pytest.fixture
    def gov(self):
        return _make_gov(
            param_whitelist_enabled=True,
            param_validator=ParameterWhitelistValidator(),
        )

    def test_legitimate_filename_with_dots_not_path_traversal(self, gov):
        """场景 7: "a..b.txt" 是合法文件名（双点），不是路径穿越。"""
        ctx = ExecutionContext(phase="executing")
        d = gov.check("write_file", {"path": "a..b.txt"}, ctx)
        assert d.allowed, (
            f"Filename 'a..b.txt' is NOT path traversal: {d.reason}"
        )

    def test_command_with_rm_substring_not_blocked(self, gov):
        """场景 8: "python form_reader.py" 包含 'rm' 子串但不是危险命令。"""
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_command", {"command": "python form_reader.py"}, ctx)
        assert d.allowed, (
            f"'python form_reader.py' should NOT be blocked: {d.reason}"
        )

    def test_git_push_is_allowed_prefix(self, gov):
        """场景 10: 'git push' 在允许前缀列表中。"""
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_command", {"command": "git push origin main"}, ctx)
        assert d.allowed, (
            f"'git push' should be allowed: {d.reason}"
        )

    def test_pip_install_is_allowed_prefix(self, gov):
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_command", {"command": "pip install pytest"}, ctx)
        assert d.allowed, f"'pip install' should be allowed: {d.reason}"

    def test_echo_command_is_allowed(self, gov):
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_command", {"command": 'echo "rm -rf is dangerous"'}, ctx)
        assert d.allowed, (
            f"echo with rm string literal should NOT be blocked: {d.reason}"
        )

    def test_mkdir_command_is_allowed(self, gov):
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_command", {"command": "mkdir -p build/output"}, ctx)
        assert d.allowed, f"'mkdir' should be allowed: {d.reason}"

    def test_cp_command_is_allowed(self, gov):
        ctx = ExecutionContext(phase="executing")
        d = gov.check("execute_command", {"command": "cp src/main.py dest/main.py"}, ctx)
        assert d.allowed, f"'cp' should be allowed: {d.reason}"

    def test_python_script_path_not_path_traversal(self, gov):
        """Python 脚本的正常输出路径不触发穿越检测。"""
        ctx = ExecutionContext(phase="executing")
        d = gov.check("write_file", {"path": "output/data_2024.json"}, ctx)
        assert d.allowed, f"Normal path should NOT be blocked: {d.reason}"

    def test_https_public_url_allowed(self, gov):
        ctx = ExecutionContext(phase="executing")
        d = gov.check("http_request", {"url": "https://pypi.org/simple/pytest/"}, ctx)
        assert d.allowed, f"Public HTTPS URL should be allowed: {d.reason}"
