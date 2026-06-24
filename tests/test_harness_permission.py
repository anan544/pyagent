"""
三级权限门控测试。

覆盖：
    - 计划级预审：高危组合检测 / 无高危正常通过 / 单工具高危
    - 步骤级检查：参数匹配通过 / 参数越界阻断 / 空 params 默认放行
    - ScopedToolRegistry：白名单可调用 / 非白名单未授权 / 原注册表零修改
    - 修补白名单：默认白名单 / 自定义白名单
"""

from unittest.mock import Mock, AsyncMock, patch
import pytest

from pyagent.harness.context.models import Step, ExecutionPlan
from pyagent.harness.context.permission import (
    PermissionGate,
    ScopedToolRegistry,
    HIGH_RISK_COMBOS,
)
from pyagent.tools.registry import ToolRegistry
from pyagent.tools.base import Tool
from pyagent.core.message import ToolMessage


# ═══════════════════════════════════════════════════════════════
# 辅助：创建测试用 Step / ExecutionPlan
# ═══════════════════════════════════════════════════════════════

def _make_step(id: int, description: str, action: str = "read",
               tool: str = None, params: dict = None,
               expected_output: str = "Done",
               acceptance_criteria: str = "OK",
               risk_level: str = "low") -> Step:
    return Step(
        id=id,
        description=description,
        action=action,
        tool=tool,
        params=params or {},
        expected_output=expected_output,
        acceptance_criteria=acceptance_criteria,
        risk_level=risk_level,
    )


def _make_plan(steps: list[Step], **kwargs) -> ExecutionPlan:
    defaults = {
        "steps": steps,
        "plan_summary": "Test plan",
    }
    defaults.update(kwargs)
    return ExecutionPlan(**defaults)


# ═══════════════════════════════════════════════════════════════
# 计划级预审测试
# ═══════════════════════════════════════════════════════════════

class TestPlanLevelAudit:

    def test_no_high_risk_tools_returns_empty(self):
        """所有工具均非高危 → 空结果。"""
        steps = [
            _make_step(0, "Read file", tool="read_file"),
            _make_step(1, "Search code", tool="search_content"),
        ]
        plan = _make_plan(steps)
        findings = PermissionGate.plan_level_audit(plan)
        assert findings == []

    def test_single_high_risk_detected(self):
        """单个高危工具（如 delete_file）→ 应检测到。"""
        steps = [
            _make_step(0, "Delete temp", tool="delete_file"),
        ]
        plan = _make_plan(steps)
        findings = PermissionGate.plan_level_audit(plan)
        assert len(findings) >= 1
        # 应包含 delete_file
        combos_flat = []
        for f in findings:
            combos_flat.extend(f["combo"])
        assert "delete_file" in combos_flat

    def test_high_risk_combo_detected(self):
        """write_file + execute_python 组合 → 应触发告警。"""
        steps = [
            _make_step(0, "Write script", tool="write_file"),
            _make_step(1, "Run script", tool="execute_python"),
        ]
        plan = _make_plan(steps)
        findings = PermissionGate.plan_level_audit(plan)
        assert len(findings) >= 1

    def test_high_risk_combo_shows_affected_steps(self):
        """告警应包含受影响的步骤 ID。"""
        steps = [
            _make_step(0, "Write script", tool="write_file"),
            _make_step(1, "Read config", tool="read_file"),
            _make_step(2, "Run script", tool="execute_python"),
        ]
        plan = _make_plan(steps)
        findings = PermissionGate.plan_level_audit(plan)
        # 应有一个发现包含 write_file + execute_python
        combo_finding = None
        for f in findings:
            if "write_file" in f["combo"] and "execute_python" in f["combo"]:
                combo_finding = f
                break
        assert combo_finding is not None
        assert 0 in combo_finding["steps"]  # write_file step
        assert 2 in combo_finding["steps"]  # execute_python step
        assert combo_finding["severity"] == "high"

    def test_pip_install_with_execute_python_detected(self):
        """pip_install + execute_python 组合 → 应触发。"""
        steps = [
            _make_step(0, "Install deps", tool="pip_install"),
            _make_step(1, "Run code", tool="execute_python"),
        ]
        plan = _make_plan(steps)
        findings = PermissionGate.plan_level_audit(plan)
        assert len(findings) >= 1

    def test_steps_without_tools_ignored(self):
        """无工具的步骤不触发告警。"""
        steps = [
            _make_step(0, "Review code", action="review", tool=None),
            _make_step(1, "Ask user", action="ask", tool=None),
        ]
        plan = _make_plan(steps)
        findings = PermissionGate.plan_level_audit(plan)
        assert findings == []

    def test_finding_contains_reason(self):
        """告警应包含 reason 字段。"""
        steps = [
            _make_step(0, "Delete", tool="delete_file"),
        ]
        plan = _make_plan(steps)
        findings = PermissionGate.plan_level_audit(plan)
        assert len(findings) >= 1
        assert "reason" in findings[0]
        assert "delete_file" in findings[0]["reason"]

    def test_git_push_force_detected(self):
        """git_push + git_force_push → 应触发。"""
        steps = [
            _make_step(0, "Push code", tool="git_push"),
            _make_step(1, "Force push", tool="git_force_push"),
        ]
        plan = _make_plan(steps)
        findings = PermissionGate.plan_level_audit(plan)
        assert len(findings) >= 1


# ═══════════════════════════════════════════════════════════════
# 步骤级执行检查测试
# ═══════════════════════════════════════════════════════════════

class TestStepLevelCheck:

    def test_matching_tool_and_params_passes(self):
        """工具名和参数完全匹配 → True。"""
        step = _make_step(0, "Read", tool="read_file",
                          params={"file_path": "test.py"})
        assert PermissionGate.step_level_check(
            step, "read_file", {"file_path": "test.py"}
        ) is True

    def test_tool_mismatch_fails(self):
        """LLM 调用了计划未声明的工具 → False。"""
        step = _make_step(0, "Read", tool="read_file")
        assert PermissionGate.step_level_check(
            step, "write_file", {"file_path": "test.py"}
        ) is False

    def test_extra_param_fails(self):
        """LLM 传了 Plan 未声明的参数 → False（越界）。"""
        step = _make_step(0, "Read", tool="read_file",
                          params={"file_path": "test.py"})
        assert PermissionGate.step_level_check(
            step, "read_file", {"file_path": "test.py", "offset": 100}
        ) is False

    def test_param_value_mismatch_fails(self):
        """参数值不匹配 → False。"""
        step = _make_step(0, "Read", tool="read_file",
                          params={"file_path": "allowed.py"})
        assert PermissionGate.step_level_check(
            step, "read_file", {"file_path": "forbidden.py"}
        ) is False

    def test_empty_params_allows_anything(self):
        """Plan 未声明 params → 允许任意参数（放行）。"""
        step = _make_step(0, "Read", tool="read_file", params={})
        assert PermissionGate.step_level_check(
            step, "read_file", {"file_path": "anything.py", "offset": 42}
        ) is True

    def test_none_tool_allows_any_tool(self):
        """Plan 声明 tool=None → 允许任意工具。"""
        step = _make_step(0, "Review", action="review", tool=None)
        assert PermissionGate.step_level_check(
            step, "read_file", {"file_path": "x.py"}
        ) is True

    def test_subset_params_passes(self):
        """LLM 参数是 Plan 声明的子集 → True。"""
        step = _make_step(0, "Read", tool="read_file",
                          params={"file_path": "test.py", "offset": 0})
        # 只传 file_path，没传 offset → 通过
        assert PermissionGate.step_level_check(
            step, "read_file", {"file_path": "test.py"}
        ) is True

    def test_partial_match_fails_on_wrong_value(self):
        """部分匹配但值不同 → False。"""
        step = _make_step(0, "Read", tool="read_file",
                          params={"file_path": "a.py", "limit": 100})
        assert PermissionGate.step_level_check(
            step, "read_file", {"file_path": "b.py"}
        ) is False


# ═══════════════════════════════════════════════════════════════
# 修补阶段白名单测试
# ═══════════════════════════════════════════════════════════════

class TestRepairWhitelist:

    def test_default_whitelist(self):
        """无 allowed_repair_tools → 使用默认白名单。"""
        steps = [_make_step(0, "Read", tool="read_file")]
        plan = _make_plan(steps)  # allowed_repair_tools 未设置
        whitelist = PermissionGate.get_repair_whitelist(plan)
        assert whitelist == {"read_file", "search_content"}

    def test_custom_whitelist_from_plan(self):
        """Plan 设置了 allowed_repair_tools → 使用自定义白名单。"""
        steps = [_make_step(0, "Read", tool="read_file")]
        plan = _make_plan(steps, allowed_repair_tools=["read_file", "write_file"])
        whitelist = PermissionGate.get_repair_whitelist(plan)
        assert whitelist == {"read_file", "write_file"}

    def test_empty_whitelist_in_plan_uses_default(self):
        """allowed_repair_tools=[] → 使用默认白名单。"""
        steps = [_make_step(0, "Read", tool="read_file")]
        plan = _make_plan(steps, allowed_repair_tools=[])
        whitelist = PermissionGate.get_repair_whitelist(plan)
        assert whitelist == {"read_file", "search_content"}

    def test_different_tasks_different_whitelists(self):
        """不同任务可使用不同的白名单。"""
        plan_a = _make_plan(
            [_make_step(0, "Read", tool="read_file")],
            allowed_repair_tools=["read_file"],
        )
        plan_b = _make_plan(
            [_make_step(0, "Write", tool="write_file")],
            allowed_repair_tools=["read_file", "search_content", "write_file"],
        )
        assert PermissionGate.get_repair_whitelist(plan_a) == {"read_file"}
        assert PermissionGate.get_repair_whitelist(plan_b) == {
            "read_file", "search_content", "write_file",
        }

    def test_none_plan_uses_default(self):
        """plan=None → 使用默认白名单。"""
        whitelist = PermissionGate.get_repair_whitelist(None)
        assert whitelist == {"read_file", "search_content"}


# ═══════════════════════════════════════════════════════════════
# ScopedToolRegistry 测试
# ═══════════════════════════════════════════════════════════════

# 辅助：创建假 Tool 用于测试
class _FakeTool(Tool):
    def __init__(self, name):
        self.name = name
        self.description = f"Fake {name}"

    async def execute(self, **kwargs):
        return f"executed {self.name}"

    def get_schema(self):
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": {}},
        }


class TestScopedToolRegistry:

    @pytest.fixture
    def parent_registry(self):
        """创建包含 read_file, write_file, execute_python 的注册表。"""
        reg = ToolRegistry()
        reg.register(_FakeTool("read_file"))
        reg.register(_FakeTool("write_file"))
        reg.register(_FakeTool("execute_python"))
        reg.register(_FakeTool("search_content"))
        return reg

    def test_get_all_schemas_only_whitelist(self, parent_registry):
        scoped = ScopedToolRegistry(parent_registry, {"read_file", "search_content"})
        schemas = scoped.get_all_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert set(names) == {"read_file", "search_content"}
        assert "write_file" not in names
        assert "execute_python" not in names

    def test_list_names_only_whitelist(self, parent_registry):
        scoped = ScopedToolRegistry(parent_registry, {"read_file"})
        names = scoped.list_names()
        assert names == ["read_file"]

    @pytest.mark.asyncio
    async def test_whitelisted_tool_executes(self, parent_registry):
        scoped = ScopedToolRegistry(parent_registry, {"read_file"})
        result = await scoped.execute("read_file", "call_1", {"path": "x"})
        assert isinstance(result, ToolMessage)
        assert "executed read_file" in result.content

    @pytest.mark.asyncio
    async def test_non_whitelisted_tool_returns_unauthorized(self, parent_registry):
        """非白名单工具返回「未授权」错误，而非实际执行。"""
        scoped = ScopedToolRegistry(parent_registry, {"read_file"})
        result = await scoped.execute("write_file", "call_2", {"path": "x"})
        assert isinstance(result, ToolMessage)
        assert "未授权" in result.content or "not authorized" in result.content.lower()
        # 不应包含实际执行结果
        assert "executed write_file" not in result.content

    @pytest.mark.asyncio
    async def test_non_whitelisted_delete_file_blocked(self, parent_registry):
        """验证修复阶段故意调用 delete_file 返回未授权。"""
        # 注册 delete_file
        parent_registry.register(_FakeTool("delete_file"))
        scoped = ScopedToolRegistry(parent_registry, {"read_file", "search_content"})
        result = await scoped.execute("delete_file", "call_3", {"path": "important.py"})
        assert isinstance(result, ToolMessage)
        assert "未授权" in result.content or "not authorized" in result.content.lower()

    def test_parent_registry_unchanged(self, parent_registry):
        """ScopedToolRegistry 不修改原 ToolRegistry。"""
        original_names = set(parent_registry.list_names())
        ScopedToolRegistry(parent_registry, {"read_file"})
        assert set(parent_registry.list_names()) == original_names

    def test_allowed_tools_property(self, parent_registry):
        scoped = ScopedToolRegistry(parent_registry, {"read_file"})
        assert scoped.allowed_tools == {"read_file"}

    def test_empty_whitelist(self, parent_registry):
        """空白名单→无工具可用。"""
        scoped = ScopedToolRegistry(parent_registry, set())
        assert scoped.get_all_schemas() == []
        assert scoped.list_names() == []

    @pytest.mark.asyncio
    async def test_empty_whitelist_blocks_all(self, parent_registry):
        scoped = ScopedToolRegistry(parent_registry, set())
        result = await scoped.execute("read_file", "call_4", {})
        assert "未授权" in result.content or "not authorized" in result.content.lower()

    def test_multiple_scoped_instances_independent(self, parent_registry):
        """多个 ScopedToolRegistry 实例互不干扰。"""
        s1 = ScopedToolRegistry(parent_registry, {"read_file"})
        s2 = ScopedToolRegistry(parent_registry, {"write_file"})
        assert s1.allowed_tools == {"read_file"}
        assert s2.allowed_tools == {"write_file"}


# ═══════════════════════════════════════════════════════════════
# HIGH_RISK_COMBOS 配置测试
# ═══════════════════════════════════════════════════════════════

class TestHighRiskCombos:

    def test_combos_are_sets(self):
        for combo in HIGH_RISK_COMBOS:
            assert isinstance(combo, set)

    def test_each_combo_non_empty(self):
        for combo in HIGH_RISK_COMBOS:
            assert len(combo) >= 1

    def test_key_combos_present(self):
        """核心高危组合必须存在。"""
        # write_file + execute_python
        found = False
        for combo in HIGH_RISK_COMBOS:
            if "write_file" in combo and "execute_python" in combo:
                found = True
                break
        assert found, "write_file + execute_python 组合缺失"

        # delete_file 单工具高危
        found = False
        for combo in HIGH_RISK_COMBOS:
            if combo == {"delete_file"}:
                found = True
                break
        assert found, "delete_file 高危标记缺失"
