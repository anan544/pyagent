"""
驾驭工程配置模块测试。

覆盖：
    - Schema 默认值与校验
    - 环境变量解析器
    - ConfigLoader 多环境加载
    - 错误处理（缺失字段、类型错误）
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from pyagent.harness.config import (
    ConfigLoader,
    ConfigLoadError,
    HarnessYamlConfig,
    LLMSchema,
    AgentSchema,
    MemorySchema,
    TokenBudgetSchema,
    resolve_env_vars,
)


# ═══════════════════════════════════════════════════════════════
# Schema 默认值测试
# ═══════════════════════════════════════════════════════════════

class TestSchemaDefaults:
    """测试各 Schema 的默认值。"""

    def test_llm_schema_defaults(self):
        """LLMSchema 应该有合理的默认值。"""
        llm = LLMSchema()
        assert llm.provider == "openai_compat"
        assert llm.model == "deepseek-chat"
        assert llm.temperature == 0.0
        assert llm.timeout == 120.0
        assert llm.max_retries == 2

    def test_agent_schema_defaults(self):
        """AgentSchema 默认工具列表应包含 4 个内置工具。"""
        agent = AgentSchema()
        assert "read_file" in agent.tools
        assert "write_file" in agent.tools
        assert "execute_python" in agent.tools
        assert "search_content" in agent.tools
        assert agent.max_iterations == 20

    def test_memory_schema_defaults(self):
        """MemorySchema 应有合理的默认值。"""
        memory = MemorySchema()
        assert memory.db_path == "pyagent_memory.db"
        assert memory.load_limit == 1000
        assert memory.token_budget.enabled is True

    def test_harness_config_defaults(self):
        """完整配置应有所有三层。"""
        config = HarnessYamlConfig()
        assert config.llm.model == "deepseek-chat"
        assert config.agent.max_iterations == 20
        assert config.memory.db_path == "pyagent_memory.db"

    def test_harness_config_partial_override(self):
        """部分覆盖字段应保留其他默认值。"""
        config = HarnessYamlConfig(
            llm={"model": "gpt-4o"},
            agent={"max_iterations": 10},
        )
        assert config.llm.model == "gpt-4o"
        assert config.llm.temperature == 0.0  # 未改，保持默认
        assert config.agent.max_iterations == 10
        assert config.agent.tools == ["read_file", "write_file",
                                       "execute_python", "search_content"]


class TestSchemaValidation:
    """测试 Schema 的类型校验。"""

    def test_llm_temperature_range(self):
        """temperature 应在 [0, 2] 范围内。"""
        LLMSchema(temperature=0.0)
        LLMSchema(temperature=2.0)
        with pytest.raises(Exception):
            LLMSchema(temperature=-0.1)
        with pytest.raises(Exception):
            LLMSchema(temperature=2.1)

    def test_agent_max_iterations_range(self):
        """max_iterations 应在 [1, 100] 范围内。"""
        AgentSchema(max_iterations=1)
        AgentSchema(max_iterations=100)
        with pytest.raises(Exception):
            AgentSchema(max_iterations=0)
        with pytest.raises(Exception):
            AgentSchema(max_iterations=101)

    def test_token_budget_safety_factor_range(self):
        """safety_factor 应在 (0, 1] 范围内。"""
        TokenBudgetSchema(safety_factor=0.1)
        TokenBudgetSchema(safety_factor=1.0)
        with pytest.raises(Exception):
            TokenBudgetSchema(safety_factor=0)
        with pytest.raises(Exception):
            TokenBudgetSchema(safety_factor=1.1)


# ═══════════════════════════════════════════════════════════════
# 环境变量解析器测试
# ═══════════════════════════════════════════════════════════════

class TestEnvResolver:
    """测试 ${ENV_VAR} 占位符解析。"""

    def test_simple_substitution(self):
        """基本的环境变量替换。"""
        os.environ["TEST_VAR"] = "test_value"
        result = resolve_env_vars("${TEST_VAR}")
        assert result == "test_value"
        del os.environ["TEST_VAR"]

    def test_missing_with_default(self):
        """缺失时使用默认值。"""
        if "MISSING_VAR" in os.environ:
            del os.environ["MISSING_VAR"]
        result = resolve_env_vars("${MISSING_VAR:fallback}")
        assert result == "fallback"

    def test_missing_without_default(self):
        """缺失且无默认值时应保留原占位符。"""
        if "MISSING_VAR" in os.environ:
            del os.environ["MISSING_VAR"]
        result = resolve_env_vars("${MISSING_VAR}")
        assert result == "${MISSING_VAR}"  # 保留原样，避免静默失败

    def test_dict_recursive(self):
        """递归处理字典中的环境变量。"""
        os.environ["HOST"] = "localhost"
        os.environ["PORT"] = "8080"

        data = {"url": "http://${HOST}:${PORT}", "nested": {"key": "${HOST}"}}
        result = resolve_env_vars(data)

        assert result["url"] == "http://localhost:8080"
        assert result["nested"]["key"] == "localhost"

        del os.environ["HOST"]
        del os.environ["PORT"]

    def test_list_recursive(self):
        """递归处理列表中的环境变量。"""
        os.environ["A"] = "1"

        data = ["${A}", "b", "${A}"]
        result = resolve_env_vars(data)
        assert result == ["1", "b", "1"]

        del os.environ["A"]

    def test_non_string_passthrough(self):
        """非字符串类型原样返回。"""
        assert resolve_env_vars(123) == 123
        assert resolve_env_vars(True) is True
        assert resolve_env_vars(None) is None

    def test_multiple_in_one_string(self):
        """字符串中多个占位符。"""
        os.environ["X"] = "hello"
        os.environ["Y"] = "world"
        result = resolve_env_vars("${X} ${Y}")
        assert result == "hello world"
        del os.environ["X"]
        del os.environ["Y"]


# ═══════════════════════════════════════════════════════════════
# ConfigLoader 测试
# ═══════════════════════════════════════════════════════════════

class TestConfigLoader:
    """测试 ConfigLoader 的文件加载和多环境选择。"""

    def test_load_valid_yaml(self):
        """加载有效的 YAML 文件。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump({
                "llm": {"model": "gpt-4o"},
                "agent": {"max_iterations": 15},
            }, f)
            temp_path = f.name

        try:
            config = ConfigLoader.load(temp_path)
            assert config.llm.model == "gpt-4o"
            assert config.agent.max_iterations == 15
            # 未指定的字段保持默认值
            assert config.llm.temperature == 0.0
        finally:
            Path(temp_path).unlink()

    def test_load_empty_yaml(self):
        """加载空 YAML 应返回全默认配置。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            temp_path = f.name

        try:
            config = ConfigLoader.load(temp_path)
            assert isinstance(config, HarnessYamlConfig)
            assert config.llm.model == "deepseek-chat"  # 默认值
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        """加载不存在的文件应抛出 ConfigLoadError。"""
        with pytest.raises(ConfigLoadError, match="文件不存在"):
            ConfigLoader.load("/nonexistent/path/config.yaml")

    def test_load_with_env_vars(self):
        """YAML 中的 ${ENV_VAR} 应被解析。"""
        os.environ["TEST_MODEL"] = "env-model"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump({"llm": {"model": "${TEST_MODEL}"}}, f)
            temp_path = f.name

        try:
            config = ConfigLoader.load(temp_path)
            assert config.llm.model == "env-model"
        finally:
            Path(temp_path).unlink()
            del os.environ["TEST_MODEL"]

    def test_load_by_env_dev(self):
        """PYAGENT_ENV=dev 应加载 config.dev.yaml。"""
        import os as _os
        with tempfile.TemporaryDirectory() as tmpdir:
            dev_path = Path(tmpdir) / "config.dev.yaml"
            with open(dev_path, "w", encoding="utf-8") as f:
                yaml.dump({"agent": {"max_iterations": 99}}, f)

            old_cwd = _os.getcwd()
            try:
                _os.chdir(tmpdir)
                _os.environ["PYAGENT_ENV"] = "dev"
                config = ConfigLoader.load_by_env()
                assert config.agent.max_iterations == 99
            finally:
                _os.chdir(old_cwd)
                if "PYAGENT_ENV" in _os.environ:
                    del _os.environ["PYAGENT_ENV"]

    def test_load_by_env_fallback(self):
        """无匹配文件时回退到 config.yaml。"""
        import os as _os
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback = Path(tmpdir) / "config.yaml"
            with open(fallback, "w", encoding="utf-8") as f:
                yaml.dump({"agent": {"max_iterations": 42}}, f)

            old_cwd = _os.getcwd()
            try:
                _os.chdir(tmpdir)
                config = ConfigLoader.load_by_env(env="nonexistent")
                assert config.agent.max_iterations == 42
            finally:
                _os.chdir(old_cwd)

    def test_load_by_env_ultimate_fallback(self):
        """无任何配置文件时返回全默认值。"""
        import os as _os
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = _os.getcwd()
            try:
                _os.chdir(tmpdir)
                config = ConfigLoader.load_by_env(env="nonexistent")
                assert isinstance(config, HarnessYamlConfig)
                assert config.agent.max_iterations == 20  # 默认
            finally:
                _os.chdir(old_cwd)

    def test_from_dict(self):
        """从字典加载配置。"""
        config = ConfigLoader.from_dict({
            "llm": {"model": "gpt-4"},
            "agent": {"tools": ["read_file"]},
        })
        assert config.llm.model == "gpt-4"
        assert config.agent.tools == ["read_file"]

    def test_from_dict_validation_error(self):
        """字典类型错误应抛出 ConfigLoadError。"""
        with pytest.raises(ConfigLoadError, match="配置校验失败"):
            ConfigLoader.from_dict({
                "agent": {"max_iterations": 999}  # 超过 100
            })

    def test_validation_error_with_path(self):
        """校验错误应包含文件路径。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            # max_iterations 超出范围 [1, 100] 应触发校验失败
            yaml.dump({"agent": {"max_iterations": 999}}, f)
            temp_path = f.name

        try:
            with pytest.raises(ConfigLoadError) as exc:
                ConfigLoader.load(temp_path)
            # 错误消息应提示校验失败（不是解析错误）
            assert "配置校验失败" in str(exc.value)
        finally:
            Path(temp_path).unlink()


# ═══════════════════════════════════════════════════════════════
# 工厂函数测试
# ═══════════════════════════════════════════════════════════════

class TestFactory:
    """测试 create_agent_from_yaml 和 create_agent_from_config。"""

    def test_create_from_config(self):
        """从 HarnessYamlConfig 创建 Agent。"""
        from pyagent.harness import create_agent_from_config

        config = HarnessYamlConfig(
            llm={"model": "gpt-4"},
            agent={"max_iterations": 5, "tools": ["read_file"]},
        )

        agent = create_agent_from_config(config)
        assert agent is not None
        assert agent.config.max_iterations == 5
        assert agent.config.model == "gpt-4"
        assert "read_file" in agent.tool_registry.list_names()

    def test_create_from_yaml(self):
        """从 YAML 文件创建 Agent。"""
        from pyagent.harness import create_agent_from_yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump({
                "llm": {"model": "gpt-4"},
                "agent": {
                    "max_iterations": 3,
                    "tools": ["read_file", "search_content"],
                },
            }, f)
            temp_path = f.name

        try:
            agent = create_agent_from_yaml(temp_path)
            assert agent is not None
            assert agent.config.max_iterations == 3
            assert "read_file" in agent.tool_registry.list_names()
            assert "search_content" in agent.tool_registry.list_names()
        finally:
            Path(temp_path).unlink()

    def test_create_from_yaml_nonexistent(self):
        """不存在的 YAML 文件应抛出异常。"""
        from pyagent.harness import create_agent_from_yaml
        with pytest.raises(ConfigLoadError, match="文件不存在"):
            create_agent_from_yaml("/nonexistent/config.yaml")
