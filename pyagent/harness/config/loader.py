"""
YAML 配置加载器。

加载流程：读取 YAML 文件 → 解析 ${ENV_VAR} 占位符 → Pydantic 校验 → 返回 HarnessYamlConfig

支持：
    - 直接加载指定文件：ConfigLoader.load("config.dev.yaml")
    - 按环境自动选择：ConfigLoader.load_by_env()  # 读取 PYAGENT_ENV 环境变量
    - 多环境命名约定：config.dev.yaml / config.prod.yaml / config.test.yaml
"""

import os
from pathlib import Path
from typing import Optional

import yaml

from .schema import HarnessYamlConfig
from .resolver import resolve_env_vars


# 环境名 → 默认配置文件名
ENV_CONFIG_MAP = {
    "dev": "config.dev.yaml",
    "development": "config.dev.yaml",
    "prod": "config.prod.yaml",
    "production": "config.prod.yaml",
    "test": "config.test.yaml",
    "testing": "config.test.yaml",
}


class ConfigLoadError(Exception):
    """配置加载失败时抛出，包含具体错误原因和文件路径。"""

    def __init__(self, message: str, path: Optional[str] = None):
        self.path = path
        location = f" ({path})" if path else ""
        super().__init__(f"配置加载失败{location}: {message}")


class ConfigLoader:
    """
    YAML 配置加载器 — 纯函数，无副作用。

    使用方式：
        # 方式 1：直接加载指定文件
        config = ConfigLoader.load("config.dev.yaml")

        # 方式 2：根据 PYAGENT_ENV 环境变量自动选择
        config = ConfigLoader.load_by_env()

        # 方式 3：指定目录和环境
        config = ConfigLoader.load_from_dir("config/", env="prod")

        # 方式 4：从字典加载（测试用）
        config = ConfigLoader.from_dict({"llm": {"model": "gpt-4"}})
    """

    @staticmethod
    def load(path: str | Path) -> HarnessYamlConfig:
        """
        加载并校验单个 YAML 配置文件。

        Args:
            path: YAML 文件路径。

        Returns:
            校验后的 HarnessYamlConfig 实例。

        Raises:
            ConfigLoadError: 文件不存在、YAML 格式错误或校验失败时抛出，
                             包含清晰的错误描述。
        """
        path = Path(path)

        if not path.exists():
            raise ConfigLoadError(f"文件不存在: {path}", str(path))

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"YAML 解析错误: {e}", str(path))
        except Exception as e:
            raise ConfigLoadError(f"文件读取失败: {e}", str(path))

        if raw is None:
            raw = {}

        if not isinstance(raw, dict):
            raise ConfigLoadError(
                f"YAML 顶层必须是字典，实际类型: {type(raw).__name__}",
                str(path),
            )

        return ConfigLoader._validate(raw, str(path))

    @staticmethod
    def load_by_env(
        config_dir: str | Path = ".",
        env: Optional[str] = None,
    ) -> HarnessYamlConfig:
        """
        根据当前环境自动选择配置文件。

        选择优先级：
            1. env 参数（显式指定）
            2. PYAGENT_ENV 环境变量
            3. config.yaml（通用回退）
            4. 全部默认值（最终回退）

        Args:
            config_dir: 配置文件所在目录，默认为当前工作目录。
            env: 环境名称。None 时读取 PYAGENT_ENV 环境变量。

        Returns:
            校验后的 HarnessYamlConfig 实例。
        """
        config_dir = Path(config_dir)
        target_env = (env or os.getenv("PYAGENT_ENV", "")).lower()

        # 尝试按环境名加载
        if target_env and target_env in ENV_CONFIG_MAP:
            env_path = config_dir / ENV_CONFIG_MAP[target_env]
            if env_path.exists():
                return ConfigLoader.load(env_path)

        # 如果环境名不在映射中，尝试直接拼接
        if target_env:
            env_path = config_dir / f"config.{target_env}.yaml"
            if env_path.exists():
                return ConfigLoader.load(env_path)

        # 回退：通用的 config.yaml
        fallback = config_dir / "config.yaml"
        if fallback.exists():
            return ConfigLoader.load(fallback)

        # 最终回退：全部默认值
        return HarnessYamlConfig()

    @staticmethod
    def load_from_dir(
        config_dir: str | Path, env: str = "dev"
    ) -> HarnessYamlConfig:
        """
        从指定目录按环境加载配置。

        Args:
            config_dir: 配置文件所在目录。
            env: 环境名称。

        Returns:
            校验后的 HarnessYamlConfig 实例。
        """
        return ConfigLoader.load_by_env(config_dir, env=env)

    @staticmethod
    def from_dict(data: dict) -> HarnessYamlConfig:
        """
        从字典加载配置（主要用于测试）。

        流程：解析环境变量 → Pydantic 校验。

        Args:
            data: 原始配置字典。

        Returns:
            校验后的 HarnessYamlConfig 实例。

        Raises:
            ConfigLoadError: 校验失败时抛出。
        """
        return ConfigLoader._validate(data, "<dict>")

    # ── 内部 ──────────────────────────────────────────

    @staticmethod
    def _validate(raw: dict, source: str) -> HarnessYamlConfig:
        """校验原始配置字典。"""
        # 1. 解析环境变量占位符
        resolved = resolve_env_vars(raw)

        # 2. Pydantic 校验
        try:
            return HarnessYamlConfig(**resolved)
        except Exception as e:
            raise ConfigLoadError(
                f"配置校验失败 — 请检查字段类型和必填项:\n{e}",
                source,
            )
