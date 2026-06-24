"""
驾驭工程 — 配置管理模块。

提供 YAML 配置的加载、校验和环境变量替换。
配置与核心逻辑解耦：此模块仅负责"读"和"验"，
AgentConfig 仍为唯一内部配置载体。
"""

from .schema import (
    HarnessYamlConfig,
    LLMSchema,
    AgentSchema,
    MemorySchema,
    TokenBudgetSchema,
)
from .resolver import resolve_env_vars
from .loader import ConfigLoader, ConfigLoadError

__all__ = [
    "ConfigLoader",
    "ConfigLoadError",
    "HarnessYamlConfig",
    "LLMSchema",
    "AgentSchema",
    "MemorySchema",
    "TokenBudgetSchema",
    "resolve_env_vars",
]
