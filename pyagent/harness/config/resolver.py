"""
环境变量占位符解析器。

支持 ${ENV_VAR} 和 ${ENV_VAR:default} 两种语法。
递归处理 str / dict / list，确保 YAML 中任何层级都能引用环境变量。
"""

import os
import re
from typing import Any

# 匹配 ${VAR_NAME} 或 ${VAR_NAME:default_value}
_ENV_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def resolve_env_vars(value: Any) -> Any:
    """
    递归解析值中的 ${ENV_VAR} 占位符。

    支持类型：str, dict, list，其他类型原样返回。

    Example:
        >>> os.environ['KEY'] = 'sk-123'
        >>> resolve_env_vars('${KEY}')
        'sk-123'
        >>> resolve_env_vars('${MISSING:default}')
        'default'
        >>> resolve_env_vars({'url': '${HOST}:${PORT:8080}'})
        {'url': '...:8080'}
    """
    if isinstance(value, str):
        return _resolve_string(value)
    elif isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    return value


def _resolve_string(text: str) -> str:
    """替换字符串中的环境变量占位符。"""

    def _replace(match):
        var_name = match.group(1)
        default = match.group(2)
        env_value = os.getenv(var_name)
        if env_value is not None:
            return env_value
        if default is not None:
            return default
        # 无环境变量且无默认值 → 保留原占位符（避免静默失败）
        return match.group(0)

    return _ENV_PATTERN.sub(_replace, text)
