"""
消息类型定义 — 沿用 OpenAI messages 格式以保证最大兼容性。

消息流转示意：
    SystemMessage → UserMessage → AssistantMessage
        ↻ 如果有 tool_calls → ToolMessage → AssistantMessage
        → 最终 AssistantMessage (content)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SystemMessage:
    """系统提示词，定义 Agent 的行为和规则。"""
    content: str
    role: str = "system"


@dataclass
class UserMessage:
    """用户输入的消息。"""
    content: str
    role: str = "user"


@dataclass
class ToolCall:
    """LLM 返回的工具调用请求。"""
    id: str
    function_name: str
    arguments: dict


@dataclass
class AssistantMessage:
    """
    LLM 的回复。

    可能包含文本回复（content），也可能包含工具调用请求（tool_calls），
    或两者都有。
    """
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None

    def has_tool_calls(self) -> bool:
        return self.tool_calls is not None and len(self.tool_calls) > 0


@dataclass
class ToolMessage:
    """工具执行后返回给 LLM 的结果。"""
    content: str
    tool_call_id: str
    name: str
    role: str = "tool"


# 统一的 Message 联合类型
Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage
