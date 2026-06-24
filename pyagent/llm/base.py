"""
LLMProvider 抽象基类 — 定义 LLM 调用的统一接口。

所有 LLM provider（OpenAI、DeepSeek、Anthropic 等）必须实现此接口。
"""

from abc import ABC, abstractmethod
from ..core.message import AssistantMessage


class LLMProvider(ABC):
    """
    LLM Provider 抽象基类。

    封装 LLM API 的调用细节，向 Agent 暴露统一的 generate() 接口。

    子类需要实现：
        async generate(messages, tools) -> AssistantMessage
    """

    @abstractmethod
    async def generate(
        self, messages: list, tools: list[dict]
    ) -> AssistantMessage:
        """
        调用 LLM API，返回 AssistantMessage。

        Args:
            messages: 消息历史列表（SystemMessage | UserMessage | AssistantMessage | ToolMessage）
            tools:    工具定义列表（OpenAI function calling 格式）

        Returns:
            AssistantMessage — 包含文本回复和/或工具调用请求

        Raises:
            LLMCallError: API 调用失败时抛出
        """
        ...


class LLMCallError(Exception):
    """LLM API 调用失败时抛出。"""

    def __init__(self, message: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(message)
