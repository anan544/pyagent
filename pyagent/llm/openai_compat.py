"""
OpenAI 兼容 LLM Provider — 支持 OpenAI、DeepSeek 等所有兼容 API。

用法：
    provider = OpenAICompatProvider(
        api_key="sk-xxx",          # 或设环境变量 OPENAI_API_KEY
        base_url="https://api.deepseek.com/v1",  # 可选，默认 OpenAI
        model="deepseek-chat",
    )
    response = await provider.generate(messages, tools)
"""

import json
import os
from typing import Optional
import httpx

from .base import LLMProvider, LLMCallError
from ..core.message import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
)


class OpenAICompatProvider(LLMProvider):
    """
    OpenAI 兼容 API 的 LLM Provider。

    支持所有采用 OpenAI API 格式的服务：
        - OpenAI (gpt-4, gpt-4o, o3-mini, etc.)
        - DeepSeek (deepseek-chat, deepseek-reasoner)
        - 其他兼容服务 (Ollama, vLLM, etc.)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4",
        max_retries: int = 2,
        timeout: float = 120.0,
    ):
        """
        Args:
            api_key: API key。默认从环境变量 OPENAI_API_KEY 读取
            base_url: API 基础 URL。可设为 DeepSeek 等服务地址
            model: 模型名称
            max_retries: 失败重试次数
            timeout: HTTP 请求超时秒数
        """
        # 智能检测 API key: 参数 > 对应服务的环境变量 > OPENAI_API_KEY
        if api_key:
            self.api_key = api_key
        elif "deepseek" in base_url:
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key 未设置。请通过参数 api_key 提供，"
                "或设置环境变量 OPENAI_API_KEY / DEEPSEEK_API_KEY。"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout

    async def generate(
        self, messages: list, tools: list[dict]
    ) -> AssistantMessage:
        """
        调用 OpenAI 兼容 API。

        Args:
            messages: 消息列表
            tools: 工具定义列表

        Returns:
            AssistantMessage — 包含文本回复和/或工具调用
        """
        # 转换消息格式
        api_messages = [self._message_to_dict(m) for m in messages]

        # 构建请求体
        body = {
            "model": self.model,
            "messages": api_messages,
            "temperature": 0.0,  # 代码场景用低温度
        }
        if tools:
            body["tools"] = tools

        # 带重试的 API 调用
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return self._parse_response(data)

            except httpx.HTTPStatusError as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt  # 指数退避: 1s, 2s
                    import asyncio
                    await asyncio.sleep(wait)
                    continue
                raise LLMCallError(
                    f"API 调用失败 (HTTP {e.response.status_code}): {e.response.text[:500]}",
                    original_error=e,
                )
            except httpx.RequestError as e:
                last_error = e
                if attempt < self.max_retries:
                    import asyncio
                    await asyncio.sleep(1)
                    continue
                raise LLMCallError(
                    f"网络请求失败: {e}",
                    original_error=e,
                )

        # 不应到达这里
        raise LLMCallError(
            f"API 调用失败（已重试 {self.max_retries} 次）",
            original_error=last_error,
        )

    # ── 私有方法 ──

    def _message_to_dict(self, msg) -> dict:
        """将内部消息类型转换为 OpenAI API 格式的 dict。"""
        if isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        elif isinstance(msg, UserMessage):
            return {"role": "user", "content": msg.content}
        elif isinstance(msg, AssistantMessage):
            d = {"role": "assistant"}
            if msg.content:
                d["content"] = msg.content
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function_name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return d
        elif isinstance(msg, ToolMessage):
            return {
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
            }
        else:
            # 未知类型，尽力而为
            return {"role": "user", "content": str(msg)}

    def _parse_response(self, data: dict) -> AssistantMessage:
        """解析 OpenAI API 响应为 AssistantMessage。"""
        choice = data["choices"][0]
        msg_data = choice["message"]

        content = msg_data.get("content")
        tool_calls = None

        raw_tool_calls = msg_data.get("tool_calls")
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                try:
                    arguments = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        function_name=tc["function"]["name"],
                        arguments=arguments,
                    )
                )

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls,
        )
