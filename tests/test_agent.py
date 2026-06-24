"""
测试 Agent 核心循环 — 使用 mock 组件验证完整流程。
"""

import asyncio
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyagent.core import (
    Agent,
    AgentConfig,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
)
from pyagent.tools import ToolRegistry
from pyagent.utils.logger import Logger


# ═══════════════════════════════════════════════════════════════
# Mock 组件 — 用于测试 Agent 循环逻辑
# ═══════════════════════════════════════════════════════════════

class _MockTool:
    """测试用的简单工具。"""
    name = "echo"
    description = "回显输入内容"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要回显的文本"}
        },
        "required": ["text"],
    }

    def get_schema(self):
        return {"type": "function", "function": {"name": "echo", "description": "echo", "parameters": self.parameters}}

    async def execute(self, text: str) -> str:
        return f"ECHO: {text}"


class _ControlledLLM:
    """
    可控的 Mock LLM — 按预设序列返回回复。

    每次调用 generate() 从预设回复列表中取出下一个。
    """

    def __init__(self, responses: list[AssistantMessage]):
        self.responses = responses
        self.call_count = 0
        self.calls = []  # 记录每次调用的参数

    async def generate(self, messages: list, tools: list[dict]) -> AssistantMessage:
        self.call_count += 1
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        # 超出的调用返回空回复
        return AssistantMessage(content="超出预期调用次数", tool_calls=None)


# ═══════════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════════

class TestAgentBasic:
    """测试 Agent 的基本行为。"""

    def test_simple_no_tool_response(self):
        """最简单的场景：LLM 直接返回文本，不走工具调用。"""
        llm = _ControlledLLM([
            AssistantMessage(content="你好，我是 Agent。", tool_calls=None),
        ])

        registry = ToolRegistry()
        registry.register(_MockTool())

        config = AgentConfig(max_iterations=10)
        agent = Agent(config=config, tool_registry=registry, llm_provider=llm)

        result = asyncio.run(agent.run("Hello!"))
        assert result == "你好，我是 Agent。"
        assert llm.call_count == 1

    def test_tool_call_then_answer(self):
        """标准 ReAct 流程：工具调用 → 处理结果 → 返回答案。"""
        llm = _ControlledLLM([
            # 第一轮：请求工具调用
            AssistantMessage(
                content="让我用工具处理一下。",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function_name="echo",
                        arguments={"text": "hello world"},
                    )
                ],
            ),
            # 第二轮：收到工具结果，给出最终回复
            AssistantMessage(
                content="工具返回了 ECHO: hello world，任务完成。",
                tool_calls=None,
            ),
        ])

        registry = ToolRegistry()
        registry.register(_MockTool())

        config = AgentConfig(max_iterations=10)
        agent = Agent(config=config, tool_registry=registry, llm_provider=llm)

        result = asyncio.run(agent.run("echo hello"))
        assert "ECHO" in result
        assert llm.call_count == 2

    def test_max_iterations_exceeded(self):
        """LLM 永远要调用工具 → 超过 max_iterations 应该抛异常。"""
        llm = _ControlledLLM([
            AssistantMessage(
                content="我需要调用工具。",
                tool_calls=[
                    ToolCall(
                        id=f"call_{i}",
                        function_name="echo",
                        arguments={"text": f"msg_{i}"},
                    )
                ],
            )
            for i in range(20)  # 20 次全部返回工具调用
        ])

        registry = ToolRegistry()
        registry.register(_MockTool())

        config = AgentConfig(max_iterations=3)
        agent = Agent(config=config, tool_registry=registry, llm_provider=llm)

        with pytest.raises(Exception) as exc_info:
            asyncio.run(agent.run("loop forever"))
        assert "循环" in str(exc_info.value) or "3" in str(exc_info.value)

    def test_tool_error_handling(self):
        """工具执行失败 → 错误信息被包装为 ToolMessage 返回给 LLM。"""
        class _FailingTool:
            name = "failer"
            description = "总是失败的工具"
            parameters = {"type": "object", "properties": {}}

            def get_schema(self):
                return {"type": "function", "function": {"name": "failer"}}

            async def execute(self) -> str:
                raise RuntimeError("工具内部错误")

        llm = _ControlledLLM([
            AssistantMessage(
                content="调用会失败的工具。",
                tool_calls=[
                    ToolCall(id="call_1", function_name="failer", arguments={})
                ],
            ),
            AssistantMessage(
                content="工具失败了，我需要告知用户。",
                tool_calls=None,
            ),
        ])

        registry = ToolRegistry()
        registry.register(_FailingTool())

        config = AgentConfig(max_iterations=10)
        agent = Agent(config=config, tool_registry=registry, llm_provider=llm)

        result = asyncio.run(agent.run("测试错误处理"))
        # 不应该崩溃，LLM 收到了错误信息
        assert llm.call_count == 2
        # 验证第二次调用时消息列表包含错误信息
        second_call_msgs = llm.calls[1]["messages"]
        tool_msgs = [m for m in second_call_msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert "工具内部错误" in tool_msgs[0].content

    def test_multiple_tool_calls_in_one_response(self):
        """LLM 一次返回多个工具调用 → 应全部执行。"""
        llm = _ControlledLLM([
            AssistantMessage(
                content="需要多次调用。",
                tool_calls=[
                    ToolCall(id="c1", function_name="echo", arguments={"text": "A"}),
                    ToolCall(id="c2", function_name="echo", arguments={"text": "B"}),
                    ToolCall(id="c3", function_name="echo", arguments={"text": "C"}),
                ],
            ),
            AssistantMessage(content="所有工具调用完成。", tool_calls=None),
        ])

        registry = ToolRegistry()
        registry.register(_MockTool())

        config = AgentConfig(max_iterations=10)
        agent = Agent(config=config, tool_registry=registry, llm_provider=llm)

        result = asyncio.run(agent.run("echo A B C"))
        assert "所有工具调用完成" in result
        # 第一轮过后应该追加了 3 条 ToolMessage
        round1_msgs = llm.calls[1]["messages"]
        tool_msgs = [m for m in round1_msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 3
        assert "ECHO: A" in tool_msgs[0].content
        assert "ECHO: B" in tool_msgs[1].content
        assert "ECHO: C" in tool_msgs[2].content


class TestMessageFlow:
    """测试消息列表的拼接逻辑。"""

    def test_message_history_structure(self):
        """验证消息历史的结构正确。"""
        llm = _ControlledLLM([
            AssistantMessage(
                tool_calls=[
                    ToolCall(id="c1", function_name="echo", arguments={"text": "X"})
                ],
            ),
            AssistantMessage(content="最终答案"),
        ])

        registry = ToolRegistry()
        registry.register(_MockTool())

        config = AgentConfig(max_iterations=10)
        agent = Agent(config=config, tool_registry=registry, llm_provider=llm)

        asyncio.run(agent.run("test"))

        # 检查第二轮 LLM 调用时的消息列表
        # Agent 循环: Round1 后 messages = [system, user, assistant, tool]
        # Round2 开始时 LLM 看到的是这 4 条消息（新的 AssistantMessage 在返回后才 append）
        msgs = llm.calls[1]["messages"]
        roles = [m.role for m in msgs]
        assert roles == ["system", "user", "assistant", "tool"]
        # 验证 tool_calls 的 assistant 消息在消息列表中
        assert msgs[2].role == "assistant"
        assert msgs[2].has_tool_calls()
        # 验证 tool 结果在消息列表中
        assert msgs[3].role == "tool"
        assert "ECHO: X" in msgs[3].content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
