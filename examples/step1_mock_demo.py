"""
第一步 Mock 演示 — 用假数据跑通 ReAct 循环。

这个脚本验证 Agent 框架的核心骨架：
    1. 消息列表的初始化和拼接逻辑
    2. LLM 调用 → 工具执行 → 结果追加 的循环
    3. 最终回复的返回和退出

所有 LLM 调用和工具执行都用假数据，不依赖任何外部 API。
运行：python examples/step1_mock_demo.py
"""

import asyncio
import sys
import os

# 将项目根目录加入 path
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
from pyagent.utils.logger import Logger


# ═══════════════════════════════════════════════════════════════
# Mock 实现 — 用假数据模拟 LLM 和工具，验证核心循环
# ═══════════════════════════════════════════════════════════════

class MockToolRegistry:
    """
    模拟的工具注册表。
    注册了一个 fake execute_python 工具，永远返回假结果。
    """

    def __init__(self):
        self._tools = {
            "execute_python": {
                "name": "execute_python",
                "description": "在沙盒中执行 Python 代码，返回标准输出",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "要执行的 Python 代码"
                        }
                    },
                    "required": ["code"]
                }
            }
        }

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_all_schemas(self) -> list[dict]:
        """返回 OpenAI function calling 格式的工具定义。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                }
            }
            for t in self._tools.values()
        ]

    async def execute(self, name: str, call_id: str, arguments: dict) -> ToolMessage:
        """
        模拟工具执行 — 永远返回固定的成功结果。
        这是「占位函数」：先让循环跑通，第二步再替换为真实实现。
        """
        if name == "execute_python":
            code = arguments.get("code", "")
            fake_output = f"[Mock 执行成功] 代码: {code!r} → 输出: Hello World"
            return ToolMessage(
                content=fake_output,
                tool_call_id=call_id,
                name=name,
            )
        else:
            return ToolMessage(
                content=f"[Mock] 未知工具: {name}",
                tool_call_id=call_id,
                name=name,
            )


class MockLLMProvider:
    """
    模拟的 LLM Provider。
    第一次调用返回工具调用请求，第二次调用返回最终文本回复。
    """

    def __init__(self):
        self._call_count = 0

    async def generate(
        self, messages: list, tools_schema: list[dict]
    ) -> AssistantMessage:
        """
        模拟 LLM 调用 — 按调用次数返回不同的假回复。
        """
        self._call_count += 1

        if self._call_count == 1:
            # 第一轮：要求执行代码
            return AssistantMessage(
                content="我需要先执行这段代码来看看结果。",
                tool_calls=[
                    ToolCall(
                        id="mock_call_001",
                        function_name="execute_python",
                        arguments={"code": "print('Hello World')"},
                    )
                ],
            )
        else:
            # 第二轮：基于工具执行结果给出最终回复
            return AssistantMessage(
                content=(
                    "代码已成功执行。输出结果是 **Hello World**。\n\n"
                    "这说明 Python 环境运行正常，`print` 函数正确输出了字符串。"
                ),
                tool_calls=None,  # 没有更多工具调用，这是最终回复
            )


# ═══════════════════════════════════════════════════════════════
# 主程序 — 组装 Mock 组件，启动 Agent
# ═══════════════════════════════════════════════════════════════

async def main():
    # 初始化日志（verbose=True 让循环过程可见）
    logger = Logger(name="MockDemo")

    # 配置 Agent
    config = AgentConfig(
        system_prompt=(
            "你是一个代码助手。用户要求执行代码时，"
            "先调用 execute_python 工具，再根据执行结果回复用户。"
        ),
        max_iterations=10,
        verbose=True,  # 开启详细日志
    )

    # 组装 Mock 组件
    tool_registry = MockToolRegistry()
    llm = MockLLMProvider()

    # 创建 Agent
    agent = Agent(
        config=config,
        tool_registry=tool_registry,
        llm_provider=llm,
        logger=logger,
    )

    # 启动！
    user_prompt = "帮我执行一段 Python 代码：print('Hello World')"
    result = await agent.run(user_prompt)

    print("\n" + "=" * 60)
    print("[RESULT] 最终结果:")
    print(result)
    print("=" * 60)

    # 验证消息循环是否完整
    assert "Hello World" in result, "最终回复应该包含执行结果"
    assert llm._call_count == 2, f"LLM 应该被调用 2 次，实际: {llm._call_count}"
    print("\n[OK] 所有断言通过！Mock 骨架运转正常。")


if __name__ == "__main__":
    asyncio.run(main())
