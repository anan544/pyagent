"""
长期记忆 Demo — 演示跨会话的消息持久化。

运行：
    python examples/memory_demo.py

流程：
    1. 第一次对话：Agent 读取并分析一个文件
    2. 对话结束时，所有消息自动持久化到 SQLite
    3. 第二次对话（同一 session_id）：Agent 能"记住"之前的对话
    4. 展示消息历史的加载和恢复

注意：本 Demo 使用 Mock LLM 演示记忆机制，不需要 API key。
      真实 LLM 场景下，Agent 的行为完全相同。
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyagent.core import Agent, AgentConfig
from pyagent.core.message import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
)
from pyagent.tools import ToolRegistry, ReadFileTool
from pyagent.memory import MemoryManager
from pyagent.utils.logger import Logger


# ═══════════════════════════════════════════════════════════════
# 模拟 LLM（演示记忆机制，无需 API key）
# ═══════════════════════════════════════════════════════════════

class ContextAwareMockLLM:
    """
    模拟 LLM — 能感知上下文中的历史消息。

    第一轮：要求读取文件 → 调用 read_file 工具
    第二轮：分析文件内容 → 返回审查意见
    有历史时：引用之前的对话内容
    """

    def __init__(self, name: str = "MockLLM"):
        self.name = name
        self.call_count = 0

    async def generate(self, messages, tools) -> AssistantMessage:
        self.call_count += 1

        # 检查消息历史中是否有"之前"的对话
        user_msgs = [m for m in messages if isinstance(m, UserMessage)]
        assistant_msgs = [m for m in messages if isinstance(m, AssistantMessage)]
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]

        # 如果历史中有文件内容，直接给出分析
        file_contents = [
            m.content for m in tool_msgs
            if hasattr(m, 'content') and m.content and "def " in str(m.content)
        ]

        if file_contents:
            # 有文件内容 → 基于内容给出分析
            content = file_contents[0]
            func_count = content.count("def ")
            return AssistantMessage(
                content=(
                    f"[{self.name}] 分析完成。\n"
                    f"该文件包含 {func_count} 个函数定义。\n"
                    f"代码结构清晰，无明显问题。"
                )
            )
        elif self.call_count == 1:
            # 首次调用 → 请求读取文件
            return AssistantMessage(
                content="让我先读取目标文件...",
                tool_calls=[
                    ToolCall(
                        id="call_read",
                        function_name="read_file",
                        arguments={"path": self._find_target_file(messages)},
                    )
                ],
            )
        else:
            return AssistantMessage(
                content=f"[{self.name}] 无法确定下一步操作。"
            )

    def _find_target_file(self, messages) -> str:
        """从用户消息中提取文件路径。"""
        for m in messages:
            if isinstance(m, UserMessage):
                for word in m.content.split():
                    if word.endswith(".py") and os.path.exists(word):
                        return word
        # fallback: 返回 demo 自身
        return __file__


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

async def main():
    log = Logger(name="MemoryDemo")

    # 使用临时目录存放数据库文件
    db_path = os.path.join(tempfile.gettempdir(), "pyagent_memory_demo.db")
    log.info(f"数据库文件: {db_path}")

    # 清理上次的演示数据
    if os.path.exists(db_path):
        os.remove(db_path)

    # 创建 MemoryManager
    memory = MemoryManager(db_path)

    # 创建会话
    SESSION_ID = "demo-session-001"
    sid = await memory.create_session(SESSION_ID, metadata={"project": "memory-demo"})
    log.info(f"会话已创建: {sid}")

    # 组装 Agent
    registry = ToolRegistry()
    registry.register(ReadFileTool())

    config = AgentConfig(
        system_prompt="你是一个代码审查助手。请阅读并分析代码文件。",
        max_iterations=5,
        verbose=True,
    )

    # ═══════════════════════════════════════════════════════════
    # 第 1 次对话
    # ═══════════════════════════════════════════════════════════

    log.info("=" * 70)
    log.info("第 1 次对话：审查 agent.py")
    log.info("=" * 70)

    llm1 = ContextAwareMockLLM(name="Round1-LLM")
    agent1 = Agent(
        config=config,
        tool_registry=registry,
        llm_provider=llm1,
        memory=memory,
        logger=log,
    )

    # 找一个真实文件作为审查目标（demo 自身也行）
    target = os.path.join(
        os.path.dirname(__file__), "..", "pyagent", "core", "agent.py"
    )
    target = os.path.abspath(target)

    result1 = await agent1.run(
        f"请审查这个文件: {target}",
        session_id=SESSION_ID,
    )
    print(f"\n[第 1 次结果]\n{result1}")

    msg_count_1 = await memory.message_count(SESSION_ID)
    log.info(f"第 1 次对话后消息数: {msg_count_1}")

    # ═══════════════════════════════════════════════════════════
    # 第 2 次对话（同一 session_id）
    # ═══════════════════════════════════════════════════════════

    log.info("")
    log.info("=" * 70)
    log.info("第 2 次对话：基于历史记忆进一步提问")
    log.info("=" * 70)

    llm2 = ContextAwareMockLLM(name="Round2-LLM")
    agent2 = Agent(
        config=config,
        tool_registry=registry,
        llm_provider=llm2,
        memory=memory,
        logger=log,
    )

    result2 = await agent2.run(
        "之前审查的文件中，最大的函数是哪个？有什么改进建议？",
        session_id=SESSION_ID,
    )
    print(f"\n[第 2 次结果]\n{result2}")

    msg_count_2 = await memory.message_count(SESSION_ID)
    log.info(f"第 2 次对话后消息总数: {msg_count_2}")

    # ═══════════════════════════════════════════════════════════
    # 展示持久化状态
    # ═══════════════════════════════════════════════════════════

    log.info("")
    log.info("=" * 70)
    log.info("持久化状态总览")
    log.info("=" * 70)

    session_info = await memory.get_session(SESSION_ID)
    log.info(f"会话 ID:      {session_info['session_id']}")
    log.info(f"创建时间:     {session_info['created_at']}")
    log.info(f"最后活跃:     {session_info['updated_at']}")
    log.info(f"消息总数:     {msg_count_2}")

    # 列出所有消息的角色分布
    all_msgs = await memory.load_messages(SESSION_ID, limit=100)
    role_counts = {}
    for m in all_msgs:
        role_counts[m.role] = role_counts.get(m.role, 0) + 1
    log.info(f"角色分布:     {role_counts}")

    # 列出所有会话
    log.info("")
    log.info("所有会话:")
    sessions = await memory.list_sessions()
    for s in sessions:
        log.info(f"  - {s['session_id']} (最后活跃: {s['updated_at']})")

    await memory.close()

    # 清理
    if os.path.exists(db_path):
        os.remove(db_path)
        log.info(f"\n清理演示数据库: {db_path}")

    log.info("Demo 完成!")


if __name__ == "__main__":
    asyncio.run(main())
