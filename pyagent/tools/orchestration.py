"""
多 Agent 协作 — 主从架构。

Main Agent (项目经理): 拆解任务、派活、汇总结果。无文件操作权限。
Sub-agent  (执行专员): 干具体活 — 读写文件、搜代码、执行 Python。

使用: 将 AgentTool 注册到主 Agent，主 Agent 设置 system_prompt 引导
      其为"项目经理"角色，只调 spawn_subagent 和 search_content。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .base import Tool
from ..core.agent import Agent
from ..core.config import AgentConfig

logger = logging.getLogger("pyagent.orchestration")


class AgentTool(Tool):
    """
    生成子 Agent 执行具体子任务。

    主 Agent 调用此工具 → 创建独立的子 Agent → 执行 → 返回结果。
    子 Agent 拥有完整的工具访问权限（读写文件、执行代码、搜索）。

    设计:
        - 子 Agent 使用独立的 Agent 实例（隔离状态）
        - 复用主 Agent 的 LLM Provider（节省连接）
        - 子 Agent 走 ReAct 模式（快，适合子任务）
    """

    name = "spawn_subagent"
    description = (
        "生成一个专门的子 Agent 来处理特定、自包含的任务。\n"
        "\n"
        "当任务需要对特定领域进行深入分析（如复杂数学、长文写作或独立编码任务）"
        "且会挤占主对话时使用此工具。\n"
        "子 Agent 拥有完整的工具权限（读文件、写文件、执行代码、搜索代码）。\n"
        "子 Agent 在单独的上下文中工作并返回简洁的摘要和结果。\n"
        "不要用于可以直接回答的简单查询。\n"
        "适合：'修改某文件'、'搜索并替换'、'运行测试' 等有明确输入输出的具体操作。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "子任务的明确目标。必须包含具体文件路径和期望输出。"
                               "示例: '在 pyagent/core/agent.py 中添加 _validate_input() 方法，"
                               "接收 str 参数，返回 bool'",
            },
            "context": {
                "type": "string",
                "description": "子 Agent 需要了解的相关上下文或背景信息。",
            },
        },
        "required": ["goal"],
    }
    risk_level = "high"

    def __init__(
        self,
        llm_provider,
        tool_registry,
        memory=None,
        max_iterations: int = 20,
        governance=None,
    ):
        super().__init__()
        self._llm = llm_provider
        self._tool_registry = tool_registry
        self._memory = memory
        self._max_iterations = max_iterations
        self._governance = governance

    async def execute(self, goal: str, context: str = "", **kwargs) -> str:
        """创建子 Agent 并执行子任务。"""
        full_prompt = goal
        if context:
            full_prompt = f"{goal}\n\n额外上下文:\n{context}"

        logger.info(f"[SubAgent] 启动: {goal[:100]}...")

        # 创建独立的 Agent 实例
        sub_agent = Agent(
            config=AgentConfig(
                system_prompt=(
                    "You are an expert software engineer. "
                    "Complete the assigned task precisely and efficiently. "
                    "Use tools to read, write, search, and execute code as needed. "
                    "When you finish the task, provide a clear summary of what you did."
                ),
                max_iterations=self._max_iterations,
                verbose=False,
            ),
            llm_provider=self._llm,
            tool_registry=self._tool_registry,
            memory=self._memory,
            governance=self._governance,
        )

        try:
            result = await sub_agent.run(full_prompt, session_id=None)
            logger.info(f"[SubAgent] 完成: {result[:100]}...")
            return result
        except Exception as e:
            logger.error(f"[SubAgent] 失败: {e}")
            return f"[子 Agent 执行失败] {e}"


class MultiAgentSession:
    """
    多 Agent 会话管理器。

    创建专注于不同目标的 Agent 实例。主 Agent 调用 spawn_subagent
    时自动创建独立子 Agent，任务完成后子 Agent 销毁。

    Usage:
        session = MultiAgentSession(llm, tools, memory)
        main_agent = session.create_main_agent()
        result = await main_agent.run("重构 pyagent/core/agent.py")
    """

    def __init__(self, llm_provider, tool_registry, memory=None, governance=None):
        self._llm = llm_provider
        self._tool_registry = tool_registry
        self._memory = memory
        self._governance = governance

    def create_main_agent(self, system_prompt: str = "", max_iterations: int = 20) -> Agent:
        """创建主 Agent（项目经理）。仅注册编排工具。"""
        from ..tools.registry import ToolRegistry

        # 主 Agent 只装编排工具
        main_tools = ToolRegistry()
        main_tools.register(
            AgentTool(
                llm_provider=self._llm,
                tool_registry=self._tool_registry,
                memory=self._memory,
                governance=self._governance,
            )
        )
        # 可以加 search_content 供主 Agent 了解代码库
        from ..tools.search import SearchTool
        main_tools.register(SearchTool())

        return Agent(
            config=AgentConfig(
                system_prompt=system_prompt or self._default_main_prompt(),
                max_iterations=max_iterations,
                verbose=True,
            ),
            llm_provider=self._llm,
            tool_registry=main_tools,
            memory=self._memory,
            governance=self._governance,
        )

    @staticmethod
    def _default_main_prompt() -> str:
        return (
            "You are a senior software engineering manager.\n\n"
            "**Your Role**\n"
            "- Break down user requirements into specific, actionable subtasks.\n"
            "- Delegate each subtask to a sub-agent using `spawn_subagent`.\n"
            "- You CANNOT directly read, write, or execute code.\n"
            "- You CAN use `search_content` to understand the codebase before delegating.\n\n"
            "**How to delegate**\n"
            "- Each `spawn_subagent` call should have a clear, specific goal.\n"
            "- Include exact file paths and expected changes in the goal.\n"
            "- Wait for the sub-agent to finish before spawning the next one.\n\n"
            "**When all subtasks are done**\n"
            "- Summarize what each sub-agent did.\n"
            "- Report any issues or failures.\n"
        )
