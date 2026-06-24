"""
ToolRegistry — 工具注册表。

负责管理所有可用工具的注册、查找和执行。
Agent 通过 ToolRegistry 与工具系统交互。
"""

from typing import Any
from .base import Tool
from ..core.message import ToolMessage


class ToolNotFoundError(Exception):
    """LLM 请求的工具未在注册表中找到。"""
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"工具 '{tool_name}' 未注册。可用工具: [请调用 list_names()]")


class ToolRegistry:
    """
    工具注册表 — 管理所有可用工具。

    使用方式：
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())

        schemas = registry.get_all_schemas()  # 传给 LLM
        tool_msg = await registry.execute("read_file", "call_1", {"path": "..."})
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        注册一个工具。

        Args:
            tool: Tool 实例

        Raises:
            ValueError: 工具名称为空，或同名工具已注册
        """
        if not tool.name:
            raise ValueError("工具名称不能为空")
        if tool.name in self._tools:
            raise ValueError(f"工具 '{tool.name}' 已经注册过了")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """移除一个已注册的工具。"""
        self._tools.pop(name, None)

    def list_names(self) -> list[str]:
        """返回所有已注册工具的名称列表。"""
        return list(self._tools.keys())

    def get_all(self) -> list[Tool]:
        """返回所有已注册工具的实例列表。"""
        return list(self._tools.values())

    def get_all_schemas(self) -> list[dict]:
        """
        返回所有工具的 OpenAI function calling schema。
        用于传给 LLM 的 tools 参数。
        """
        return [tool.get_schema() for tool in self._tools.values()]

    async def execute(
        self, name: str, call_id: str, arguments: dict[str, Any]
    ) -> ToolMessage:
        """
        根据工具名称和参数执行工具，返回 ToolMessage。

        Args:
            name: 工具名称
            call_id: LLM 返回的 tool_call id（用于关联回复）
            arguments: LLM 返回的工具参数

        Returns:
            包装了执行结果的 ToolMessage

        Raises:
            ToolNotFoundError: 工具未注册
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)

        result = await tool.execute(**arguments)
        return ToolMessage(
            content=str(result),
            tool_call_id=call_id,
            name=name,
        )
