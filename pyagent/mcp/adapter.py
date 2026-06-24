"""
MCP → PyAgent 适配器 — 将 MCP 工具转换为 PyAgent Tool 对象。

每种 MCP 工具自动生成一个 PyAgent Tool 子类，Agent 无需感知底层差异。
"""

from __future__ import annotations

import logging
from typing import Any

from .protocol import ToolDef, mcp_schema_to_openai
from .client import MCPClient
from ..tools.base import Tool

logger = logging.getLogger("pyagent.mcp.adapter")


def _make_mcp_tool_class(
    tool_def: ToolDef, client: MCPClient, server_name: str
) -> type[Tool]:
    """
    动态生成 PyAgent Tool 子类，包装单个 MCP 工具。

    返回的类可以直接加入 ToolRegistry。
    """
    name = tool_def.name
    description = tool_def.description
    schema = mcp_schema_to_openai(tool_def.inputSchema)

    # 必要字段提取
    required = tool_def.inputSchema.get("required", [])
    properties = tool_def.inputSchema.get("properties", {})

    class MCPTool(Tool):
        _mcp_client = client
        _mcp_server = server_name

        # Tool 基类需要的属性
        tool_name = name
        tool_description = description
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        async def execute(self, **kwargs) -> str:
            """执行 MCP 工具调用。"""
            try:
                result = await client.call_tool(name, kwargs)
                if result.isError:
                    return f"[MCP 错误] {result.content}"
                # 提取文本内容
                texts = []
                for item in result.content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            texts.append(item.get("text", ""))
                        elif item.get("type") == "resource":
                            texts.append(f"[资源: {item.get('resource', {})}]")
                    else:
                        texts.append(str(item))
                return "\n".join(texts) if texts else "(无输出)"
            except Exception as e:
                logger.error(f"[MCP:{server_name}] {name} 执行失败: {e}")
                return f"[MCP 错误] {e}"

    # 动态设置类名
    MCPTool.__name__ = f"MCP_{server_name}_{name}"
    MCPTool.__qualname__ = MCPTool.__name__

    return MCPTool


async def mcp_tools_to_pyagent_tools(
    client: MCPClient,
) -> list[Tool]:
    """
    将 MCP 客户端的所有工具转换为 PyAgent Tool 实例。

    Args:
        client: 已连接的 MCPClient

    Returns:
        PyAgent Tool 实例列表，可直接 register 到 ToolRegistry
    """
    tools = await client.list_tools()
    result = []
    for td in tools:
        tool_cls = _make_mcp_tool_class(td, client, client.name)
        result.append(tool_cls())
        logger.info(f"[MCP] {tool_cls.__name__}: {td.description[:80]}")
    return result
