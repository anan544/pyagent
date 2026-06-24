"""
MCP (Model Context Protocol) 集成层。

轻量级 MCP 客户端实现，支持：
    - stdio 传输（启动子进程，通过 stdin/stdout 通信）
    - SSE 传输（HTTP long-polling，后续）
    - 自动将 MCP 服务器工具转换为 PyAgent Tool

使用方式:
    client = MCPClient("cubesandbox", ["ssh", "ananan@vm", "python3", "mcp_server.py"])
    await client.connect()
    tools = client.list_tools()  # → [PyAgent Tool, ...]
    result = await client.call_tool("execute_python", {"code": "print(42)"})

协议参考: https://spec.modelcontextprotocol.io/
"""

from .client import MCPClient, MCPConfig
from .adapter import mcp_tools_to_pyagent_tools

__all__ = ["MCPClient", "MCPConfig", "mcp_tools_to_pyagent_tools"]
