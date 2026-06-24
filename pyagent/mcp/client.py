"""
MCP Client — stdio 传输实现。

启动子进程作为 MCP Server，通过 stdin/stdout 发送 JSON-RPC 消息。
支持工具发现（list_tools）和工具调用（call_tool）。

参考: https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .protocol import (
    JSONRPCRequest,
    JSONRPCResponse,
    ToolDef,
    ToolCall,
    ToolResult,
    Implementation,
)

logger = logging.getLogger("pyagent.mcp")


@dataclass
class MCPConfig:
    """MCP 服务器连接配置。"""

    name: str                           # 服务器名称（唯一标识）
    command: list[str]                  # 启动命令，如 ["python3", "mcp_server.py"]
    description: str = ""               # 服务器描述
    auto_reconnect: bool = True         # 断连自动重连
    startup_timeout: float = 10.0       # 连接超时（秒）


class MCPClient:
    """
    MCP 客户端 — 通过 stdio 连接单个 MCP 服务器。

    生命周期:
        1. connect()    → 启动子进程 + initialize 握手
        2. list_tools() → 获取工具列表
        3. call_tool()  → 调用工具
        4. disconnect() → 关闭连接

    Usage:
        client = MCPClient("cubesandbox", ["python3", "mcp_server.py"])
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("execute_python", {"code": "print(42)"})
    """

    def __init__(
        self,
        name: str,
        command: list[str],
        description: str = "",
        startup_timeout: float = 10.0,
    ):
        self.name = name
        self.command = command
        self.description = description
        self.startup_timeout = startup_timeout

        self._process: Optional[asyncio.subprocess.Process] = None
        self._next_id: int = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._connected = False
        self._server_info: Optional[Implementation] = None
        self._server_capabilities: dict = {}
        self._tools: list[ToolDef] = []

    # ── 生命周期 ────────────────────────────────────

    async def connect(self) -> None:
        """启动 MCP 服务器进程并完成 initialize 握手。"""
        logger.info(f"[MCP:{self.name}] 启动: {' '.join(self.command)}")

        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise MCPConnectError(f"命令未找到: {self.command[0]}") from e
        except Exception as e:
            raise MCPConnectError(f"启动失败: {e}") from e

        # 启动后台读取 stdout
        self._reader_task = asyncio.create_task(self._read_loop())
        self._connected = True

        # initialize 握手
        try:
            result = await self._send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "pyagent", "version": "0.10.0"},
                },
                timeout=self.startup_timeout,
            )
            self._server_info = Implementation(
                name=result.get("serverInfo", {}).get("name", "unknown"),
                version=result.get("serverInfo", {}).get("version", "0.0.0"),
            )
            self._server_capabilities = result.get("capabilities", {})
            logger.info(
                f"[MCP:{self.name}] 握手成功: "
                f"{self._server_info.name} v{self._server_info.version}"
            )

            # 发送 initialized 通知
            await self._send_notification("notifications/initialized", {})

        except Exception as e:
            await self.disconnect()
            raise MCPConnectError(f"initialize 失败: {e}") from e

    async def disconnect(self) -> None:
        """关闭 MCP 连接。"""
        self._connected = False
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        if self._process:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.kill()
                await self._process.wait()
            except Exception:
                pass
            self._process = None
        # 清理 pending
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(MCPDisconnectedError(f"[MCP:{self.name}] 已断开"))
        self._pending.clear()
        logger.info(f"[MCP:{self.name}] 已断开")

    # ── 工具操作 ────────────────────────────────────

    async def list_tools(self) -> list[ToolDef]:
        """获取工具列表（首次调用会缓存）。"""
        if self._tools:
            return self._tools
        result = await self._send_request("tools/list", {}, timeout=10.0)
        tools_raw = result.get("tools", [])
        self._tools = [
            ToolDef(
                name=t["name"],
                description=t.get("description", ""),
                inputSchema=t.get("inputSchema", {}),
            )
            for t in tools_raw
        ]
        logger.info(f"[MCP:{self.name}] 发现 {len(self._tools)} 个工具")
        return self._tools

    async def call_tool(self, tool_name: str, arguments: dict) -> ToolResult:
        """调用 MCP 工具。"""
        result = await self._send_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout=60.0,
        )
        return ToolResult(
            content=result.get("content", []),
            isError=result.get("isError", False),
        )

    # ── 内部通信 ────────────────────────────────────

    def _ensure_connected(self):
        if not self._connected:
            raise MCPDisconnectedError(f"[MCP:{self.name}] 未连接")

    async def _send_request(
        self, method: str, params: dict, timeout: float = 30.0
    ) -> dict:
        """发送 JSON-RPC 请求并等待响应。"""
        self._ensure_connected()
        req_id = self._next_id
        self._next_id += 1

        req = JSONRPCRequest(method=method, params=params, id=req_id)
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        try:
            payload = req.to_json() + "\n"
            self._process.stdin.write(payload.encode())
            await self._process.stdin.drain()
        except Exception as e:
            self._pending.pop(req_id, None)
            raise MCPDisconnectedError(f"写入失败: {e}") from e

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise MCPTimeoutError(f"[MCP:{self.name}] {method} 超时 ({timeout}s)")
        finally:
            self._pending.pop(req_id, None)

        if response.is_error:
            raise MCPCallError(
                f"[MCP:{self.name}] {method} 错误: {response.error}"
            )
        return response.result or {}

    async def _send_notification(self, method: str, params: dict) -> None:
        """发送 JSON-RPC 通知（无需等待响应）。"""
        self._ensure_connected()
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        try:
            payload = json.dumps(msg) + "\n"
            self._process.stdin.write(payload.encode())
            await self._process.stdin.drain()
        except Exception:
            pass  # 通知失败不影响

    async def _read_loop(self):
        """后台读取 MCP 服务器 stdout 响应。"""
        try:
            while self._connected and self._process:
                line = await self._process.stdout.readline()
                if not line:
                    break

                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                try:
                    resp = JSONRPCResponse.from_json(line_str)
                    fut = self._pending.get(resp.id)
                    if fut and not fut.done():
                        fut.set_result(resp)
                except json.JSONDecodeError:
                    logger.warning(f"[MCP:{self.name}] 无效响应: {line_str[:100]}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._connected:
                logger.warning(f"[MCP:{self.name}] 读取错误: {e}")
        finally:
            if self._connected:
                logger.warning(f"[MCP:{self.name}] 连接意外断开")
                self._connected = False

    # ── 属性 ────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def tools(self) -> list[ToolDef]:
        return list(self._tools)


# ── 异常 ────────────────────────────────────────────


class MCPError(Exception):
    """MCP 基础异常。"""
    pass


class MCPConnectError(MCPError):
    """连接错误。"""
    pass


class MCPDisconnectedError(MCPError):
    """断连错误。"""
    pass


class MCPTimeoutError(MCPError):
    """超时错误。"""
    pass


class MCPCallError(MCPError):
    """调用错误。"""
    pass
