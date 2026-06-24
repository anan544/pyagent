"""
MCP 协议类型 — JSON-RPC 2.0 消息定义。

规范: https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# ── JSON-RPC 2.0 消息 ──────────────────────────────


@dataclass
class JSONRPCRequest:
    """JSON-RPC 请求。"""
    method: str
    params: dict[str, Any] | None = None
    id: int | str = 0
    jsonrpc: str = "2.0"

    def to_json(self) -> str:
        msg = {"jsonrpc": self.jsonrpc, "method": self.method, "id": self.id}
        if self.params is not None:
            msg["params"] = self.params
        return json.dumps(msg)


@dataclass
class JSONRPCResponse:
    """JSON-RPC 响应。"""
    id: int | str
    result: Any = None
    error: dict | None = None
    jsonrpc: str = "2.0"

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @classmethod
    def from_json(cls, text: str) -> JSONRPCResponse:
        data = json.loads(text)
        return cls(
            id=data.get("id", 0),
            result=data.get("result"),
            error=data.get("error"),
            jsonrpc=data.get("jsonrpc", "2.0"),
        )


@dataclass
class JSONRPCNotification:
    """JSON-RPC 通知（无 id，无需响应）。"""
    method: str
    params: dict[str, Any] | None = None
    jsonrpc: str = "2.0"


# ── MCP 协议消息 ──────────────────────────────────


@dataclass
class ToolDef:
    """MCP 工具定义（对应 list_tools 返回）。"""
    name: str
    description: str = ""
    inputSchema: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """工具调用请求。"""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """工具调用结果。"""
    content: list[dict[str, Any]] = field(default_factory=list)
    isError: bool = False


# ── MCP 初始化 ────────────────────────────────────


@dataclass
class ClientCapabilities:
    """客户端能力声明。"""
    roots: dict | None = None
    sampling: dict | None = None


@dataclass
class ServerCapabilities:
    """服务端能力声明。"""
    tools: dict | None = None
    resources: dict | None = None
    prompts: dict | None = None


@dataclass
class Implementation:
    name: str
    version: str


# ── 工具 schema → PyAgent schema 转换 ─────────────


def mcp_schema_to_openai(mcp_schema: dict) -> dict:
    """
    将 MCP tool inputSchema 转为 OpenAI function calling schema。

    MCP schema 是 JSON Schema 格式，OpenAI 基本兼容。
    主要处理：确保 type/required/properties 在正确位置。
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": mcp_schema.get("properties", {}),
    }
    if "required" in mcp_schema:
        schema["required"] = mcp_schema["required"]

    # 递归处理嵌套 properties
    for prop_name, prop_def in schema["properties"].items():
        if isinstance(prop_def, dict):
            if "type" not in prop_def:
                prop_def["type"] = "string"  # 默认
            if "description" not in prop_def:
                prop_def["description"] = ""

    return schema
