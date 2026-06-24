#!/usr/bin/env python3
"""
Agent-Reach MCP Server — AI Agent 的互联网之眼。

Agent-Reach 提供 17+ 平台的免费信息获取：
    - web: 任意网页阅读（Jina Reader）
    - youtube: 字幕提取 + 搜索
    - github: 公开仓库读写搜索
    - twitter/x, reddit, bilibili, 小红书, linkedin...
    - exa_search: 全网语义搜索

安装: pip install https://github.com/Panniantong/agent-reach/archive/main.zip
仓库: https://github.com/Panniantong/agent-reach
协议: JSON-RPC 2.0 over stdin/stdout
"""

import json
import os
import shutil
import subprocess
import sys
from typing import Any

AGENT_REACH_BIN = shutil.which("agent-reach") or "agent-reach"


def send_response(req_id: Any, result: Any) -> None:
    msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def send_error(req_id: Any, code: int, message: str) -> None:
    msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def run_agent_reach(channel: str, action: str, query: str = "", extra_args: list | None = None) -> dict:
    """
    调用 agent-reach 执行平台操作。

    Args:
        channel: 平台名 (web, youtube, github, twitter, bilibili, exa_search...)
        action: 操作 (search, read, doctor)
        query: 查询内容
    """
    cmd = [AGENT_REACH_BIN, channel, action]
    if query:
        cmd.append(query)
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        return {
            "channel": channel,
            "action": action,
            "query": query,
            "output": stdout[:8000] if stdout else "",
            "error": stderr[:500] if result.returncode != 0 else "",
            "exit_code": result.returncode,
            "truncated": len(stdout) > 8000,
        }
    except FileNotFoundError:
        return {"error": f"agent-reach 未安装。pip install https://github.com/Panniantong/agent-reach/archive/main.zip"}
    except subprocess.TimeoutExpired:
        return {"error": f"agent-reach {channel} {action} 超时（60s）"}


TOOLS = [
    {
        "name": "web_search",
        "description": (
            "在互联网上搜索信息。支持全网搜索(exa)、网页阅读(web)、"
            "推特/Reddit/YouTube/GitHub/B站 等 17+ 平台。"
            "零 API 费用，无需配置 Key。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "enum": ["web", "exa_search", "twitter", "reddit", "youtube", "github",
                             "bilibili", "xiaohongshu", "linkedin", "rss", "v2ex"],
                    "description": "搜索平台",
                },
                "query": {"type": "string", "description": "搜索关键词或 URL"},
            },
            "required": ["channel", "query"],
        },
    },
    {
        "name": "read_webpage",
        "description": (
            "读取任意网页内容（通过 Jina Reader）。传入 URL，返回清洗后的文本。"
            "适合：阅读文档、博客、新闻、GitHub Issue 等。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要读取的网页 URL"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "check_reach_status",
        "description": (
            "检查 Agent-Reach 各平台连接状态。"
            "自动诊断每个 channel 的可用性并给出修复建议。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def main():
    print("[mcp-agent-reach] started", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = req.get("id", 0)
        method = req.get("method", "")
        params = req.get("params", {})

        if method == "initialize":
            send_response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "agent-reach", "version": "0.1.0"},
            })
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send_response(req_id, {"tools": TOOLS})
        elif method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})

            if tool_name == "web_search":
                result = run_agent_reach(
                    channel=args.get("channel", "web"),
                    action="search",
                    query=args.get("query", ""),
                )
                send_response(req_id, {
                    "content": [{"type": "text", "text": result.get("output", json.dumps(result))}],
                })

            elif tool_name == "read_webpage":
                result = run_agent_reach(
                    channel="web",
                    action="read",
                    query=args.get("url", ""),
                )
                send_response(req_id, {
                    "content": [{"type": "text", "text": result.get("output", json.dumps(result))}],
                })

            elif tool_name == "check_reach_status":
                result = run_agent_reach(channel="", action="doctor")
                send_response(req_id, {
                    "content": [{"type": "text", "text": result.get("output", json.dumps(result))}],
                })

            else:
                send_error(req_id, -32601, f"未知工具: {tool_name}")
        else:
            send_error(req_id, -32601, f"未知方法: {method}")


if __name__ == "__main__":
    main()
