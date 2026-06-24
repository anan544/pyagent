#!/usr/bin/env python3
"""
ck Semantic Search MCP Server — 语义 + 混合代码搜索。

ck 是 BeaconBay/ck (Rust) 的 Python 包装器，支持：
    - 语义搜索：按含义找代码 (--sem)
    - 混合搜索：语义 + BM25 关键词 (--hybrid) — 推荐
    - JSONL 输出：天然适配 Agent 消费

安装: cargo install ck-search
仓库: https://github.com/BeaconBay/ck
协议: JSON-RPC 2.0 over stdin/stdout
"""

import json
import os
import subprocess
import sys
from typing import Any


def send_response(req_id: Any, result: Any) -> None:
    msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def send_error(req_id: Any, code: int, message: str) -> None:
    msg = json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def run_ck_search(
    query: str,
    path: str = ".",
    mode: str = "hybrid",
    top_k: int = 10,
    full_section: bool = False,
) -> dict:
    """
    运行 ck 搜索。

    Args:
        query: 搜索查询（自然语言或关键词）
        path: 搜索路径
        mode: "sem" (纯语义), "hybrid" (语义+BM25, 推荐), "grep" (纯正则)
        top_k: 返回结果数
        full_section: 是否返回完整函数体（默认仅签名 + 上下文）
    """
    cmd = ["ck", "--jsonl"]

    if mode == "hybrid":
        cmd.append("--hybrid")
    elif mode == "sem":
        cmd.append("--sem")

    if full_section:
        cmd.append("--full-section")

    # 限制结果数（ck 默认无限制）
    cmd.extend(["--max-results", str(top_k)])
    cmd.extend(["--path", path])
    cmd.append(query)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 and result.returncode != 1:
            return {"error": result.stderr[:500]}

        # 解析 JSONL
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        results = []
        for line in lines:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return {
            "query": query,
            "mode": mode,
            "total": len(results),
            "results": results,
            "truncated": len(results) >= top_k,
        }
    except FileNotFoundError:
        return {"error": "ck 未安装。cargo install ck-search"}
    except subprocess.TimeoutExpired:
        return {"error": "搜索超时（30s）"}


TOOLS = [
    {
        "name": "semantic_search",
        "description": (
            "语义搜索：按含义（非关键词）查找代码。\n"
            "例如：'怎么创建 eBPF map' 会找到相关的 C 代码片段，即使不含 '创建' 一词。\n"
            "支持混合模式（语义 + BM25 关键词），推荐用于代码审查和 bug 查找。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "自然语言搜索查询"},
                "path": {"type": "string", "description": "搜索路径，默认当前目录", "default": "."},
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "sem", "grep"],
                    "description": "搜索模式：hybrid(推荐)=语义+关键词, sem=纯语义, grep=纯正则",
                    "default": "hybrid",
                },
                "top_k": {"type": "integer", "description": "返回结果数上限", "default": 10},
                "full_section": {"type": "boolean", "description": "是否返回完整函数体", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_similar_code",
        "description": (
            "给定一段代码片段，在代码库中找到语义相似的代码。\n"
            "适合：找重复逻辑、类似实现、可复用模块。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code_snippet": {"type": "string", "description": "参考代码片段"},
                "path": {"type": "string", "description": "搜索路径", "default": "."},
                "top_k": {"type": "integer", "description": "返回结果数", "default": 5},
            },
            "required": ["code_snippet"],
        },
    },
]


def main():
    print("[mcp-ck] started", file=sys.stderr)

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
                "serverInfo": {"name": "ck-semantic-search", "version": "0.1.0"},
            })
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send_response(req_id, {"tools": TOOLS})
        elif method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})

            if tool_name == "semantic_search":
                result = run_ck_search(
                    query=args.get("query", ""),
                    path=args.get("path", os.getcwd()),
                    mode=args.get("mode", "hybrid"),
                    top_k=args.get("top_k", 10),
                    full_section=args.get("full_section", False),
                )
                send_response(req_id, {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}],
                })

            elif tool_name == "find_similar_code":
                # ck 可以直接搜代码片段
                snippet = args.get("code_snippet", "")
                result = run_ck_search(
                    query=snippet,
                    path=args.get("path", os.getcwd()),
                    mode="sem",
                    top_k=args.get("top_k", 5),
                    full_section=True,
                )
                send_response(req_id, {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}],
                })

            else:
                send_error(req_id, -32601, f"未知工具: {tool_name}")
        else:
            send_error(req_id, -32601, f"未知方法: {method}")


if __name__ == "__main__":
    main()
