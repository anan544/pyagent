#!/usr/bin/env python3
"""
Code2Prompt MCP Server — 代码库结构摘要工具。

将任意目录转换为 LLM 友好的 Markdown 摘要：
    - 文件树 + 依赖图
    - 每个文件的关键函数签名 + 行数
    - Token 估算

依赖: pip install code2prompt
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


def run_code2prompt(path: str, focus: str = "", exclude: str = "") -> dict:
    """运行 code2prompt，返回结构化结果。"""
    cmd = ["code2prompt", "--path", path, "--token", "--json"]
    if focus:
        cmd += ["--include", focus]
    if exclude:
        cmd += ["--exclude", exclude]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return {"error": result.stderr[:500]}

        # code2prompt --json 输出是 JSON 格式
        try:
            data = json.loads(result.stdout)
            return {
                "files": data.get("files", []),
                "total_tokens": data.get("total_tokens", 0),
                "total_files": data.get("total_files", 0),
                "summary": data.get("content", result.stdout)[:10000],
            }
        except json.JSONDecodeError:
            return {
                "files": [],
                "total_tokens": 0,
                "total_files": 0,
                "summary": result.stdout[:10000],
            }
    except FileNotFoundError:
        return {"error": "code2prompt 未安装。pip install code2prompt"}
    except subprocess.TimeoutExpired:
        return {"error": "code2prompt 超时（60s）"}


TOOLS = [
    {
        "name": "summarize_codebase",
        "description": (
            "分析代码库，生成 LLM 友好的结构化摘要。"
            "包含：文件依赖图、类图、每个文件的关键函数签名。"
            "适合快速理解陌生项目、代码审查前的全局预览。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "代码库根目录路径"},
                "focus": {"type": "string", "description": "关注的文件 glob，如 '*.py' 或 'src/*.rs'"},
                "exclude": {"type": "string", "description": "排除的文件 glob，如 '*.log,*.json'"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_project_structure",
        "description": (
            "快速列出项目文件结构（树形），不做深度分析。"
            "比 summarize_codebase 更快，适合先看一眼项目布局。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "项目根目录"},
                "max_depth": {"type": "integer", "description": "最大深度，默认 3", "default": 3},
            },
            "required": ["path"],
        },
    },
]


def main():
    print("[mcp-code2prompt] started", file=sys.stderr)

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
                "serverInfo": {"name": "code2prompt", "version": "0.2.0"},
            })
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send_response(req_id, {"tools": TOOLS})
        elif method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})

            if tool_name == "summarize_codebase":
                path = args.get("path", os.getcwd())
                focus = args.get("focus", "")
                exclude = args.get("exclude", "")
                result = run_code2prompt(path, focus, exclude)
                send_response(req_id, {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}],
                })

            elif tool_name == "list_project_structure":
                path = args.get("path", os.getcwd())
                max_depth = args.get("max_depth", 3)
                try:
                    # 简单实现：用 find + 截断
                    proc = subprocess.run(
                        ["find", path, "-maxdepth", str(max_depth),
                         "-not", "-path", "*/\\.*", "-not", "-path", "*/node_modules/*",
                         "-not", "-path", "*/__pycache__/*", "-not", "-path", "*/target/*"],
                        capture_output=True, text=True, timeout=10,
                    )
                    tree_text = proc.stdout[:5000]
                    send_response(req_id, {
                        "content": [{"type": "text", "text": tree_text}],
                    })
                except Exception as e:
                    send_error(req_id, -32000, str(e))

            else:
                send_error(req_id, -32601, f"未知工具: {tool_name}")
        else:
            send_error(req_id, -32601, f"未知方法: {method}")


if __name__ == "__main__":
    main()
