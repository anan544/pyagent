#!/usr/bin/env python3
"""
CubeSandbox MCP Server — 通过 MCP stdio 协议暴露沙箱代码执行能力。

部署: scp 到 VM，配置 PyAgent MCP 客户端连它。

协议: JSON-RPC 2.0 over stdin/stdout
工具:
    - execute_python: 在隔离微 VM 中执行 Python 代码
    - sandbox_health: 检查沙箱服务状态
"""

import json
import os
import sys
import time
import subprocess
import urllib.request
import urllib.error
from typing import Any

API = "http://localhost:3000"
TEMPLATE = os.environ.get("CUBE_SANDBOX_TEMPLATE", "tpl-a1ac6a013c6747a5bf64812f")
CUBE_RUNTIME = "/usr/local/services/cubetoolbox/cube-shim/bin/cube-runtime"


# ── 沙箱操作（复用 sandbox_exec.py 逻辑）───────────


def sandbox_create(timeout_sec: int = 90) -> str:
    """创建沙箱，返回 sandboxID。"""
    payload = json.dumps({"templateID": TEMPLATE, "timeout": timeout_sec}).encode()
    req = urllib.request.Request(
        f"{API}/sandboxes",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)["sandboxID"]


def sandbox_delete(sandbox_id: str) -> None:
    """删除沙箱（fire-and-forget）。"""
    try:
        req = urllib.request.Request(f"{API}/sandboxes/{sandbox_id}", method="DELETE")
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def sandbox_exec(sandbox_id: str, code: str, timeout_sec: int = 30) -> str:
    """通过 debug console 在沙箱中执行 Python 代码，返回 stdout。"""
    import base64

    encoded_code = base64.b64encode(code.encode()).decode()
    remote_cmd = (
        f"echo {encoded_code} | base64 -d > /run/code.py; "
        f"python3 -u /run/code.py > /run/out.txt 2>&1; "
        f"cat /run/out.txt"
    )

    expect_script = f"""
set timeout {timeout_sec + 5}
log_user 0
spawn {CUBE_RUNTIME} login {sandbox_id}
expect "bash-" {{ send "{remote_cmd}\\r" }}
expect "bash-" {{}}
"""
    result = subprocess.run(
        ["sudo", "expect", "-c", expect_script],
        capture_output=True,
        text=True,
        timeout=timeout_sec + 10,
    )

    raw = result.stdout
    import re as _re

    idx = raw.find("cat /run/out.txt")
    if idx < 0:
        return raw.strip()
    after_cat = raw[idx:]
    match = _re.search(r"\x1b\[\?2004l\r?\n(.*?)\r?\n\x1b\[\?2004h", after_cat, _re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


# ── MCP 协议处理 ────────────────────────────────────


def send_response(id: Any, result: Any) -> None:
    """发送 JSON-RPC 响应到 stdout。"""
    msg = json.dumps({"jsonrpc": "2.0", "id": id, "result": result})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def send_error(id: Any, code: int, message: str) -> None:
    """发送 JSON-RPC 错误到 stdout。"""
    msg = json.dumps({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ── 工具定义 ────────────────────────────────────────

TOOLS = [
    {
        "name": "execute_python",
        "description": (
            "在 CubeSandbox 隔离微 VM 中执行 Python 代码。"
            "每次调用创建独立的沙箱实例，执行完成后自动销毁。"
            "不支持交互式 input()，不支持 GUI。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码",
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时秒数，默认 30",
                    "default": 30,
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "sandbox_health",
        "description": "检查 CubeSandbox 沙箱服务健康状态，返还可用的沙箱和模板信息。",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── 主循环 ──────────────────────────────────────────


def main():
    """MCP stdio 主循环。"""
    # 日志到 stderr（不影响 stdout 协议）
    print("[mcp-cubesandbox] started", file=sys.stderr)

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

        # ── initialize ──
        if method == "initialize":
            send_response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "cubesandbox", "version": "0.1.0"},
            })

        # ── notifications/initialized ──
        elif method == "notifications/initialized":
            pass  # 无需响应

        # ── tools/list ──
        elif method == "tools/list":
            send_response(req_id, {"tools": TOOLS})

        # ── tools/call ──
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            if tool_name == "execute_python":
                code = arguments.get("code", "")
                timeout = arguments.get("timeout", 30)
                try:
                    sid = sandbox_create(timeout + 30)
                    time.sleep(4)  # 等 VM 启动
                    stdout = sandbox_exec(sid, code, timeout)
                    sandbox_delete(sid)
                    send_response(req_id, {
                        "content": [{"type": "text", "text": stdout or "(无输出)"}],
                    })
                except Exception as e:
                    send_error(req_id, -32000, f"沙箱执行失败: {e}")

            elif tool_name == "sandbox_health":
                try:
                    resp = urllib.request.urlopen(f"{API}/health", timeout=5)
                    data = json.load(resp)
                    send_response(req_id, {
                        "content": [{"type": "text", "text": json.dumps(data, indent=2)}],
                    })
                except Exception as e:
                    send_error(req_id, -32000, f"健康检查失败: {e}")

            else:
                send_error(req_id, -32601, f"未知工具: {tool_name}")

        # ── 未知方法 ──
        else:
            send_error(req_id, -32601, f"未知方法: {method}")


if __name__ == "__main__":
    main()
