#!/usr/bin/env python3
"""CubeSandbox 代码执行器 — 部署在 VM 上，通过 debug console 执行 Python 代码。

用法:
    python3 sandbox_exec.py '{"code":"print(1+1)","timeout":30}'

返回 JSON:
    {"stdout": "...", "stderr": "...", "sandbox_id": "..."}
"""
import json
import sys
import time
import subprocess
import urllib.request
import urllib.error

API = "http://localhost:3000"
TEMPLATE = "tpl-a1ac6a013c6747a5bf64812f"
CUBE_RUNTIME = "/usr/local/services/cubetoolbox/cube-shim/bin/cube-runtime"


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
        req = urllib.request.Request(
            f"{API}/sandboxes/{sandbox_id}", method="DELETE"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def sandbox_exec(sandbox_id: str, code: str, timeout_sec: int = 30) -> str:
    """通过 debug console 在沙箱中执行 Python 代码，返回 stdout。"""
    import base64 as _b64
    encoded_code = _b64.b64encode(code.encode()).decode()

    # 写入 /run（tmpfs, 可写）, 执行, 读结果
    remote_cmd = (
        f"echo {encoded_code} | base64 -d > /run/code.py; "
        f"python3 -u /run/code.py > /run/out.txt 2>&1; "
        f"cat /run/out.txt"
    )

    expect_script = f'''
set timeout {timeout_sec + 5}
log_user 1
spawn {CUBE_RUNTIME} login {sandbox_id}
expect "bash-" {{ send "{remote_cmd}\\r" }}
expect "bash-" {{}}
'''

    result = subprocess.run(
        ["sudo", "expect", "-c", expect_script],
        capture_output=True,
        text=True,
        timeout=timeout_sec + 10,
    )

    raw = result.stdout
    import re as _re
    # 原始格式: ...cat /run/out.txt\r\n\x1b[?2004l\r<输出>\r\n\x1b[?2004hbash-
    # 先找 cat /run/out.txt 之后的位置
    idx = raw.find("cat /run/out.txt")
    if idx < 0:
        return raw.strip()
    after_cat = raw[idx:]
    # 匹配 ANSI 序列后的输出: \x1b[?2004l\n<内容>\n\x1b[?2004h
    match = _re.search(r"\x1b\[\?2004l\r?\n(.*?)\r?\n\x1b\[\?2004h", after_cat, _re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


def main():
    raw = sys.stdin.read().strip() if not sys.stdin.isatty() else sys.argv[1]
    req = json.loads(raw)
    code = req["code"]
    timeout = req.get("timeout", 30)

    sid = sandbox_create(timeout + 30)
    time.sleep(4)  # 等 VM 启动

    stdout = ""
    stderr = ""
    try:
        stdout = sandbox_exec(sid, code, timeout)
    except subprocess.TimeoutExpired:
        stderr = f"代码执行超时 ({timeout}s)"
    except Exception as e:
        stderr = str(e)
    finally:
        sandbox_delete(sid)

    print(json.dumps({"stdout": stdout, "stderr": stderr, "sandbox_id": sid}))


if __name__ == "__main__":
    main()
