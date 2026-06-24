"""
CubeSandbox 代码执行工具 — 代码在远端微 VM 中执行。

与 CodeExecutorTool 接口兼容，但底层使用 CubeSandbox PVM 微虚拟机
而非本地 subprocess。每个代码执行都是一个独立的微 VM。

安全特性：
    - 代码在 KVM 级隔离的微 VM 中执行
    - 每次执行独立沙箱（启动 → 执行 → 销毁）
    - 超时自动销毁沙箱
    - SSH 连接复用
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from typing import Optional

from .base import Tool

logger = logging.getLogger("pyagent.tools.sandbox")

# 默认 VM 配置
DEFAULT_VM_HOST = "192.168.100.130"
DEFAULT_VM_USER = "ananan"
DEFAULT_VM_KEY = None  # 设为 SSH key 路径可加速
DEFAULT_TIMEOUT = 30


class SandboxTool(Tool):
    """
    在 CubeSandbox 微 VM 中执行 Python 代码。

    每个 execute() 调用会：
        1. 在远端 VM 创建沙箱微 VM
        2. 通过 debug console 执行代码
        3. 返回 stdout
        4. 自动销毁沙箱

    配置：
        vm_host: CubeSandbox 宿主机 IP
        vm_user: SSH 用户名
        vm_key:   SSH 私钥路径（None 则用密码，但需配 sshpass）
    """

    name = "execute_python"
    risk_level = "high"
    description = (
        "在 CubeSandbox 隔离沙箱微 VM 中执行 Python 代码，返回标准输出。"
        "不支持交互式 input() 调用，不支持 GUI 操作。"
        "适合用于：运行单元测试、验证代码逻辑、执行数据计算。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码",
            },
            "timeout": {
                "type": "integer",
                "description": "执行超时秒数，默认 30",
                "default": 30,
            },
        },
        "required": ["code"],
    }

    def __init__(
        self,
        vm_host: str = DEFAULT_VM_HOST,
        vm_user: str = DEFAULT_VM_USER,
        vm_key: Optional[str] = DEFAULT_VM_KEY,
        default_timeout: int = DEFAULT_TIMEOUT,
    ):
        self.vm_host = vm_host
        self.vm_user = vm_user
        self.vm_key = vm_key
        self.default_timeout = default_timeout

    async def execute(self, code: str, timeout: int | None = None) -> str:
        """
        在沙箱中执行 Python 代码。

        Args:
            code: 要执行的 Python 代码
            timeout: 超时秒数，默认 30

        Returns:
            执行结果（stdout + stderr）
        """
        timeout = timeout or self.default_timeout

        payload = json.dumps({"code": code, "timeout": timeout})
        executor_cmd = "python3 /usr/local/bin/sandbox_exec.py"

        try:
            output = await self._ssh_exec(payload, timeout)
            result = json.loads(output)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            sid = result.get("sandbox_id", "unknown")

            parts = []
            if stdout:
                parts.append(stdout)
            if stderr:
                parts.append(f"[stderr]\n{stderr}")
            if not stdout and not stderr:
                parts.append("(无输出)")

            return "\n".join(parts)
        except subprocess.TimeoutExpired:
            return (
                f"[超时] 代码执行超过 {timeout} 秒，已被强制终止。\n"
                f"请检查是否存在死循环或耗时过长的操作。"
            )
        except Exception as e:
            logger.error(f"沙箱执行失败: {e}")
            return f"[错误] 沙箱执行失败: {e}"

    async def _ssh_exec(self, stdin_data: str, timeout: int) -> str:
        """通过 SSH 在 VM 上执行 sandbox_exec.py。"""
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", f"ServerAliveInterval={timeout + 10}",
        ]
        if self.vm_key:
            ssh_cmd += ["-i", self.vm_key]

        ssh_cmd.append(f"{self.vm_user}@{self.vm_host}")

        cmd = ssh_cmd + [executor_cmd := "python3 /usr/local/bin/sandbox_exec.py"]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(stdin_data.encode()),
                timeout=timeout + 30,  # extra time for VM boot + cleanup
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise subprocess.TimeoutExpired(cmd=" ".join(cmd), timeout=timeout)

        stdout_str = stdout.decode("utf-8", errors="replace").strip()
        stderr_str = stderr.decode("utf-8", errors="replace").strip()

        if stderr_str:
            logger.warning(f"SSH stderr: {stderr_str[:500]}")

        return stdout_str
