"""
代码执行工具 — 在 subprocess 中执行 Python 代码。

安全设计：
    - subprocess 隔离执行
    - 超时机制（默认 30 秒）
    - 捕获 stdout / stderr
    - 不允许交互式输入
"""

import asyncio
import os
import tempfile
from .base import Tool


class CodeExecutorTool(Tool):
    """
    在独立的子进程中执行 Python 代码。

    安全考虑：
        - 代码在临时文件中执行，不影响主进程
        - 30 秒超时，防止死循环
        - 同时捕获 stdout 和 stderr
    """

    name = "execute_python"
    risk_level = "high"
    description = (
        "在隔离的临时沙盒环境中执行 Python 脚本。\n"
        "\n"
        "代码将在沙盒中运行，您将收到 stdout、stderr 和返回码。\n"
        "如果代码崩溃或有语法错误，请修复后重新运行。\n"
        "⚠️ subprocess/os.system/os.popen 被安全策略禁止，"
        "请改用 execute_command 工具来启动进程或执行 Shell 命令。\n"
        "⚠️ 不支持交互式 input() 调用，不支持 GUI 操作。\n"
        "适合：运行单元测试、验证代码逻辑、数据计算、数据处理。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的原始 Python 代码（不要用 markdown 代码块包裹）。",
            },
            "timeout": {
                "type": "integer",
                "description": "执行超时秒数，默认 30",
                "default": 30,
            },
            "explanation": {
                "type": "string",
                "description": "一句话说明这段代码做什么以及为什么需要运行。",
            },
        },
        "required": ["code"],
    }

    def __init__(self, default_timeout: int = 30):
        self.default_timeout = default_timeout

    async def execute(self, code: str, timeout: int | None = None, cwd: str | None = None, **kwargs) -> str:
        """
        在 subprocess 中执行 Python 代码。

        Args:
            code: 要执行的 Python 代码
            timeout: 超时秒数，默认 30
            cwd: 工作目录。None 时使用当前目录。

        Returns:
            执行结果（stdout + stderr + 退出码）
        """
        timeout = timeout or self.default_timeout

        # 写入临时文件（优先写到 cwd 下，方便代码访问项目文件）
        tmp_dir = cwd if cwd else None
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
            dir=tmp_dir,
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            # 启动子进程（在 cwd 下执行）
            process = await asyncio.create_subprocess_exec(
                "python",
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,  # ★ 设置工作目录为用户工作区
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return (
                    f"[超时] 代码执行超过 {timeout} 秒，已被强制终止。\n"
                    f"请检查是否存在死循环或耗时过长的操作。"
                )

            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()

            parts = [f"[退出码: {process.returncode}]"]
            if stdout_str:
                parts.append(f"[stdout]\n{stdout_str}")
            if stderr_str:
                parts.append(f"[stderr]\n{stderr_str}")
            if not stdout_str and not stderr_str:
                parts.append("(无输出)")

            return "\n".join(parts)

        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
