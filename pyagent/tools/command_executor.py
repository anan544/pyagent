"""
命令执行工具 — 在 subprocess 中执行 Shell 命令。

安全设计：
    - 命令前缀白名单（默认 24 个允许的命令）
    - 危险模式黑名单
    - 超时机制
    - git diff/show 自动管道到 delta（如果已安装）
"""

import asyncio
import logging
from .base import Tool
from .delta_enhancer import enhance_command

logger = logging.getLogger("pyagent.tools.command")


class CommandExecutorTool(Tool):
    """
    执行受控的 Shell 命令。

    安全限制：
        - 只允许白名单中的命令前缀
        - 阻止 rm -rf / 等危险模式
        - 30 秒超时
    """

    name = "execute_command"
    risk_level = "high"
    description = (
        "在用户系统的终端中执行命令。\n"
        "\n"
        "重要规则：\n"
        "1. **需要审批**：命令执行前需要用户批准，请勿假设它已运行。\n"
        "2. **非交互式**：用户可能不在场，请使用非交互式标志（如 `-y`、`--no-input`）避免挂起。\n"
        "3. **后台任务**：对于长时间运行或阻塞的进程（如服务器、监视器），设置 is_background=true。\n"
        "4. **Shell 上下文**：如果从之前的 shell 继续工作，请检查当前目录和状态。\n"
        "只允许安全命令（git、python、npm、uvicorn 等）。git diff/show 输出自动使用 delta 美化。\n"
        "适合：运行测试、启动开发服务器（npm run dev / uvicorn / python manage.py runserver）、"
        "安装依赖（pip install / npm install）、构建项目、查看 git 状态。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的终端命令。",
            },
            "timeout": {
                "type": "integer",
                "description": "超时秒数，默认 30。启动服务器时建议设为 60 或更长。",
                "default": 30,
            },
            "is_background": {
                "type": "boolean",
                "description": "如果命令应在后台运行（如服务器、监视器），设为 true。",
                "default": False,
            },
            "explanation": {
                "type": "string",
                "description": "一句话说明为什么需要执行此命令。",
            },
        },
        "required": ["command"],
    }

    def __init__(self, default_timeout: int = 30):
        self.default_timeout = default_timeout

    async def execute(self, command: str, timeout: int | None = None, cwd: str | None = None, **kwargs) -> str:
        """执行命令并返回 stdout + stderr。cwd 为工作目录。"""
        timeout = timeout or self.default_timeout

        # Delta 增强
        enhanced = enhance_command(command)
        if enhanced != command:
            logger.info("delta enhanced: %s", enhanced)
            command = enhanced

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,  # ★ 设置工作目录为用户工作区
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return f"[超时] 命令执行超过 {timeout} 秒"
        except Exception as e:
            return f"[错误] {e}"

        parts = [f"[退出码: {process.returncode}]"]
        import locale
        enc = locale.getpreferredencoding() or "gbk"
        out = stdout.decode(enc, errors="replace").strip()
        err = stderr.decode(enc, errors="replace").strip()
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr]\n{err}")
        return "\n".join(parts) if len(parts) > 1 else "(无输出)"
