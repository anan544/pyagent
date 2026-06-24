#!/usr/bin/env python3
"""
PyAgent 终端交互式 Chat

类似 Claude Code 的体验：输入问题 → 看 Agent 思考/调用工具 → 返回答案。
支持多轮对话，上下文自动累积。

用法:
    python -m pyagent.cli.chat
    python -m pyagent.cli.chat --config config.prod.yaml

快捷键:
    Ctrl+C  中断当前回复
    Ctrl+D  退出
    /clear  清空对话历史
"""

import asyncio
import sys
import os
import signal

_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HOME not in sys.path:
    sys.path.insert(0, _HOME)

from pyagent.harness.config.loader import ConfigLoader
from pyagent.harness.api.dependencies import AgentManager


# ── ANSI 颜色 ──────────────────────────────

class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


LOGO = f"""
{Color.CYAN}╔══════════════════════════════════════╗
║       {Color.BOLD}PyAgent Chat{Color.RESET}{Color.CYAN}                    ║
║       {Color.DIM}v0.10.0 + CubeSandbox{Color.RESET}{Color.CYAN}          ║
╚══════════════════════════════════════╝{Color.RESET}
"""

HELP_TEXT = f"""
{Color.DIM}命令:{Color.RESET}
  {Color.YELLOW}/clear{Color.RESET}   清空对话历史
  {Color.YELLOW}/history{Color.RESET}  查看历史消息数
  {Color.YELLOW}/help{Color.RESET}     显示帮助
  {Color.YELLOW}Ctrl+C{Color.RESET}    中断当前操作
  {Color.YELLOW}Ctrl+D{Color.RESET}    退出
"""


class TerminalChat:
    """终端交互式 Agent 对话。"""

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path
        self.manager = AgentManager()
        self.running = True
        self._interrupted = False

    async def start(self):
        """初始化并进入主循环。"""
        print(LOGO)

        # 加载配置
        try:
            await self.manager.initialize(self.config_path)
        except Exception as e:
            print(f"{Color.RED}✗ 配置加载失败: {e}{Color.RESET}")
            sys.exit(1)

        cfg = self.manager._config
        sandbox_info = ""
        if getattr(cfg, "sandbox", None) and getattr(cfg.sandbox, "enabled", False):
            sandbox_info = f" | {Color.GREEN}沙箱: {cfg.sandbox.vm_host}{Color.RESET}"
        """模型信息"""
        print(f"{Color.DIM}模型: {cfg.llm.model}{sandbox_info}{Color.RESET}")
        print(HELP_TEXT)
        print(f"{Color.DIM}{'─' * 50}{Color.RESET}\n")

        signal.signal(signal.SIGINT, self._handle_interrupt)
        session_id = f"chat-{os.getpid()}"

        while self.running:
            try:
                # 读用户输入
                user_input = await self._get_input()
                if user_input is None:  # Ctrl+D
                    break
                if not user_input.strip():
                    continue

                # 处理命令
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue

                # 调 Agent
                await self._run_agent(user_input, session_id)

            except KeyboardInterrupt:
                print(f"\n{Color.YELLOW}已中断{Color.RESET}")
                self._interrupted = False
            except EOFError:
                break

        print(f"\n{Color.DIM}再见！{Color.RESET}")
        await self.manager.shutdown()

    async def _get_input(self) -> str | None:
        """异步读取用户输入。"""
        try:
            loop = asyncio.get_event_loop()
            prompt = f"\n{Color.BOLD}{Color.GREEN}You › {Color.RESET}"
            return await loop.run_in_executor(None, lambda: input(prompt))
        except EOFError:
            return None

    def _handle_interrupt(self, signum, frame):
        self._interrupted = True
        print(f"\n{Color.YELLOW}⏎ 中断中...{Color.RESET}")

    def _handle_command(self, cmd: str):
        cmd = cmd.strip().lower()
        if cmd == "/clear":
            print(f"{Color.DIM}对话历史已清空{Color.RESET}")
        elif cmd == "/history":
            print(f"{Color.DIM}多轮对话由 Agent 内部管理，每次 /clear 重置{Color.RESET}")
        elif cmd == "/help":
            print(HELP_TEXT)
        else:
            print(f"{Color.RED}未知命令: {cmd}{Color.RESET}")

    async def _run_agent(self, user_input: str, session_id: str):
        """执行 Agent 并打印结果。"""
        if self.manager.agent is None:
            print(f"{Color.RED}Agent 未初始化{Color.RESET}")
            return

        print(f"\n{Color.BOLD}{Color.BLUE}Agent › {Color.RESET}", end="", flush=True)

        try:
            self.manager.agent.config.verbose = False
            result = await self.manager.agent.run(user_input, session_id=session_id)
            print(result)

        except asyncio.CancelledError:
            print(f"\n{Color.YELLOW}已取消{Color.RESET}")
        except Exception as e:
            print(f"\n{Color.RED}✗ 执行错误: {e}{Color.RESET}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PyAgent 终端交互式 Chat")
    parser.add_argument("--config", "-c", default=None, help="YAML 配置文件路径")
    args = parser.parse_args()

    chat = TerminalChat(args.config)
    try:
        asyncio.run(chat.start())
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
