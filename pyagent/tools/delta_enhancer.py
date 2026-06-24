"""
Delta Enhancer — git diff/show 自动美化输出。

检测到 git diff/log/show 命令时，自动管道到 delta（如果已安装）。
delta 提供：语法高亮、行号、+/- 侧边栏、文件导航。

安装 delta: https://github.com/dandavison/delta
    cargo install git-delta  # 或
    winget install dandavison.delta  # Windows

用法：
    from .delta_enhancer import enhance_command
    enhanced_cmd = enhance_command("git diff")  # → "git --no-pager diff | delta"
"""

import shutil
from typing import Optional

DELTA_AVAILABLE: Optional[bool] = None


def _check_delta() -> bool:
    """检查 delta 是否已安装。结果缓存。"""
    global DELTA_AVAILABLE
    if DELTA_AVAILABLE is None:
        DELTA_AVAILABLE = shutil.which("delta") is not None
    return DELTA_AVAILABLE


def enhance_command(command: str) -> str:
    """
    增强 git 命令——自动附加 delta 管道。

    支持的自动增强：
        git diff        → git --no-pager diff | delta
        git show        → git --no-pager show | delta
        git log -p      → git --no-pager log -p | delta
        git stash show -p → git --no-pager stash show -p | delta

    如果 delta 未安装，原样返回。
    """
    if not _check_delta():
        return command

    cmd = command.strip()

    # 已包含 delta 或 --no-pager 的不重复处理
    if "delta" in cmd:
        return command
    if "--no-pager" in cmd:
        return command

    # 匹配模式: git <subcommand>
    parts = cmd.split()
    if len(parts) < 2 or parts[0] != "git":
        return command

    subcmd = parts[1]

    # git diff / git show / git log (带 -p 或 --patch)
    if subcmd in ("diff", "show"):
        # 插入 --no-pager（如果还没有）
        idx = cmd.index(subcmd) + len(subcmd)
        enhanced = cmd[:idx] + " --no-pager" + cmd[idx:] + " | delta"
        return enhanced

    if subcmd == "log":
        if any(p in parts for p in ("-p", "--patch", "--stat")):
            idx = cmd.index(subcmd) + len(subcmd)
            enhanced = cmd[:idx] + " --no-pager" + cmd[idx:] + " | delta"
            return enhanced

    if subcmd == "stash":
        if len(parts) >= 3 and parts[2] in ("show", "list"):
            idx = cmd.index(parts[2]) + len(parts[2])
            enhanced = cmd[:idx] + " --no-pager" + cmd[idx:] + " | delta"
            return enhanced

    return command


def delta_version() -> Optional[str]:
    """返回 delta 版本字符串，未安装返回 None。"""
    if not _check_delta():
        return None
    import subprocess
    try:
        r = subprocess.run(["delta", "--version"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return None
