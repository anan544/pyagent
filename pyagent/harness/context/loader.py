"""
上下文文件加载器。

负责加载 agent.md 等多文件上下文规范：
    - 支持相对路径（相对于配置文件所在目录）和绝对路径
    - 自动回退查找项目根目录 agent.md
    - 多文件按配置顺序拼接，添加来源标记
"""

import os
from pathlib import Path
from typing import Optional


class ContextFileLoader:
    """
    项目上下文文件加载器。

    加载链：
        1. 从配置指定的 context_files 列表加载
        2. 若未指定，自动查找项目根目录 agent.md
        3. 若都不存在，返回空列表（不报错）
    """

    def __init__(self, config_dir: Optional[str | Path] = None):
        """
        Args:
            config_dir: 配置文件所在目录，用于解析相对路径。
                        默认使用当前工作目录。
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()

    # ── 主入口 ──────────────────────────────────────

    def load(
        self,
        context_files: Optional[list[str]] = None,
        project_root: Optional[str | Path] = None,
    ) -> list[str]:
        """
        加载上下文文件，返回内容列表（含来源标记）。

        Args:
            context_files: 配置文件路径列表，支持相对/绝对路径。
            project_root: 项目根目录，用于自动查找 agent.md。

        Returns:
            格式如 ["[来源: ./agent.md]\n文件内容", ...]
        """
        loaded: list[str] = []

        if context_files:
            loaded = self._load_paths(context_files)
        else:
            loaded = self._auto_lookup(project_root)

        return loaded

    # ── 内部 ────────────────────────────────────────

    def _load_paths(self, paths: list[str]) -> list[str]:
        """按顺序加载指定路径的文件。"""
        results: list[str] = []
        for p in paths:
            path = self._resolve(p)
            content = self._read_file(path)
            if content is not None:
                results.append(f"[来源: {p}]\n{content}")
        return results

    def _auto_lookup(self, project_root: Optional[str | Path] = None) -> list[str]:
        """自动查找 agent.md。"""
        root = Path(project_root) if project_root else Path.cwd()
        candidate = root / "agent.md"
        content = self._read_file(candidate)
        if content is not None:
            return [f"[来源: agent.md]\n{content}"]
        return []

    def _resolve(self, path: str) -> Path:
        """解析路径：绝对路径直接返回，相对路径以 config_dir 为基准。"""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.config_dir / p).resolve()

    @staticmethod
    def _read_file(path: Path) -> Optional[str]:
        """读取文件，不存在时返回 None。"""
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            pass
        return None
