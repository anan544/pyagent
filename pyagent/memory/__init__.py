"""
PyAgent 长期记忆模块。

提供基于 SQLite 的消息持久化、滑动窗口管理和上下文压缩。
架构：Database (纯 SQL) → MemoryManager (业务) → Agent (集成)
"""

from .manager import MemoryManager, LoadResult
from .budget import TokenBudget
from .compressor import ContextCompressor

__all__ = [
    "MemoryManager",
    "LoadResult",
    "TokenBudget",
    "ContextCompressor",
]
