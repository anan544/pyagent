"""
会话风险上下文 — 滑动窗口工具调用追踪（优化 2）。

维护当前会话中最近的工具调用记录，用于组合风险检测。
每条记录包含工具名 + 关键参数摘要（params_summary），
支持后续的字段级 combo 精炼匹配（如 match_on: file_path）。

设计要点：
    - 基于 monotonic time 的滑动窗口，O(1) 过期驱逐
    - params_summary 从工具参数中提取 file_path、command 等语义字段
    - 线程安全：所有方法同步，由 GovernanceWrapper 层协调并发

使用方式：
    ctx = SessionRiskContext(window_seconds=300.0)
    ctx.record_call("write_file", {"path": "a.py", "content": "..."})
    recent = ctx.recent_tool_names(include_current="execute_python")
    records = ctx.recent_records()  # 用于字段级匹配
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("pyagent.security.session_risk")


# ── 参数摘要提取 ──────────────────────────────────

# 从各工具 params 中提取的语义字段名（按优先级）
_PARAM_EXTRACTORS: dict[str, list[str]] = {
    "write_file": ["path", "file_path"],
    "read_file": ["path", "file_path"],
    "delete_file": ["path", "file_path"],
    "execute_python": ["code"],   # 截断前 80 字符
    "execute_command": ["command"],
    "search_content": ["pattern", "path"],
}


def _extract_params_summary(tool_name: str, params: Optional[dict]) -> dict:
    """从工具参数中提取关键字段摘要。

    Args:
        tool_name: 工具名。
        params: LLM 传入的参数 dict。

    Returns:
        提取的 {field: truncated_value} dict。
    """
    if not params:
        return {}
    extractors = _PARAM_EXTRACTORS.get(tool_name, [])
    summary: dict[str, str] = {}
    for key in extractors:
        value = params.get(key)
        if value is None:
            # 尝试备选键（如 file_path 作为 path 的别名）
            continue
        str_value = str(value)
        # 截断以控制内存占用
        if key in ("code", "content"):
            summary[key] = str_value[:80]
        else:
            summary[key] = str_value[:200]
    return summary


# ── 记录类型 ──────────────────────────────────────

@dataclass
class _ToolCallRecord:
    """单次工具调用记录（内部类型）。"""
    tool_name: str
    params_summary: dict
    timestamp: float


# ── SessionRiskContext ────────────────────────────

class SessionRiskContext:
    """会话级滑动窗口风险上下文。

    维护最近 N 条工具调用记录（时间窗口 + 数量上限），
    提供工具名集合查询 + 原始记录查询，供 combo 检测使用。
    """

    def __init__(
        self,
        window_seconds: float = 300.0,
        max_records: int = 50,
    ):
        """
        Args:
            window_seconds: 记录保留时长（秒）。默认 5 分钟。
            max_records: 最大保留记录数（安全阀）。
        """
        self._window_s = window_seconds
        self._max_records = max_records
        self._records: deque[_ToolCallRecord] = deque()

    # ── 公共 API ──────────────────────────────────

    def record_call(
        self,
        tool_name: str,
        params: Optional[dict[str, Any]] = None,
    ):
        """记录一次成功的工具调用。

        应在工具执行成功**之后**调用，确保风险上下文的准确性。

        Args:
            tool_name: 工具名称。
            params: 工具参数（自动提取关键字段摘要）。
        """
        now = time.monotonic()
        summary = _extract_params_summary(tool_name, params)
        self._records.append(_ToolCallRecord(
            tool_name=tool_name,
            params_summary=summary,
            timestamp=now,
        ))
        self._evict_expired()
        # 数量上限驱逐（最旧的优先）
        while len(self._records) > self._max_records:
            self._records.popleft()

    def recent_tool_names(
        self,
        include_current: Optional[str] = None,
    ) -> list[str]:
        """获取窗口内去重后的工具名列表。

        Args:
            include_current: 候选工具名（尚未记录的），会被加入结果集合。

        Returns:
            去重工具名列表。
        """
        self._evict_expired()
        names = {r.tool_name for r in self._records}
        if include_current:
            names.add(include_current)
        return list(names)

    def recent_records(self) -> list[dict]:
        """获取窗口内的原始记录列表（用于字段级 combo 精炼匹配）。

        Returns:
            记录列表，每条为 {"tool_name": str, "params_summary": dict, "timestamp": float}。
        """
        self._evict_expired()
        return [
            {
                "tool_name": r.tool_name,
                "params_summary": dict(r.params_summary),
                "timestamp": r.timestamp,
            }
            for r in self._records
        ]

    def reset(self):
        """清空所有记录（新会话开始时调用）。"""
        self._records.clear()

    # ── 内部方法 ──────────────────────────────────

    def _evict_expired(self):
        """移除窗口外的过期记录。"""
        now = time.monotonic()
        cutoff = now - self._window_s
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()

    # ── 查询属性 ──────────────────────────────────

    @property
    def window_size(self) -> int:
        """当前窗口内的记录数。"""
        self._evict_expired()
        return len(self._records)

    @property
    def window_seconds(self) -> float:
        """窗口时长（秒）。"""
        return self._window_s
