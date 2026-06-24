"""
审计日志读取器 — 增量、容错、可过滤。

与 AuditLogger 配套：AuditLogger 写入 JSONL，AuditLogReader 增量读取。

核心能力：
    - 增量游标：记录文件偏移量，只读新增行
    - 防御性解析：逐行 try-except，跳过损坏行，绝不中断
    - 结构化过滤：按 decision/trace_id/time/rule_id 惰性过滤
    - 统计聚合：按决策类型/规则/工具分组计数
    - 游标持久化：JSON 文件保存/恢复

设计原则：
    - 纯逻辑模块，不依赖 Agent/LLM/Memory
    - 生成器返回模式，内存友好
    - 损坏行记录保留，供问题追溯

使用方式：
    from pyagent.harness.context.audit_reader import AuditLogReader, LogCursor

    reader = AuditLogReader(".claude/audit_fallback.jsonl")

    # 方式 1：全量读取
    all_events = reader.read_all()

    # 方式 2：增量读取（游标驱动）
    cursor = AuditLogReader.load_cursor(".claude/audit_cursor.json")
    for event in reader.read_since(cursor):
        process(event)
    AuditLogReader.save_cursor(cursor, ".claude/audit_cursor.json")

    # 方式 3：增量 + 过滤
    new_events = reader.read_since(cursor)
    blocked = AuditLogReader.filter(
        new_events, decision="BLOCK", min_risk_score=70
    )
    for event in blocked:
        alert(event)

    # 方式 4：统计
    stats = AuditLogReader.get_statistics(reader.read_all())
    print(f"BLOCK events: {stats['by_decision'].get('BLOCK', 0)}")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Generator, Iterable, Literal, Optional, Union

from .observability import SecurityAuditEvent, Sanitizer

logger = logging.getLogger("pyagent.harness.audit_reader")


# ═══════════════════════════════════════════════════════════════
# LogCursor
# ═══════════════════════════════════════════════════════════════

@dataclass
class LogCursor:
    """
    增量读取游标 — 记录上次成功读取的位置。

    每次 read_since() 调用后会原地更新，调用方负责持久化。

    Attributes:
        file_path: 审计日志文件路径。
        last_offset: 上次成功读取的字节偏移量。
        last_inode: 文件 inode（用于检测轮转/替换）。
        last_position: 已读取的行号（调试用，1-based）。
        checksum: 保留字段（供未来校验使用）。
    """

    file_path: str = ""
    last_offset: int = 0
    last_inode: int = 0
    last_position: int = 0
    checksum: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LogCursor":
        return cls(
            file_path=data.get("file_path", ""),
            last_offset=data.get("last_offset", 0),
            last_inode=data.get("last_inode", 0),
            last_position=data.get("last_position", 0),
            checksum=data.get("checksum", ""),
        )


# ═══════════════════════════════════════════════════════════════
# AuditLogReader
# ═══════════════════════════════════════════════════════════════

class AuditLogReader:
    """
    审计日志读取器 — 增量、防御性解析、结构化过滤。

    特性：
        - 增量读取：通过 LogCursor 记录文件偏移量，只读取新增行
        - 防御性解析：逐行 try-except，损坏行跳过并记录，绝不中断
        - 文件轮转检测：inode 变化或文件截断时自动重置游标
        - 并发安全：容忍尾部不完整行（其他进程正在写入）
        - 惰性过滤：返回生成器，不在内存中构建全量列表
        - 损坏行追溯：通过 corrupt_lines 属性查看所有被跳过的行

    使用方式：
        reader = AuditLogReader(".claude/audit_fallback.jsonl")
        events = reader.read_all()
        stats = AuditLogReader.get_statistics(events)
    """

    def __init__(
        self,
        file_path: str,
        sanitizer: Optional[Sanitizer] = None,
    ):
        """
        Args:
            file_path: 审计日志 JSONL 文件路径。
            sanitizer: 脱敏器实例。None 时自动创建默认 Sanitizer。
        """
        self.file_path = file_path
        self.sanitizer = sanitizer or Sanitizer()
        self._corrupt_lines: list[dict] = []

    # ── 属性 ──────────────────────────────────────

    @property
    def corrupt_lines(self) -> list[dict]:
        """
        返回最近一次读取中跳过的损坏行记录。

        每项包含:
            - position: 行号（1-based）
            - offset: 字节偏移
            - error: 错误信息
            - snippet: 行内容截断（≤200 字符）
        """
        return list(self._corrupt_lines)

    # ── 全量读取 ──────────────────────────────────

    def read_all(self) -> list[SecurityAuditEvent]:
        """
        全量读取整个审计日志文件。

        Returns:
            SecurityAuditEvent 列表。文件不存在时返回空列表。
        """
        path = Path(self.file_path)
        if not path.exists():
            logger.debug("审计日志文件不存在: %s", self.file_path)
            return []

        cursor = LogCursor(file_path=self.file_path)
        return list(self.read_since(cursor))

    # ── 增量读取 ──────────────────────────────────

    def read_since(
        self,
        cursor: Optional[LogCursor] = None,
        batch_size: int = 0,
    ) -> Generator[SecurityAuditEvent, None, None]:
        """
        从游标位置增量读取新事件。

        游标原地更新：每次调用后 cursor.last_offset 已指向
        最后一个成功读取的行末尾，可直接持久化。

        并发安全：
            - 文件尾不完整行（无 \\n）→ 跳过，不更新游标
            - 损坏行 → 跳过并记录，游标越过该行防止卡住
            - 文件被截断 → 游标重置为 0

        Args:
            cursor: 读取游标。None 时从文件开头读取。
            batch_size: 批次大小。>0 时每次 yield 一批事件后暂停。
                        0 表示一次性返回所有事件。

        Yields:
            SecurityAuditEvent 实例。
        """
        self._corrupt_lines.clear()

        if cursor is None:
            cursor = LogCursor(file_path=self.file_path)
        elif cursor.file_path != self.file_path:
            cursor.file_path = self.file_path
            cursor.last_offset = 0
            cursor.last_inode = 0
            cursor.last_position = 0

        path = Path(self.file_path)
        if not path.exists():
            logger.debug("审计日志文件不存在: %s", self.file_path)
            return

        # 文件状态检查
        try:
            stat = path.stat()
        except OSError as e:
            logger.warning("无法获取文件状态: %s — %s", self.file_path, e)
            return

        # 检测轮转/截断：文件大小小于游标偏移
        if stat.st_size < cursor.last_offset:
            logger.warning(
                "[AuditLogReader] 文件截断检测: size=%d < offset=%d，重置游标",
                stat.st_size, cursor.last_offset,
            )
            cursor.last_offset = 0
            cursor.last_position = 0

        # 检测文件替换：inode 变化（Unix）或文件大小突然变小
        current_inode = stat.st_ino if hasattr(stat, 'st_ino') else 0
        if cursor.last_inode != 0 and current_inode != 0 and \
           current_inode != cursor.last_inode:
            logger.warning(
                "[AuditLogReader] 文件替换检测: inode %d → %d，重置游标",
                cursor.last_inode, current_inode,
            )
            cursor.last_offset = 0
            cursor.last_position = 0

        cursor.last_inode = current_inode

        # 读取新增内容
        try:
            with open(path, 'r', encoding='utf-8') as f:
                f.seek(cursor.last_offset)
                raw = f.read()
        except OSError as e:
            logger.error("[AuditLogReader] 读取失败: %s — %s", self.file_path, e)
            return

        if not raw:
            return

        # 检测尾部不完整行（并发写入保护）
        complete = raw.endswith('\n')
        lines = raw.split('\n')
        process_lines = lines if complete else lines[:-1]

        # 逐行解析
        byte_offset = cursor.last_offset
        batch: list[SecurityAuditEvent] = []

        for line in process_lines:
            # 计算字节偏移（UTF-8 编码 + 换行符）
            line_bytes = len(line.encode('utf-8')) + 1  # +1 for \n
            byte_offset += line_bytes

            # 跳过空行（尾部空串或空白行），不计入 position
            if not line.strip():
                cursor.last_offset = byte_offset
                continue

            cursor.last_position += 1

            # 防御性解析
            event = self._parse_line(line, cursor.last_position, byte_offset)
            if event is not None:
                batch.append(event)
                if batch_size > 0 and len(batch) >= batch_size:
                    yield from batch
                    batch.clear()

            # 无论成败，游标越过错行（防止卡住）
            cursor.last_offset = byte_offset

        # 批次尾声
        if batch:
            yield from batch

    # ── 统计 ──────────────────────────────────────

    def count(self, cursor: Optional[LogCursor] = None) -> int:
        """
        统计新增事件数（消耗生成器）。

        Args:
            cursor: 读取游标。None 时统计全量。

        Returns:
            事件数量。
        """
        n = 0
        for _ in self.read_since(cursor):
            n += 1
        return n

    # ── 内部 ──────────────────────────────────────

    def _parse_line(
        self, line: str, position: int, offset: int,
    ) -> Optional[SecurityAuditEvent]:
        """
        防御性解析单行 JSON。

        Args:
            line: JSONL 单行文本。
            position: 行号（调试用）。
            offset: 字节偏移（调试用）。

        Returns:
            SecurityAuditEvent 或 None（解析失败）。
        """
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning(
                "[AuditLogReader] 行 %d (offset=%d): JSON 解析失败 — %s",
                position, offset, str(e)[:120],
            )
            self._corrupt_lines.append({
                "position": position,
                "offset": offset - len(line.encode('utf-8')) - 1,
                "error": f"JSONDecodeError: {e}",
                "snippet": line[:200],
            })
            return None

        try:
            event = SecurityAuditEvent(**data)
            return event
        except Exception as e:
            logger.warning(
                "[AuditLogReader] 行 %d (offset=%d): SecurityAuditEvent 构造失败 — %s",
                position, offset, str(e)[:120],
            )
            self._corrupt_lines.append({
                "position": position,
                "offset": offset - len(line.encode('utf-8')) - 1,
                "error": f"ModelValidationError: {e}",
                "snippet": line[:200],
            })
            return None

    # ── 静态方法：过滤 ────────────────────────────

    @staticmethod
    def filter(
        events: Iterable[SecurityAuditEvent],
        *,
        decision: Optional[Union[
            Literal["ALLOW", "BLOCK", "REPAIR"],
            list[str], set[str], tuple[str, ...],
        ]] = None,
        trace_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        rule_id: Optional[Union[str, list[str], set[str], tuple[str, ...]]] = None,
        min_risk_score: Optional[int] = None,
        max_risk_score: Optional[int] = None,
        tool_name: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> Generator[SecurityAuditEvent, None, None]:
        """
        惰性过滤器 — 在生成器上叠加过滤条件。

        所有条件为 AND 关系。未指定的条件不过滤。
        支持单值和多值匹配（decision/rule_id 可传入列表）。

        Args:
            events: SecurityAuditEvent 可迭代对象（生成器或列表）。
            decision: 按决策类型过滤。支持 "ALLOW" / "BLOCK" / "REPAIR"
                      或它们的列表。
            trace_id: 按 Trace ID 精确过滤。
            start_time: ISO 时间字符串，过滤此时间之后的事件（含）。
            end_time: ISO 时间字符串，过滤此时间之前的事件（含）。
            rule_id: 按门控规则 ID 过滤。
            min_risk_score: 最低风险评分（≥）。
            max_risk_score: 最高风险评分（≤）。
            tool_name: 按工具名称过滤。
            phase: 按 PEVR 阶段过滤。

        Yields:
            满足所有条件的 SecurityAuditEvent。
        """
        for event in events:
            # decision 过滤
            if decision is not None:
                if isinstance(decision, (list, tuple, set, frozenset)):
                    if event.decision not in decision:
                        continue
                elif event.decision != decision:
                    continue

            # trace_id 过滤
            if trace_id is not None and event.trace_id != trace_id:
                continue

            # rule_id 过滤
            if rule_id is not None:
                if isinstance(rule_id, (list, tuple, set, frozenset)):
                    if event.rule_id not in rule_id:
                        continue
                elif event.rule_id != rule_id:
                    continue

            # risk_score 范围过滤
            if min_risk_score is not None and event.risk_score < min_risk_score:
                continue
            if max_risk_score is not None and event.risk_score > max_risk_score:
                continue

            # tool_name 过滤
            if tool_name is not None and event.tool_name != tool_name:
                continue

            # phase 过滤
            if phase is not None and event.phase != phase:
                continue

            # 时间窗口过滤
            if start_time is not None or end_time is not None:
                ts = event.timestamp
                if not ts:
                    continue  # 无时间戳的事件被时间过滤排除
                if start_time is not None and ts < start_time:
                    continue
                if end_time is not None and ts > end_time:
                    continue

            yield event

    # ── 静态方法：统计聚合 ─────────────────────────

    @staticmethod
    def get_statistics(
        events: Iterable[SecurityAuditEvent],
    ) -> dict:
        """
        对事件流进行聚合统计（消耗生成器）。

        返回统计项：
            - total_events: 事件总数
            - by_decision: {decision: count}
            - by_rule_id: {rule_id: count}
            - by_tool: {tool_name: count}
            - by_phase: {phase: count}
            - avg_risk_score: 平均风险评分
            - max_risk_score: 最高风险评分
            - p0_count: P0 级事件数（BLOCK + REPAIR）
            - time_range: {start, end} ISO 时间范围
            - top_blocked_tools: BLOCK 最多的 5 个工具 [(tool, count)]

        Args:
            events: SecurityAuditEvent 可迭代对象。

        Returns:
            统计 dict。
        """
        total = 0
        by_decision: dict[str, int] = {}
        by_rule: dict[str, int] = {}
        by_tool: dict[str, int] = {}
        by_phase: dict[str, int] = {}
        risk_scores: list[int] = []
        p0_count = 0
        time_min: str = ""
        time_max: str = ""

        # BLOCK 事件的工具计数（用于 top_blocked_tools）
        blocked_by_tool: dict[str, int] = {}

        for event in events:
            total += 1

            # 决策分布
            by_decision[event.decision] = by_decision.get(event.decision, 0) + 1

            # 规则分布
            if event.rule_id:
                by_rule[event.rule_id] = by_rule.get(event.rule_id, 0) + 1

            # 工具分布
            if event.tool_name:
                by_tool[event.tool_name] = by_tool.get(event.tool_name, 0) + 1

            # 阶段分布
            if event.phase:
                by_phase[event.phase] = by_phase.get(event.phase, 0) + 1

            # 风险评分
            risk_scores.append(event.risk_score)

            # P0 计数
            if event.is_p0:
                p0_count += 1

            # BLOCK 工具排名
            if event.decision == "BLOCK" and event.tool_name:
                blocked_by_tool[event.tool_name] = \
                    blocked_by_tool.get(event.tool_name, 0) + 1

            # 时间范围
            if event.timestamp:
                if not time_min or event.timestamp < time_min:
                    time_min = event.timestamp
                if not time_max or event.timestamp > time_max:
                    time_max = event.timestamp

        # Top-5 blocked tools
        top_blocked = sorted(
            blocked_by_tool.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "total_events": total,
            "by_decision": by_decision,
            "by_rule_id": by_rule,
            "by_tool": by_tool,
            "by_phase": by_phase,
            "avg_risk_score": (
                round(sum(risk_scores) / len(risk_scores), 1)
                if risk_scores else 0.0
            ),
            "max_risk_score": max(risk_scores) if risk_scores else 0,
            "p0_count": p0_count,
            "time_range": {"start": time_min, "end": time_max},
            "top_blocked_tools": top_blocked,
        }

    # ── 静态方法：游标持久化 ───────────────────────

    @staticmethod
    def save_cursor(cursor: LogCursor, cursor_path: str) -> None:
        """
        将游标持久化为 JSON 文件。

        Args:
            cursor: 当前游标。
            cursor_path: 游标文件路径（如 .claude/audit_cursor.json）。
        """
        path = Path(cursor_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(cursor.to_dict(), f, ensure_ascii=False)
            logger.debug(
                "[AuditLogReader] 游标已保存: offset=%d pos=%d → %s",
                cursor.last_offset, cursor.last_position, cursor_path,
            )
        except OSError as e:
            logger.error("[AuditLogReader] 游标保存失败: %s — %s", cursor_path, e)

    @staticmethod
    def load_cursor(cursor_path: str) -> Optional[LogCursor]:
        """
        从 JSON 文件恢复游标。

        Args:
            cursor_path: 游标文件路径。

        Returns:
            LogCursor 或 None（文件不存在 / 损坏）。
        """
        path = Path(cursor_path)
        if not path.exists():
            return None

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cursor = LogCursor.from_dict(data)
            logger.debug(
                "[AuditLogReader] 游标已恢复: offset=%d pos=%d ← %s",
                cursor.last_offset, cursor.last_position, cursor_path,
            )
            return cursor
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning(
                "[AuditLogReader] 游标加载失败: %s — %s",
                cursor_path, str(e)[:120],
            )
            return None

    @staticmethod
    def delete_cursor(cursor_path: str) -> bool:
        """
        删除游标文件（任务完成时清理）。

        Args:
            cursor_path: 游标文件路径。

        Returns:
            True 如果已删除或不存在，False 如果删除失败。
        """
        path = Path(cursor_path)
        try:
            if path.exists():
                path.unlink()
                logger.debug("[AuditLogReader] 游标已删除: %s", cursor_path)
            return True
        except OSError as e:
            logger.error("[AuditLogReader] 游标删除失败: %s — %s", cursor_path, e)
            return False
