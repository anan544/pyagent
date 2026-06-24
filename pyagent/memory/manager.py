"""
MemoryManager — 记忆管理器的业务逻辑层。

对内封装 Database 操作，对外提供消息粒度的接口。
Agent 只需要知道 MemoryManager，不需要知道 SQLite 的存在。

v0.3.0 新增：
    - 滑动窗口（Token Budget 驱动的消息裁剪）
    - 上下文压缩（LLM 生成结构化摘要替换早期消息）
    - 增量压缩（只压缩自上次摘要以来的新消息）

职责：
    - 在 Message 对象和数据库行之间做序列化/反序列化
    - Token 估算（简单版：字符数 / 4）
    - 滑动窗口管理（精确区 vs 压缩区）
    - 提供 load / save / list / delete 等业务接口
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..core.message import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
    Message,
)
from .database import Database
from .budget import TokenBudget
from .compressor import ContextCompressor, SUMMARY_ROLE


# 默认每次加载最近的消息数量，防止冷启动时上下文爆炸
DEFAULT_LOAD_LIMIT = 1000  # 提高到 1000，由 Token Budget 控制裁剪


@dataclass
class LoadResult:
    """
    load_messages() 的返回结果。

    包含处理后的消息列表和压缩/裁剪的统计信息。
    """
    messages: list[Message] = field(default_factory=list)
    """最终返回的消息列表：摘要（如有）+ 近期消息"""

    total_tokens: int = 0
    """返回消息的 Token 总数（估算值）"""

    original_count: int = 0
    """数据库中原始消息数量"""

    original_tokens: int = 0
    """数据库中原始消息的 Token 总数"""

    compressed_count: int = 0
    """被压缩的消息数量（这些消息已从 DB 删除，替换为摘要）"""

    trimmed_count: int = 0
    """被滑动窗口裁剪的消息数量（单纯丢弃，无摘要）"""

    summary: str | None = None
    """压缩摘要的文本内容（如果有）"""

    @property
    def was_compressed(self) -> bool:
        """是否执行了压缩。"""
        return self.compressed_count > 0

    @property
    def was_trimmed(self) -> bool:
        """是否执行了滑动窗口裁剪。"""
        return self.trimmed_count > 0


class MemoryManager:
    """
    Agent 的长期记忆管理器。

    使用方式：
        # 基础用法
        memory = MemoryManager("my_project.db")
        await memory.create_session("session-1")

        # 带滑动窗口（不调用 LLM）
        budget = TokenBudget.for_model("deepseek-chat")
        result = await memory.load_messages("session-1", budget=budget)
        # result.messages 已被裁剪到预算内

        # 带上下文压缩（调用 LLM）
        compressor = ContextCompressor(llm_provider)
        result = await memory.load_messages(
            "session-1", budget=budget, compressor=compressor
        )
        # 早期消息被压缩为结构化摘要
    """

    def __init__(self, db_path: str = "pyagent_memory.db"):
        """
        Args:
            db_path: SQLite 数据库文件路径。
        """
        self.db = Database(db_path)

    # ── Session 管理 ───────────────────────────────────────────

    async def create_session(
        self, session_id: str | None = None, metadata: dict | None = None
    ) -> str:
        """创建新会话。"""
        sid = session_id or uuid.uuid4().hex[:12]
        await self.db.create_session(sid, metadata)
        return sid

    async def get_session(self, session_id: str) -> dict | None:
        """获取会话信息。"""
        return await self.db.get_session(session_id)

    async def list_sessions(self, limit: int = 20) -> list[dict]:
        """列出最近的会话。"""
        return await self.db.list_sessions(limit)

    async def delete_session(self, session_id: str) -> None:
        """删除会话及其所有消息。"""
        await self.db.delete_session(session_id)

    async def update_session(self, session_id: str) -> None:
        """更新会话时间戳。"""
        await self.db.update_session_timestamp(session_id)

    # ── 消息存取 ───────────────────────────────────────────────

    async def save_message(self, session_id: str, message: Message) -> int:
        """
        保存一条消息到数据库（增量写入）。
        """
        row = {
            "session_id": session_id,
            "role": message.role,
            "token_count": self._estimate_tokens(message),
        }

        if isinstance(message, AssistantMessage):
            row["content"] = message.content
            if message.tool_calls:
                row["tool_calls"] = json.dumps(
                    [
                        {"id": tc.id, "function_name": tc.function_name,
                         "arguments": tc.arguments}
                        for tc in message.tool_calls
                    ],
                    ensure_ascii=False,
                )
            else:
                row["tool_calls"] = None

        elif isinstance(message, ToolMessage):
            row["content"] = message.content
            row["tool_call_id"] = message.tool_call_id
            row["tool_name"] = message.name

        else:
            row["content"] = message.content

        msg_id = await self.db.insert_message(row)
        await self.db.update_session_timestamp(session_id)
        return msg_id

    async def load_messages(
        self,
        session_id: str,
        limit: int = DEFAULT_LOAD_LIMIT,
        budget: TokenBudget | None = None,
        compressor: ContextCompressor | None = None,
    ) -> list[Message] | LoadResult:
        """
        加载会话的历史消息。

        **无 budget 时（v0.2.0 兼容模式）：**
            直接返回最近 N 条消息列表（list[Message]）。

        **有 budget 时（滑动窗口 + 可选压缩）：**
            返回 LoadResult，包含处理后的消息列表和统计信息。

        Args:
            session_id: 会话 ID。
            limit: fallback 模式下最多加载多少条。
            budget: 可选的 TokenBudget，触发滑动窗口逻辑。
            compressor: 可选的 ContextCompressor，触发 LLM 压缩。

        Returns:
            无 budget → list[Message]（向后兼容）
            有 budget → LoadResult（含统计信息）
        """
        # ── 模式 1：无预算，兼容 v0.2.0 行为 ──
        if budget is None:
            rows = await self.db.load_messages(session_id, limit=limit)
            messages: list[Message] = []
            for row in rows:
                msg = self._row_to_message(row)
                if msg is not None and not isinstance(msg, SystemMessage):
                    messages.append(msg)
            return messages  # ← 直接返回 list，向后兼容

        # ── 模式 2：有预算，执行滑动窗口/压缩 ──
        # 加载全部消息（需要完整遍历以计算 Token 分配）
        all_rows = await self.db.load_messages_unlimited(session_id)
        original_count = len(all_rows)
        original_tokens = sum(r.get("token_count", 0) for r in all_rows)
        original_tokens = int(original_tokens)

        # 转换为 Message 对象（跳过 system）
        all_messages: list[Message] = []
        summary_messages: list[Message] = []
        for row in all_rows:
            msg = self._row_to_message(row)
            if msg is None:
                continue
            if isinstance(msg, SystemMessage):
                continue
            # 分离摘要消息
            if row.get("is_summary"):
                summary_messages.append(msg)
            else:
                all_messages.append(msg)

        # 如果总 Token 在预算内，直接返回
        non_summary_tokens = sum(
            r.get("token_count", 0) for r in all_rows
            if not r.get("is_summary")
        )
        non_summary_tokens = int(non_summary_tokens)

        if non_summary_tokens <= budget.available_budget:
            result_msgs = summary_messages + all_messages
            return LoadResult(
                messages=result_msgs,
                total_tokens=non_summary_tokens + sum(
                    r.get("token_count", 0) for r in all_rows
                    if r.get("is_summary")
                ),
                original_count=original_count,
                original_tokens=original_tokens,
            )

        # ── Token 超预算：分割为压缩区 + 精确区 ──
        compression_zone, precision_zone = budget.split_messages(all_messages)

        result = LoadResult(
            original_count=original_count,
            original_tokens=original_tokens,
        )

        # ── 有 compressor 且压缩区非空 → LLM 压缩 ──
        if compressor and compression_zone:
            summary_text = await self._compress_and_persist(
                session_id, compression_zone, compressor, summary_messages
            )
            result.summary = summary_text
            result.compressed_count = len(compression_zone)

            # 将摘要作为 UserMessage 注入（LLM 可读的结构化文本）
            summary_msg = UserMessage(
                content=f"[上下文摘要 — 早期对话精华]\n{summary_text}"
            )
            result.messages = [summary_msg] + precision_zone

        elif compression_zone:
            # ── 无 compressor → 纯滑动窗口（丢弃早期消息）──
            result.trimmed_count = len(compression_zone)
            result.messages = precision_zone

        else:
            # 压缩区为空（不常见，理论不会到这里）
            result.messages = precision_zone

        # 计算返回消息的 Token 数
        result.total_tokens = sum(
            self._estimate_tokens(m) for m in result.messages
        )
        return result

    async def message_count(self, session_id: str) -> int:
        """返回会话中的消息总数。"""
        return await self.db.message_count(session_id)

    # ── 压缩（内部）───────────────────────────────────────────

    async def _compress_and_persist(
        self,
        session_id: str,
        messages: list[Message],
        compressor: ContextCompressor,
        existing_summaries: list[Message],
    ) -> str:
        """
        将消息列表压缩为摘要，持久化到数据库，删除原始消息。

        增量压缩逻辑：
            - 如果已有旧摘要 → 将其内容作为上下文传给 LLM
            - 新旧摘要合并 → 单一结构化摘要
            - 删除被压缩的原始消息，写入新摘要

        Args:
            session_id: 会话 ID。
            messages: 需要压缩的消息列表。
            compressor: 压缩器实例。
            existing_summaries: 已有的摘要消息。

        Returns:
            最终的摘要文本。
        """
        # 如果有旧摘要，作为上下文前缀
        prefix = ""
        if existing_summaries:
            old_texts = []
            for sm in existing_summaries:
                content = sm.content if hasattr(sm, 'content') else str(sm)
                if content:
                    old_texts.append(content)
            if old_texts:
                prefix = "【历史摘要（已确认的事实）】\n" + "\n---\n".join(old_texts) + "\n\n"

        # 生成新摘要
        summary_text = await compressor.compress(messages)

        # 合并：旧摘要在前，新摘要在后
        full_summary = prefix + summary_text if prefix else summary_text

        # 持久化：删除被压缩的消息 + 旧摘要，写入新合并摘要
        if messages:
            min_id = self._get_message_db_id(messages[0])
            max_id = self._get_message_db_id(messages[-1])
            if min_id and max_id:
                # 删除被压缩的原始消息
                await self.db.delete_messages_in_range(session_id, min_id, max_id)

                # 删除旧的摘要消息（将被新的合并摘要替代）
                # 注意：必须用 delete_message_by_id 而非 delete_messages_in_range，
                # 因为后者有 AND is_summary=0 过滤，无法删除摘要行。
                for sm in existing_summaries:
                    sm_id = self._get_message_db_id(sm)
                    if sm_id:
                        await self.db.delete_message_by_id(session_id, sm_id)

        # 写入新摘要
        summary_tokens = max(1, len(full_summary) // 4)
        await self.db.insert_summary(session_id, full_summary, summary_tokens)

        return full_summary

    @staticmethod
    def _get_message_db_id(message: Message) -> int | None:
        """从 Message 对象获取数据库 ID（如果存在）。"""
        # Message 对象本身没有 db id，需要从保存时返回
        # 这里通过一个隐藏属性获取（在 save_message 时设置）
        return getattr(message, '_db_id', None)

    # ── 序列化/反序列化 ────────────────────────────────────────

    def _row_to_message(self, row: dict) -> Message | None:
        """将数据库行转换为对应的 Message 对象。"""
        role = row["role"]
        content = row.get("content")

        if role == "system":
            msg = SystemMessage(content=content or "")
            msg._db_id = row["id"]  # type: ignore
            return msg

        elif role == "user":
            msg = UserMessage(content=content or "")
            msg._db_id = row["id"]  # type: ignore
            return msg

        elif role == "assistant":
            tool_calls = None
            tc_json = row.get("tool_calls")
            if tc_json:
                try:
                    tc_list = json.loads(tc_json)
                    tool_calls = [
                        ToolCall(
                            id=tc["id"],
                            function_name=tc["function_name"],
                            arguments=tc["arguments"],
                        )
                        for tc in tc_list
                    ]
                except (json.JSONDecodeError, KeyError):
                    pass
            msg = AssistantMessage(content=content, tool_calls=tool_calls)
            msg._db_id = row["id"]  # type: ignore
            return msg

        elif role == "tool":
            msg = ToolMessage(
                content=content or "",
                tool_call_id=row.get("tool_call_id", ""),
                name=row.get("tool_name", ""),
            )
            msg._db_id = row["id"]  # type: ignore
            return msg

        elif role == SUMMARY_ROLE:
            msg = UserMessage(content=content or "")
            msg._db_id = row["id"]  # type: ignore
            return msg

        return None

    # ── Token 估算 ─────────────────────────────────────────────

    @staticmethod
    def _estimate_tokens(message: Message) -> int:
        """简单 Token 估算：字符数 / 4。"""
        text = ""

        if isinstance(message, (SystemMessage, UserMessage, ToolMessage)):
            text = message.content or ""

        elif isinstance(message, AssistantMessage):
            text = message.content or ""
            if message.tool_calls:
                for tc in message.tool_calls:
                    text += tc.function_name
                    text += json.dumps(tc.arguments, ensure_ascii=False)

        return max(1, len(text) // 4)

    # ── 生命周期 ─────────────────────────────────────────────────

    async def close(self) -> None:
        """关闭数据库连接，释放资源。"""
        await self.db.close()
