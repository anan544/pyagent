"""
Database Layer — 纯 SQLite 操作，不包含任何业务逻辑。

职责：
    - 管理数据库连接和 schema 迁移
    - 提供最基础的 CRUD 方法（insert / select / update / delete）
    - 不感知 Message 对象，只处理原始数据

使用 aiosqlite 实现非阻塞数据库操作。
"""

import aiosqlite
from pathlib import Path


# ── Schema ──────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant', 'tool', 'summary')),
    content TEXT,
    tool_calls TEXT,
    tool_call_id TEXT,
    tool_name TEXT,
    token_count INTEGER NOT NULL DEFAULT 0,
    is_summary INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, id);

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
"""

# 升级脚本：为旧数据库添加 is_summary 列
MIGRATE_ADD_IS_SUMMARY = """
ALTER TABLE messages ADD COLUMN is_summary INTEGER NOT NULL DEFAULT 0;
"""


class Database:
    """
    纯 SQLite 操作层。

    所有方法都是原始的数据操作，返回 dict/list，
    不包含任何 Message 对象转换逻辑。
    """

    def __init__(self, db_path: str = "pyagent_memory.db"):
        """
        Args:
            db_path: SQLite 数据库文件路径。
                     默认在当前工作目录下创建 pyagent_memory.db。
        """
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def _get_conn(self) -> aiosqlite.Connection:
        """延迟初始化数据库连接（首次使用时自动创建）。"""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.executescript(SCHEMA_SQL)
            # 兼容旧数据库：尝试添加 is_summary 列
            await self._migrate_add_is_summary()
            await self._conn.commit()
        return self._conn

    async def _migrate_add_is_summary(self) -> None:
        """为 v0.2.0 之前的旧数据库添加 is_summary 列。"""
        try:
            await self._conn.executescript(MIGRATE_ADD_IS_SUMMARY)
        except Exception:
            pass  # 列已存在，忽略

    # ── Session CRUD ────────────────────────────────────────────

    async def create_session(self, session_id: str, metadata: dict | None = None) -> None:
        """创建新会话。如果 session_id 已存在则更新 updated_at。"""
        import json
        conn = await self._get_conn()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        await conn.execute(
            """INSERT INTO sessions (session_id, metadata)
               VALUES (?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   updated_at = datetime('now', 'localtime'),
                   metadata = excluded.metadata""",
            (session_id, meta_json),
        )
        await conn.commit()

    async def get_session(self, session_id: str) -> dict | None:
        """获取会话元数据。"""
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    async def list_sessions(self, limit: int = 20) -> list[dict]:
        """列出最近的会话（按 updated_at 降序）。"""
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def update_session_timestamp(self, session_id: str) -> None:
        """更新会话的 updated_at 时间戳。"""
        conn = await self._get_conn()
        await conn.execute(
            "UPDATE sessions SET updated_at = datetime('now', 'localtime') WHERE session_id = ?",
            (session_id,),
        )
        await conn.commit()

    async def delete_session(self, session_id: str) -> None:
        """删除会话及其所有消息（CASCADE）。"""
        conn = await self._get_conn()
        await conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await conn.commit()

    # ── Message CRUD ────────────────────────────────────────────

    async def insert_message(self, row: dict) -> int:
        """
        插入一条消息。

        Args:
            row: 包含 session_id, role, content, tool_calls,
                 tool_call_id, tool_name, token_count 的字典。

        Returns:
            新插入消息的自增 ID。
        """
        conn = await self._get_conn()
        cursor = await conn.execute(
            """INSERT INTO messages (session_id, role, content, tool_calls,
                                     tool_call_id, tool_name, token_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                row["session_id"],
                row["role"],
                row.get("content"),
                row.get("tool_calls"),
                row.get("tool_call_id"),
                row.get("tool_name"),
                row.get("token_count", 0),
            ),
        )
        await conn.commit()
        return cursor.lastrowid

    async def load_messages(
        self, session_id: str, limit: int | None = None
    ) -> list[dict]:
        """
        加载会话的消息历史。

        Args:
            session_id: 会话 ID。
            limit: 最多返回最近 N 条消息。None 表示不限制。

        Returns:
            消息 dict 列表（按 id 升序）。
        """
        conn = await self._get_conn()
        if limit is not None:
            # 先取最后 N 条，再按 id 升序返回（保证时间顺序）
            async with conn.execute(
                """SELECT * FROM (
                       SELECT * FROM messages
                       WHERE session_id = ?
                       ORDER BY id DESC LIMIT ?
                   ) ORDER BY id ASC""",
                (session_id, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def message_count(self, session_id: str) -> int:
        """返回会话的消息总数。"""
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row["cnt"]

    async def delete_session_messages(self, session_id: str) -> None:
        """删除会话的所有消息（保留 session 记录）。"""
        conn = await self._get_conn()
        await conn.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )
        await conn.commit()

    async def insert_summary(self, session_id: str, content: str,
                             token_count: int = 0) -> int:
        """
        插入一条压缩摘要消息。

        Args:
            session_id: 会话 ID。
            content: 摘要文本。
            token_count: 摘要的 Token 估算值。

        Returns:
            新插入消息的自增 ID。
        """
        conn = await self._get_conn()
        cursor = await conn.execute(
            """INSERT INTO messages (session_id, role, content, token_count, is_summary)
               VALUES (?, 'summary', ?, ?, 1)""",
            (session_id, content, token_count),
        )
        await conn.commit()
        return cursor.lastrowid

    async def get_last_summary(self, session_id: str) -> dict | None:
        """
        获取会话的最新摘要消息。

        Returns:
            最新的 summary 消息 dict，如果没有则返回 None。
        """
        conn = await self._get_conn()
        async with conn.execute(
            """SELECT * FROM messages
               WHERE session_id = ? AND is_summary = 1
               ORDER BY id DESC LIMIT 1""",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def delete_messages_in_range(self, session_id: str,
                                       min_id: int, max_id: int) -> int:
        """
        删除指定 ID 范围内的非摘要消息。

        Args:
            session_id: 会话 ID。
            min_id: 最小 ID（含）。
            max_id: 最大 ID（含）。

        Returns:
            删除的行数。
        """
        conn = await self._get_conn()
        cursor = await conn.execute(
            """DELETE FROM messages
               WHERE session_id = ?
                 AND id >= ? AND id <= ?
                 AND is_summary = 0""",
            (session_id, min_id, max_id),
        )
        await conn.commit()
        return cursor.rowcount

    async def delete_message_by_id(self, session_id: str,
                                    message_id: int) -> bool:
        """
        按 ID 精确删除单条消息（包括摘要消息）。

        用于压缩流程中替换旧摘要——delete_messages_in_range
        会过滤 is_summary=1 的行，此方法不做此过滤。

        Args:
            session_id: 会话 ID。
            message_id: 消息 ID。

        Returns:
            True 如果删除了至少一行，否则 False。
        """
        conn = await self._get_conn()
        cursor = await conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND id = ?",
            (session_id, message_id),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def get_total_tokens(self, session_id: str) -> int:
        """返回会话所有消息的 Token 总数。"""
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) as total FROM messages WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row["total"]

    async def load_messages_unlimited(self, session_id: str) -> list[dict]:
        """
        加载会话的所有消息（不限数量），按 id 升序。

        用于滑动窗口计算——需要完整遍历以确定 Token 分配。
        """
        conn = await self._get_conn()
        async with conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── 生命周期 ─────────────────────────────────────────────────

    async def close(self) -> None:
        """关闭数据库连接。"""
        if self._conn:
            await self._conn.close()
            self._conn = None
