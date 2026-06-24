"""
会话管理路由 — GET /sessions/{id}, GET /sessions/{id}/history

会话状态查询依赖现有 MemoryManager 的 session_id 机制，
HTTP 层仅负责透传 session_id，不额外维护会话状态。
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import get_agent_manager
from ..schemas import SessionInfo, SessionHistoryItem, SessionDetail

logger = logging.getLogger("pyagent.harness")

router = APIRouter(tags=["sessions"])


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """
    查询会话基本信息。

    Returns:
        SessionInfo: session_id, created_at, updated_at, message_count, metadata
    """
    manager = get_agent_manager()

    if not manager.memory:
        raise HTTPException(
            status_code=503,
            detail="记忆管理器未初始化，会话功能不可用",
        )

    session = await manager.memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")

    msg_count = await manager.memory.message_count(session_id)

    raw_meta = session.get("metadata", {})
    if isinstance(raw_meta, str):
        import json as _json
        try:
            meta = _json.loads(raw_meta)
        except Exception:
            meta = {}
    else:
        meta = raw_meta or {}

    return SessionInfo(
        session_id=session["session_id"],
        created_at=session.get("created_at", ""),
        updated_at=session.get("updated_at", ""),
        message_count=msg_count,
        metadata=meta,
    )


@router.get("/sessions/{session_id}/history", response_model=SessionDetail)
async def get_session_history(
    session_id: str,
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="返回消息的最大条数",
    ),
):
    """
    查询会话的完整消息历史（含每条消息的 role、content、时间等）。

    用于 Harness 平台的审计或会话回放功能。

    Args:
        session_id: 会话 ID。
        limit: 最多返回的消息条数（1-1000）。
    """
    manager = get_agent_manager()

    if not manager.memory:
        raise HTTPException(
            status_code=503,
            detail="记忆管理器未初始化，会话功能不可用",
        )

    # 验证会话存在
    session = await manager.memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")

    # 直接通过 Database 层获取原始行
    rows = await manager.memory.db.load_messages(session_id, limit=limit)
    total = await manager.memory.message_count(session_id)

    messages = []
    for row in rows:
        if row["role"] == "system":
            continue  # 跳过系统消息
        messages.append(
            SessionHistoryItem(
                id=row["id"],
                role=row["role"],
                content=row.get("content"),
                tool_name=row.get("tool_name"),
                tool_call_id=row.get("tool_call_id"),
                created_at=row.get("created_at", ""),
            )
        )

    raw_meta2 = session.get("metadata", {})
    if isinstance(raw_meta2, str):
        import json as _json2
        try:
            meta2 = _json2.loads(raw_meta2)
        except Exception:
            meta2 = {}
    else:
        meta2 = raw_meta2 or {}

    return SessionDetail(
        session=SessionInfo(
            session_id=session["session_id"],
            created_at=session.get("created_at", ""),
            updated_at=session.get("updated_at", ""),
            message_count=total,
            metadata=meta2,
        ),
        messages=messages,
        total_messages=total,
    )


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(
    limit: int = Query(default=50, ge=1, le=200, description="返回会话的最大数量"),
    workspace: Optional[str] = Query(default=None, description="按工作区路径过滤（可选）"),
):
    """
    列出最近的会话。

    Args:
        limit: 最多返回的会话数（1-200）。
        workspace: 可选，按工作区路径过滤。
    """
    manager = get_agent_manager()

    if not manager.memory:
        raise HTTPException(
            status_code=503,
            detail="记忆管理器未初始化，会话功能不可用",
        )

    sessions = await manager.memory.list_sessions(limit=limit)

    result = []
    for s in sessions:
        raw_meta = s.get("metadata", {})
        # metadata 可能是 JSON 字符串，需要解析
        if isinstance(raw_meta, str):
            import json as _json
            try:
                meta = _json.loads(raw_meta)
            except Exception:
                meta = {}
        else:
            meta = raw_meta or {}
        # 工作区过滤
        if workspace and meta.get("workspace", "") != workspace:
            continue
        msg_count = await manager.memory.message_count(s["session_id"])
        result.append(
            SessionInfo(
                session_id=s["session_id"],
                created_at=s.get("created_at", ""),
                updated_at=s.get("updated_at", ""),
                message_count=msg_count,
                metadata=meta,
            )
        )

    return result
