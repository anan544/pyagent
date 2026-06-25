"""
核心运行路由 — POST /run

支持两种模式：
    - 非流式（默认）：等待 agent.run() 完成后返回 RunResponse
    - 流式（?stream=true）：SSE 事件流，实时推送 Agent 的思考和工具调用步骤

v2.1: 动态注入时间和位置上下文到 Agent system prompt。
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, AsyncGenerator

from fastapi import APIRouter, Request, Query, HTTPException
from fastapi.responses import StreamingResponse

from ..dependencies import get_agent_manager
from ..schemas import RunRequest, RunResponse, ToolCallInfo, TokenUsage

logger = logging.getLogger("pyagent.harness")

router = APIRouter(tags=["run"])


@router.post("/run", response_model=RunResponse)
async def run_agent(
    run_request: RunRequest,
    request: Request,
    stream: bool = Query(default=False, description="是否启用 SSE 流式输出"),
):
    """
    执行 Agent 对话。

    支持流式（SSE）和非流式两种模式：
        - 默认：POST /run → 等待完成，返回 RunResponse
        - 流式：POST /run?stream=true → SSE 事件流

    Args:
        run_request: 包含 session_id、user_input、context 的请求体。
        stream: 是否启用 SSE 流式输出。

    Returns:
        非流式：RunResponse JSON
        流式：   SSE 事件流（text/event-stream）
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    manager = get_agent_manager()

    if not manager.is_initialized:
        from ..schemas import ErrorResponse
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="Agent 未初始化",
                detail="服务尚未完成启动，请稍后重试",
                trace_id=trace_id,
            ).model_dump(),
        )

    agent = manager.agent
    session_id = run_request.session_id

    # 自动创建会话（存储 workspace 到 metadata）
    if not session_id and manager.memory:
        meta = {}
        ws_path = (run_request.context or {}).get("workspace", "")
        if ws_path:
            meta["workspace"] = ws_path
            meta["title"] = run_request.user_input[:80]
        session_id = await manager.memory.create_session(metadata=meta if meta else None)
        logger.info("自动创建会话: %s (workspace=%s)", session_id, ws_path or "—")

    # ★ v2.0: 将工作区路径注入 Agent，使工具在用户项目目录执行
    ws_path = (run_request.context or {}).get("workspace", "")
    if ws_path:
        agent.workspace = ws_path
    else:
        agent.workspace = None

    # ★ v2.1: 构建动态上下文（时间 + 客户端位置）
    dynamic_ctx = _build_dynamic_context(request, run_request)
    agent.dynamic_context = dynamic_ctx

    # ★ v2.5: 远程工具执行模式
    remote_tools = run_request.context.get("remote_tools", False) if run_request.context else False
    if remote_tools:
        agent.enable_remote_tools(True)
        logger.info("Tool Relay 模式已启用 — 工具将发给客户端执行")

    # ── 流式模式 ──────────────────────────────────
    if stream:
        return StreamingResponse(
            _stream_run(agent, run_request, session_id, trace_id, manager),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── 非流式模式 ────────────────────────────────
    # 执行 Agent
    reply = await agent.run(
        run_request.user_input,
        session_id=session_id,
    )

    # 收集工具调用（Agent 内部自动记录）
    tool_calls = [
        ToolCallInfo(
            id=tc.get("id", f"call_{i}"),
            name=tc.get("name", "unknown"),
            arguments=tc.get("arguments", {}),
        )
        for i, tc in enumerate(agent.captured_tool_calls)
    ]

    # Token 使用估算
    token_usage = TokenUsage()

    return RunResponse(
        reply=reply,
        tool_calls=tool_calls,
        token_usage=token_usage,
        trace_id=trace_id,
        session_id=session_id,
    )


@router.post("/run/{session_id}/tool-result")
async def tool_result(
    session_id: str,
    tool_result: dict,
    request: Request,
):
    """
    ★ v2.5: Tool Relay — 客户端执行完工具后回传结果。

    Body:
        {"call_id": "call_1", "result": "工具输出内容..."}

    此端点会唤醒暂停的 Agent 循环，使其继续推理。
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    manager = get_agent_manager()
    agent = manager.agent

    if agent is None:
        raise HTTPException(status_code=503, detail="Agent 未初始化")

    call_id = tool_result.get("call_id", "")
    result_content = tool_result.get("result", "")

    try:
        agent.inject_tool_result(call_id, result_content)
        logger.info(
            "[TOOL-RELAY] 注入结果: call_id=%s, 长度=%d",
            call_id, len(result_content),
        )
        return {"status": "ok", "call_id": call_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── SSE 流式生成器 ─────────────────────────────────

async def _stream_run(
    agent,
    run_request: RunRequest,
    session_id: Optional[str],
    trace_id: str,
    manager,
) -> AsyncGenerator[str, None]:
    """
    SSE 流式执行 Agent — 生成 text/event-stream 格式的数据。

    事件类型：
        - start:    开始执行
        - tool:     工具调用
        - progress: 中间日志
        - done:     完成（含最终回复）
        - error:    出错
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    # 自定义 logger：将日志消息推送到 SSE 队列 + 写文件
    class SSELogger:
        def __call__(self, msg: str):
            logger.info(msg)  # ★ 同时写日志文件，方便排查
            try:
                queue.put_nowait(msg)
            except asyncio.QueueFull:
                pass  # 队列满时丢弃

    # Monkey-patch agent 的 logger 以捕获中间步骤
    original_log = agent.log
    agent.log = SSELogger()

    def _sse(event: str, data: str) -> str:
        """格式化一条 SSE 消息。"""
        return f"event: {event}\ndata: {data}\n\n"

    try:
        try:
            # 发送 start 事件
            yield _sse("start", json.dumps({
                "trace_id": trace_id,
                "session_id": session_id,
                "user_input": run_request.user_input[:200],
            }, ensure_ascii=False))

            # 后台运行 Agent
            run_task = asyncio.create_task(
                agent.run(run_request.user_input, session_id=session_id)
            )
            if manager:
                manager.track_task(run_task)

            sent_tc_count = 0
            sent_thought: str = ""

            # 消费日志队列 → SSE 事件
            while not run_task.done():
                # ★ v2.2: 发送思考内容
                current_thought = getattr(agent, 'current_thought', '') or ''
                if current_thought and current_thought != sent_thought:
                    yield _sse("thought", json.dumps({
                        "content": current_thought,
                    }, ensure_ascii=False))
                    sent_thought = current_thought

                # ★ v2.5: 检测远程工具请求 — 暂停 SSE 流等待客户端
                pending = getattr(agent, 'pending_tool', None)
                if pending is not None:
                    yield _sse("tool_request", json.dumps({
                        "id": pending["id"],
                        "name": pending["name"],
                        "arguments": pending["arguments"],
                    }, ensure_ascii=False))
                    # 找到对应的 Future 并等待
                    _pt = agent._pending_tool
                    if _pt is not None:
                        _future = _pt[1]
                        try:
                            await _future  # 暂停，等待 API 端点注入结果
                        except Exception:
                            pass  # 错误由 agent 端处理
                    continue

                # ★ 轮询新的工具调用（结构化数据，发送给前端）
                current_tc = agent.captured_tool_calls
                while sent_tc_count < len(current_tc):
                    tc = current_tc[sent_tc_count]
                    yield _sse("tool_call", json.dumps({
                        "id": tc.get("id", ""),
                        "name": tc.get("name", "unknown"),
                        "arguments": tc.get("arguments", {}),
                    }, ensure_ascii=False))
                    sent_tc_count += 1

                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=0.1)
                    event_type = "tool" if "->" in msg else "progress"
                    yield _sse(event_type, json.dumps(
                        {"message": msg}, ensure_ascii=False
                    ))
                except asyncio.TimeoutError:
                    continue

            # ★ Agent 完成后，发送未发送的思考内容和工具调用
            # 注意：快速完成的请求可能让 while 循环来不及发送 thought
            current_thought = getattr(agent, 'current_thought', '') or ''
            if current_thought and current_thought != sent_thought:
                yield _sse("thought", json.dumps({
                    "content": current_thought,
                }, ensure_ascii=False))
                sent_thought = current_thought

            current_tc = agent.captured_tool_calls
            while sent_tc_count < len(current_tc):
                tc = current_tc[sent_tc_count]
                yield _sse("tool_call", json.dumps({
                    "id": tc.get("id", ""),
                    "name": tc.get("name", "unknown"),
                    "arguments": tc.get("arguments", {}),
                }, ensure_ascii=False))
                sent_tc_count += 1

            # 等待 Agent 完成
            try:
                reply = await run_task
                yield _sse("done", json.dumps({
                    "reply": reply,
                    "trace_id": trace_id,
                    "session_id": session_id,
                }, ensure_ascii=False))
            except asyncio.CancelledError:
                yield _sse("error", json.dumps({
                    "error": "请求被取消（客户端断开或服务器超时）",
                    "trace_id": trace_id,
                }, ensure_ascii=False))
            except Exception as e:
                yield _sse("error", json.dumps({
                    "error": str(e),
                    "trace_id": trace_id,
                }, ensure_ascii=False))

        except asyncio.CancelledError:
            # ★ 整个 SSE 流被取消（客户端断开），静默退出不发送额外事件
            # 因为连接已断开，发送事件也没人接收
            logger.info("[SSE] 流被取消 (trace_id=%s)", trace_id)
        except Exception as e:
            # ★ 兜底：确保即使轮询循环崩溃也能发送错误事件
            logger.exception("[SSE] 流异常 (trace_id=%s): %s", trace_id, e)
            try:
                yield _sse("error", json.dumps({
                    "error": f"服务内部错误: {e}",
                    "trace_id": trace_id,
                }, ensure_ascii=False))
            except Exception:
                pass  # 连接已断开，忽略

    finally:
        agent.log = original_log


# ═══════════════════════════════════════════════════════════
# ★ v2.1: 动态上下文构建器
# ═══════════════════════════════════════════════════════════

# Accept-Language → 地理位置映射
_LANG_LOCATION_MAP: dict[str, str] = {
    "zh-CN": "中国，北京",
    "zh-TW": "中国，台北",
    "zh-HK": "中国，香港",
    "zh-SG": "新加坡",
    "ja": "日本，东京",
    "ko": "韩国，首尔",
    "en-US": "美国，纽约",
    "en-GB": "英国，伦敦",
    "en-CA": "加拿大，多伦多",
    "en-AU": "澳大利亚，悉尼",
    "de": "德国，柏林",
    "fr": "法国，巴黎",
    "es": "西班牙，马德里",
    "pt": "巴西，圣保罗",
    "ru": "俄罗斯，莫斯科",
    "th": "泰国，曼谷",
    "vi": "越南，河内",
}

# Accept-Language → 时区偏移（小时）
_LANG_TIMEZONE_MAP: dict[str, float] = {
    "zh-CN": 8, "zh-TW": 8, "zh-HK": 8, "zh-SG": 8,
    "ja": 9, "ko": 9,
    "en-US": -5, "en-GB": 0, "en-CA": -5, "en-AU": 10,
    "de": 1, "fr": 1, "es": 1,
    "pt": -3, "ru": 3,
    "th": 7, "vi": 7,
}


def _build_dynamic_context(request: Request, run_request: RunRequest) -> str:
    """从请求中提取时间和客户端位置信息，构建 <time_location> 上下文块。

    优先级：
        1. VSCode 客户端通过 context.user_tz 显式传递时区
        2. HTTP Accept-Language 头推断位置和时区
        3. 使用服务器本地时间（fallback）
    """
    # ── 1. 时间 ──
    user_tz = (run_request.context or {}).get("user_tz", "")
    tz_offset = (run_request.context or {}).get("tz_offset")

    if tz_offset is not None:
        try:
            offset_hours = float(tz_offset)
        except (ValueError, TypeError):
            offset_hours = None
    else:
        offset_hours = None

    # ── 2. 位置推断 ──
    accept_lang = request.headers.get("Accept-Language", "")
    location = ""
    if offset_hours is None:
        # 从 Accept-Language 推断
        for lang_key in _LANG_LOCATION_MAP:
            if accept_lang.startswith(lang_key):
                location = _LANG_LOCATION_MAP[lang_key]
                offset_hours = _LANG_TIMEZONE_MAP.get(lang_key, 8)
                break

    if not location:
        location = user_tz or "未知"
    if offset_hours is None:
        offset_hours = 8  # 默认 UTC+8

    # ── 3. 计算用户本地时间 ──
    now_utc = datetime.now(timezone.utc)
    user_now = now_utc + timedelta(hours=offset_hours)

    # ── 4. 构建上下文块 ──
    ctx = (
        "<time_location>\n"
        f"  当前时间：{user_now.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(UTC{'+' if offset_hours >= 0 else ''}{offset_hours:.0f})\n"
        f"  当前地点：{location}\n"
        f"  星期：{['星期一','星期二','星期三','星期四','星期五','星期六','星期日'][user_now.weekday()]}\n"
        "</time_location>"
    )
    return ctx
