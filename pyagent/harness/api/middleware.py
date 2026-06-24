"""
中间件：Trace ID 注入 + 结构化 JSON 日志。

- TraceMiddleware：自动提取/生成 trace_id，注入请求状态和响应头
- setup_logging：配置 JSON 格式日志输出，便于 ELK / 日志平台采集
"""

import json
import time
import uuid
import logging
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


# ── 结构化 JSON 日志 ──────────────────────────────────

class JsonFormatter(logging.Formatter):
    """
    JSON 格式日志 Formatter。

    输出格式：
        {"timestamp": "2026-06-19T...", "level": "INFO", "logger": "...",
         "message": "...", "trace_id": "...", "event_type": "..."}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 附加上下文字段（由 TraceLogger 或中间件注入）
        for attr in ("trace_id", "event_type", "session_id", "method", "path",
                     "status_code", "duration_ms"):
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)

        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """配置全局 JSON 日志，返回 harness 专用 logger。"""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("pyagent.harness")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    return logger


logger = setup_logging()


# ── Trace ID 中间件 ───────────────────────────────────

class TraceMiddleware(BaseHTTPMiddleware):
    """
    Trace ID 中间件。

    行为：
        - 从请求头 X-Trace-ID 提取 trace_id，不存在则自动生成
        - 将 trace_id 写入 request.state.trace_id 供下游路由使用
        - 将 trace_id 注入响应头 X-Trace-ID
        - 记录请求开始/结束的结构化日志
    """

    async def dispatch(self, request: Request, call_next):
        # 提取或生成 trace_id
        trace_id = (
            request.headers.get("X-Trace-ID")
            or uuid.uuid4().hex[:16]
        )
        request.state.trace_id = trace_id

        start_time = time.time()

        # 记录请求开始
        logger.info(
            f"→ {request.method} {request.url.path}",
            extra={
                "trace_id": trace_id,
                "event_type": "request_start",
                "method": request.method,
                "path": request.url.path,
            },
        )

        try:
            response: Response = await call_next(request)
        except Exception:
            logger.error(
                f"✗ {request.method} {request.url.path} — 未捕获异常",
                extra={
                    "trace_id": trace_id,
                    "event_type": "request_error",
                },
                exc_info=True,
            )
            raise

        duration_ms = int((time.time() - start_time) * 1000)

        # 注入响应头
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Response-Time-ms"] = str(duration_ms)

        # 记录请求完成
        logger.info(
            f"← {response.status_code} {request.method} {request.url.path} "
            f"({duration_ms}ms)",
            extra={
                "trace_id": trace_id,
                "event_type": "request_end",
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )

        return response
