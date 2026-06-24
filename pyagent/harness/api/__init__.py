"""
驾驭工程 — HTTP API 模块。

基于 FastAPI 的轻量 HTTP 服务，将 PyAgent 包装为可纳管的 Web 服务。
与 core/tools/memory 解耦：PyAgent 仍可纯库方式使用。

快速启动：
    import uvicorn
    from pyagent.harness.api import create_app

    app = create_app("config.dev.yaml")
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from .server import create_app, create_app_from_config
from .schemas import (
    RunRequest,
    RunResponse,
    ToolCallInfo,
    TokenUsage,
    SessionInfo,
    SessionDetail,
    SessionHistoryItem,
    HealthResponse,
    ErrorResponse,
)
from .dependencies import AgentManager, get_agent_manager, reset_agent_manager

__all__ = [
    "create_app",
    "create_app_from_config",
    "RunRequest",
    "RunResponse",
    "ToolCallInfo",
    "TokenUsage",
    "SessionInfo",
    "SessionDetail",
    "SessionHistoryItem",
    "HealthResponse",
    "ErrorResponse",
    "AgentManager",
    "get_agent_manager",
    "reset_agent_manager",
]
