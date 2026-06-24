"""
API 请求/响应 Schema — 对齐 Harness 平台统一消息规范。

请求：session_id + user_input + context
响应：reply + tool_calls + token_usage + trace_id
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ── 请求 ──────────────────────────────────────────────

class RunRequest(BaseModel):
    """POST /run 请求体。"""

    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID。提供时可继续之前对话；不提供则自动创建新会话。",
    )
    user_input: str = Field(
        ...,
        min_length=1,
        description="用户输入的问题或任务描述",
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="可选附加上下文（如文件路径、项目元数据等）",
    )


# ── 响应 ──────────────────────────────────────────────

class ToolCallInfo(BaseModel):
    """单次工具调用的记录。"""

    id: str = Field(..., description="工具调用 ID")
    name: str = Field(..., description="工具名称")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="调用参数",
    )


class TokenUsage(BaseModel):
    """Token 使用统计。"""

    prompt_tokens: int = Field(default=0, description="输入消耗的 Token 数")
    completion_tokens: int = Field(default=0, description="输出消耗的 Token 数")
    total_tokens: int = Field(default=0, description="总 Token 消耗")


class RunResponse(BaseModel):
    """POST /run 响应体。"""

    reply: str = Field(..., description="Agent 的最终文本回复")
    tool_calls: List[ToolCallInfo] = Field(
        default_factory=list,
        description="本轮执行中调用过的工具列表",
    )
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage,
        description="Token 使用统计（估算值）",
    )
    trace_id: str = Field(..., description="链路追踪 ID，贯穿整个 ReAct 循环")
    session_id: Optional[str] = Field(
        default=None,
        description="当前会话 ID，后续请求可传入以继续对话",
    )


class SessionInfo(BaseModel):
    """会话基本信息。"""

    session_id: str
    created_at: str
    updated_at: str
    message_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionHistoryItem(BaseModel):
    """会话历史中的单条消息。"""

    id: int
    role: str
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    created_at: str


class SessionDetail(BaseModel):
    """会话详细信息（含历史消息）。"""

    session: SessionInfo
    messages: List[SessionHistoryItem] = Field(default_factory=list)
    total_messages: int = 0


class HealthResponse(BaseModel):
    """GET /health 响应体。"""

    status: str = Field(
        ...,
        description='"healthy" | "degraded" | "unhealthy"',
    )
    version: str = Field(..., description="PyAgent 版本号")
    llm_connected: bool = Field(
        ...,
        description="LLM 服务是否可达",
    )
    db_writable: bool = Field(
        ...,
        description="数据库是否可写",
    )
    tools_available: List[str] = Field(
        default_factory=list,
        description="当前注册的工具名称列表",
    )


class ErrorResponse(BaseModel):
    """通用错误响应。"""

    error: str = Field(..., description="错误类型")
    detail: Optional[str] = Field(default=None, description="错误详情")
    trace_id: Optional[str] = Field(default=None, description="链路追踪 ID")
