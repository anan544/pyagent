"""
健康检查路由 — GET /health

验证内容：
    1. LLM 连通性（发送 ping 请求）
    2. 数据库可写性（创建/删除临时会话）
    3. 进程存活（HTTP 层自动保证）

返回状态：healthy | degraded | unhealthy
"""

import logging
from fastapi import APIRouter

from ..dependencies import get_agent_manager
from ..schemas import HealthResponse

logger = logging.getLogger("pyagent.harness")

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查端点。

    检查项：
        - LLM 服务连通性（发送最小 ping 请求）
        - SQLite 数据库可写性（临时写入测试）
        - 当前注册的工具列表

    Returns:
        HealthResponse:
            status: "healthy"（全正常）| "degraded"（部分异常）| "unhealthy"（不可用）
            llm_connected: LLM 是否可达
            db_writable: 数据库是否可写
            tools_available: 已注册工具名列表
    """
    manager = get_agent_manager()
    from .... import __version__

    # Agent 未初始化
    if not manager.is_initialized:
        return HealthResponse(
            status="unhealthy",
            version=__version__,
            llm_connected=False,
            db_writable=False,
            tools_available=[],
        )

    # 执行健康检查
    check = await manager.health_check()

    # 确定整体状态
    if check["llm_connected"] and check["db_writable"]:
        status = "healthy"
    elif check["llm_connected"] or check["db_writable"]:
        status = "degraded"
    else:
        status = "unhealthy"

    # 工具列表
    tools_available = (
        manager.registry.list_names() if manager.registry else []
    )

    return HealthResponse(
        status=status,
        version=__version__,
        llm_connected=check["llm_connected"],
        db_writable=check["db_writable"],
        tools_available=tools_available,
    )
