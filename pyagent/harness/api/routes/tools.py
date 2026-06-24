"""
Agent Tools 路由 — GET /tools

返回当前 Agent 注册的所有工具定义（name + description + parameters JSON Schema）。
用于：
    - VSCode 扩展展示可用工具列表
    - 外部系统了解 Agent 能力
    - 调试工具注册是否正确
"""

import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..dependencies import get_agent_manager
from ..schemas import ErrorResponse

logger = logging.getLogger("pyagent.harness")

router = APIRouter(tags=["tools"])


@router.get("/tools", summary="获取所有可用工具的 JSON Schema")
async def get_tools(request: Request):
    """
    返回当前 Agent 注册的所有工具定义。

    每个工具包含：
        - name:        工具名称（LLM function calling 使用的名称）
        - description: 工具的功能描述和用法指南
        - parameters:  JSON Schema 格式的参数定义
        - risk_level:  工具的风险等级（low / medium / high）

    Returns:
        JSON 数组，每项一个工具定义。
    """
    trace_id = getattr(request.state, "trace_id", "unknown")
    manager = get_agent_manager()

    if not manager.is_initialized:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="Agent 未初始化",
                detail="服务尚未完成启动，请稍后重试",
                trace_id=trace_id,
            ).model_dump(),
        )

    registry = manager.registry
    if registry is None:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="工具注册表未就绪",
                detail="工具注册表尚未初始化",
                trace_id=trace_id,
            ).model_dump(),
        )

    tools_list = []
    for tool in registry.get_all():
        tools_list.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "risk_level": tool.risk_level,
        })

    return JSONResponse(content=tools_list)
