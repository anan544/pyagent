"""
FastAPI 服务工厂 — 创建 PyAgent HTTP 服务实例。

提供 create_app() 工厂函数，支持：
    - 配置驱动的 Agent 初始化（启动时自动加载 YAML）
    - Trace ID 中间件（请求全链路追踪）
    - 结构化 JSON 日志
    - 优雅关闭（等待进行中任务完成 + 记忆落盘）

依赖隔离：
    api/ 目录可以 import FastAPI，但 core/tools/memory 模块绝不导入 Web 框架。
    PyAgent 仍可作为纯库（pip install pyagent）使用，无需 Web 依赖。
"""

import logging
from typing import Optional

from fastapi import FastAPI

from .middleware import TraceMiddleware, setup_logging
from .dependencies import get_agent_manager, reset_agent_manager
from .routes import health_router, sessions_router, run_router, tools_router

logger = logging.getLogger("pyagent.harness")


def create_app(
    config_path: Optional[str] = None,
    config_dict: Optional[dict] = None,
    title: str = "PyAgent Harness API",
    version: str = "0.1.0",
) -> FastAPI:
    """
    创建并配置 FastAPI 应用。

    Args:
        config_path: YAML 配置文件路径。None 时自动检测（PYAGENT_ENV）。
        config_dict: 原始配置字典（测试用，优先级高于 config_path）。
        title: API 服务标题。
        version: API 版本号。

    Returns:
        配置完成的 FastAPI 实例，可直接 uvicorn.run(app) 启动。

    Usage:
        # 生产环境
        app = create_app(config_path="config.prod.yaml")

        # 开发环境（自动检测 PYAGENT_ENV）
        app = create_app()

        # 测试环境（字典注入）
        app = create_app(config_dict={"llm": {"model": "gpt-4"}})
    """
    # 确保日志已配置
    setup_logging()

    app = FastAPI(
        title=title,
        version=version,
        description="PyAgent 驾驭工程 HTTP API — 使 PyAgent 可被 Harness 平台纳管调用",
    )

    # ── 中间件（按添加顺序倒序执行）──
    app.add_middleware(TraceMiddleware)

    # ── 生命周期 ──────────────────────────────────

    @app.on_event("startup")
    async def on_startup():
        """服务启动：初始化 Agent 实例。"""
        logger.info("=" * 50)
        logger.info("PyAgent Harness API 启动中...")

        manager = get_agent_manager()
        await manager.initialize(
            config_path=config_path,
            config_dict=config_dict,
        )

        logger.info("PyAgent Harness API 启动完成")

    @app.on_event("shutdown")
    async def on_shutdown():
        """服务关闭：优雅释放资源。"""
        logger.info("PyAgent Harness API 关闭中...")

        manager = get_agent_manager()
        await manager.shutdown()

        logger.info("PyAgent Harness API 已关闭")

    # ── 路由注册 ──────────────────────────────────

    app.include_router(run_router)
    app.include_router(sessions_router)
    app.include_router(health_router)
    app.include_router(tools_router)

    # ── 根路由 ────────────────────────────────────

    @app.get("/", include_in_schema=False)
    async def root():
        """API 根路径 — 返回基本信息。"""
        return {
            "service": "PyAgent Harness API",
            "version": version,
            "docs": "/docs",
            "health": "/health",
        }

    return app


def create_app_from_config(
    config_path: str,
    title: str = "PyAgent Harness API",
) -> FastAPI:
    """
    便捷函数：从 YAML 配置文件创建 FastAPI 应用。

    Args:
        config_path: YAML 配置文件路径（必填）。
        title: API 服务标题。

    Returns:
        FastAPI 实例。
    """
    return create_app(config_path=config_path, title=title)
