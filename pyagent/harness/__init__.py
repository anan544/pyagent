"""
驾驭工程（Harness）— PyAgent 的工程化外壳。

使 PyAgent 可被 Harness 平台纳管调用的完整解决方案：
    - config/ — YAML 配置加载 + Pydantic 校验 + ${ENV_VAR} 解析
    - api/    — FastAPI HTTP 服务 + SSE 流式 + Trace ID + JSON 日志

核心入口：
    # 方式 1：从 YAML 创建 Agent（纯库模式）
    from pyagent.harness import create_agent_from_yaml
    agent = create_agent_from_yaml("config.dev.yaml")

    # 方式 2：启动 HTTP 服务（服务模式）
    import uvicorn
    from pyagent.harness.api import create_app
    app = create_app("config.dev.yaml")
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from typing import Optional

from .config import ConfigLoader, ConfigLoadError, HarnessYamlConfig
from .config import LLMSchema, AgentSchema, MemorySchema, TokenBudgetSchema

from ..core.agent import Agent
from ..core.config import AgentConfig
from ..tools import ToolRegistry, ReadFileTool, WriteFileTool, CodeExecutorTool, SearchTool, TimeLocationTool, DatabaseTool, DebugReplTool
from ..llm import OpenAICompatProvider
from ..memory import MemoryManager, TokenBudget


__all__ = [
    # 工厂函数
    "create_agent_from_yaml",
    "create_agent_from_config",
    # 配置
    "ConfigLoader",
    "ConfigLoadError",
    "HarnessYamlConfig",
    "LLMSchema",
    "AgentSchema",
    "MemorySchema",
    "TokenBudgetSchema",
]


# ── 工具名映射 ──────────────────────────────────────

TOOL_MAP: dict = {
    "read_file": ReadFileTool,
    "write_file": WriteFileTool,
    "execute_python": CodeExecutorTool,
    "search_content": SearchTool,
    "time_location": TimeLocationTool,
    "database_query": DatabaseTool,
    "debug_repl": DebugReplTool,
}


# ── 工厂函数 ────────────────────────────────────────

def create_agent_from_yaml(path: str) -> Agent:
    """
    从 YAML 配置文件创建 Agent 实例。

    内部流程：加载 YAML → 校验 → 构建 LLM/Tools/Memory → 装配 Agent

    Args:
        path: YAML 配置文件路径。

    Returns:
        配置完成的 Agent 实例，可直接调用 agent.run()。

    Raises:
        ConfigLoadError: 配置文件缺失或校验失败。

    Usage:
        agent = create_agent_from_yaml("config.dev.yaml")
        result = await agent.run("审查代码", session_id="session-1")
    """
    config = ConfigLoader.load(path)
    return create_agent_from_config(config)


def create_agent_from_config(config: HarnessYamlConfig) -> Agent:
    """
    从 HarnessYamlConfig 对象创建 Agent 实例。

    Args:
        config: 已校验的 HarnessYamlConfig 实例。

    Returns:
        配置完成的 Agent 实例。
    """
    # 1. LLM Provider
    llm_cfg = config.llm
    llm = OpenAICompatProvider(
        api_key=llm_cfg.api_key,
        base_url=llm_cfg.base_url,
        model=llm_cfg.model,
        max_retries=llm_cfg.max_retries,
        timeout=llm_cfg.timeout,
    )

    # 2. 工具注册表
    registry = ToolRegistry()
    for tool_name in config.agent.tools:
        tool_name = tool_name.strip()
        if tool_name in TOOL_MAP:
            registry.register(TOOL_MAP[tool_name]())
        else:
            import warnings
            warnings.warn(
                f"未知工具 '{tool_name}'，已跳过。"
                f"可用工具: {list(TOOL_MAP.keys())}"
            )

    # 3. 记忆管理器
    memory = MemoryManager(config.memory.db_path)

    # 4. Token 预算
    token_budget = None
    tb = config.memory.token_budget
    if tb.enabled:
        token_budget = TokenBudget(
            model_max_tokens=tb.model_max_tokens,
            safety_factor=tb.safety_factor,
            precision_ratio=tb.precision_ratio,
        )

    # 5. AgentConfig
    agent_config = AgentConfig(
        system_prompt=config.agent.system_prompt,
        max_iterations=config.agent.max_iterations,
        model=config.llm.model,
        verbose=config.agent.verbose,
        token_budget=token_budget,
    )

    # 6. 装配 Agent
    agent = Agent(
        config=agent_config,
        tool_registry=registry,
        llm_provider=llm,
        memory=memory,
    )

    return agent
