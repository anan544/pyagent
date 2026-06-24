"""
依赖注入模块 — Agent 实例管理与生命周期。

在服务启动时初始化全局 Agent 实例，所有路由通过 get_agent_manager()
获取共享的 AgentManager，避免每次请求重复创建 Agent 的开销。

设计：
    - 单实例模式（后续可扩展为实例池）
    - 优雅关闭：注册 SIGTERM 处理器，等待进行中的 agent.run() 完成
    - 健康检查：验证 LLM 连通性和数据库可写性
"""

import asyncio
import logging
import signal
from typing import Optional

from ..config.loader import ConfigLoader
from ..config.schema import HarnessYamlConfig
from ...core.agent import Agent
from ...core.config import AgentConfig
from ...tools import (
    ToolRegistry,
    ReadFileTool,
    WriteFileTool,
    CodeExecutorTool,
    SearchTool,
    CommandExecutorTool,
    TimeLocationTool,
    DatabaseTool,
    DebugReplTool,
)
from ...tools.sandbox import SandboxTool
from ...mcp import MCPClient, mcp_tools_to_pyagent_tools
from ...llm import OpenAICompatProvider
from ...memory import MemoryManager, TokenBudget


logger = logging.getLogger("pyagent.harness")

# ── 工具名 → 工具类映射 ──────────────────────────────

TOOL_MAP: dict = {
    "read_file": ReadFileTool,
    "write_file": WriteFileTool,
    "execute_python": CodeExecutorTool,  # 默认本地执行；可被沙箱/MCP覆盖
    "execute_command": CommandExecutorTool,
    "search_content": SearchTool,
    "spawn_subagent": None,  # 特殊：需要 llm_provider + registry，初始化时动态创建
    "time_location": TimeLocationTool,
    "database_query": DatabaseTool,
    "debug_repl": DebugReplTool,
}


class AgentManager:
    """
    Agent 实例管理器。

    管理 Agent 及其依赖（LLM Provider、ToolRegistry、MemoryManager）
    的完整生命周期。

    使用方式：
        manager = AgentManager()
        await manager.initialize("config.dev.yaml")
        result = await manager.agent.run("任务描述", session_id="s1")
        await manager.shutdown()
    """

    def __init__(self):
        self._agent: Optional[Agent] = None
        self._config: Optional[HarnessYamlConfig] = None
        self._memory: Optional[MemoryManager] = None
        self._registry: Optional[ToolRegistry] = None
        self._mcp_clients: list = []
        self._running_tasks: set = set()  # 跟踪进行中的 agent.run() 任务

    # ── 属性 ────────────────────────────────────────

    @property
    def agent(self) -> Optional[Agent]:
        """当前 Agent 实例。"""
        return self._agent

    @property
    def config(self) -> Optional[HarnessYamlConfig]:
        """当前加载的配置。"""
        return self._config

    @property
    def memory(self) -> Optional[MemoryManager]:
        """当前 MemoryManager 实例。"""
        return self._memory

    @property
    def registry(self) -> Optional[ToolRegistry]:
        """当前 ToolRegistry 实例。"""
        return self._registry

    @property
    def is_initialized(self) -> bool:
        """是否已完成初始化。"""
        return self._agent is not None

    # ── 生命周期 ────────────────────────────────────

    async def initialize(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[dict] = None,
    ):
        """
        初始化 Agent 实例。

        配置来源优先级：config_dict > config_path > 自动检测

        Args:
            config_path: YAML 配置文件路径。None 时自动检测。
            config_dict: 原始配置字典（测试用）。
        """
        # 1. 加载配置
        if config_dict is not None:
            self._config = ConfigLoader.from_dict(config_dict)
        elif config_path is not None:
            self._config = ConfigLoader.load(config_path)
        else:
            self._config = ConfigLoader.load_by_env()

        logger.info(
            "配置加载完成 | model=%s | tools=%s | max_iterations=%s",
            self._config.llm.model,
            self._config.agent.tools,
            self._config.agent.max_iterations,
        )

        # 2. 创建 LLM Provider
        llm_cfg = self._config.llm
        llm = OpenAICompatProvider(
            api_key=llm_cfg.api_key,
            base_url=llm_cfg.base_url,
            model=llm_cfg.model,
            max_retries=llm_cfg.max_retries,
            timeout=llm_cfg.timeout,
        )
        logger.info("LLM Provider 创建完成: %s @ %s", llm_cfg.model, llm_cfg.base_url)

        # 3. 创建工具注册表
        self._registry = ToolRegistry()

        # ── MCP 客户端初始化（优先级最高） ──
        mcp_clients: list = []
        mcp_cfg = getattr(self._config, "mcp", None)
        use_mcp = mcp_cfg is not None and getattr(mcp_cfg, "enabled", False)
        mcp_tool_names: set[str] = set()

        if use_mcp:
            for server_cfg in mcp_cfg.servers:
                if not getattr(server_cfg, "enabled", True):
                    continue
                cmd_parts = server_cfg.command.split()
                client = MCPClient(
                    name=server_cfg.name,
                    command=cmd_parts,
                    description=getattr(server_cfg, "description", ""),
                )
                try:
                    await client.connect()
                    mcp_tools = await mcp_tools_to_pyagent_tools(client)
                    for tool in mcp_tools:
                        self._registry.register(tool)
                        mcp_tool_names.add(tool.name)
                        logger.info(
                            "注册 MCP 工具: %s (来自 %s)",
                            tool.name, server_cfg.name,
                        )
                    mcp_clients.append(client)
                except Exception as e:
                    logger.warning("MCP 服务器 '%s' 连接失败: %s", server_cfg.name, e)
        self._mcp_clients = mcp_clients

        # ── 原生工具注册（跳过已被 MCP 覆盖的工具） ──
        governance = None  # ★ 提前声明，spawn_subagent 注册时需要引用
        sandbox_cfg = getattr(self._config, "sandbox", None)
        use_sandbox = (
            sandbox_cfg is not None
            and getattr(sandbox_cfg, "enabled", False)
            and not use_mcp  # MCP 优先级高于 SSH sandbox
        )
        for tool_name in self._config.agent.tools:
            tool_name = tool_name.strip()
            if tool_name in mcp_tool_names:
                continue  # 已被 MCP 工具覆盖
            if tool_name == "execute_python" and use_sandbox:
                tool = SandboxTool(
                    vm_host=getattr(sandbox_cfg, "vm_host", "192.168.100.130"),
                    vm_user=getattr(sandbox_cfg, "vm_user", "ananan"),
                )
                self._registry.register(tool)
                logger.info("注册工具: execute_python (CubeSandbox 微 VM)")
            elif tool_name == "spawn_subagent":
                from ...tools.orchestration import AgentTool
                tool = AgentTool(
                    llm_provider=llm,
                    tool_registry=self._registry,
                    memory=self._memory,
                    governance=governance,
                )
                self._registry.register(tool)
                logger.info("注册工具: spawn_subagent (子 Agent 派发)")
            elif tool_name in TOOL_MAP and TOOL_MAP[tool_name] is not None:
                self._registry.register(TOOL_MAP[tool_name]())
                logger.info("注册工具: %s", tool_name)
            else:
                logger.warning("未知工具 '%s'，已跳过。可用工具: %s",
                               tool_name, list(TOOL_MAP.keys()))

        # 4. v0.10.0: 创建安全治理栈
        governance = None
        sec_cfg = self._config.security
        if sec_cfg.enabled:
            from ..context.session_risk_context import SessionRiskContext
            from ..context.security_circuit_breaker import SecurityCircuitBreaker
            from ..context.parameter_validator import ParameterWhitelistValidator
            from ..context.combo_rules import ComboRule, ComboRuleEngine
            from ..context.security_governance import SecurityGovernance
            from ..context.governance_wrapper import GovernanceWrapper

            session_risk = SessionRiskContext(
                window_seconds=sec_cfg.combo_detection.sliding_window_seconds,
                max_records=sec_cfg.combo_detection.max_records,
            )
            sec_cb = SecurityCircuitBreaker(
                max_blocks=sec_cfg.circuit_breaker.max_blocks,
                window_seconds=sec_cfg.circuit_breaker.window_seconds,
                cooldown_seconds=sec_cfg.circuit_breaker.cooldown_seconds,
                backoff_factor=sec_cfg.circuit_breaker.backoff_factor,
                max_backoff_seconds=sec_cfg.circuit_breaker.max_backoff_seconds,
            )
            param_validator = ParameterWhitelistValidator(
                allowed_command_prefixes=sec_cfg.param_whitelist.allowed_command_prefixes or None,
                blocked_command_patterns=sec_cfg.param_whitelist.blocked_command_patterns or None,
                domain_allowlist=sec_cfg.param_whitelist.domain_allowlist or None,
                blocked_ip_patterns=sec_cfg.param_whitelist.blocked_ip_patterns or None,
            )
            # 字段级精炼规则
            combo_engine = None
            if sec_cfg.combo_rules:
                rules = [
                    ComboRule(
                        name=r.name,
                        sequence=r.sequence,
                        match_on=r.match_on,
                        within_window=r.within_window,
                    )
                    for r in sec_cfg.combo_rules
                ]
                combo_engine = ComboRuleEngine(rules)
                logger.info("ComboRuleEngine 已加载 %d 条精炼规则", len(rules))

            gov = SecurityGovernance(
                phase_restrictions_enabled=sec_cfg.phase_restrictions.enabled,
                blocked_in_readonly_phases=sec_cfg.phase_restrictions.blocked_in_readonly_phases,
                combo_detection_enabled=sec_cfg.combo_detection.enabled,
                param_whitelist_enabled=sec_cfg.param_whitelist.enabled,
                session_risk=session_risk,
                param_validator=param_validator,
                combo_rule_engine=combo_engine,
                degrade_threshold_ms=sec_cfg.performance.degrade_threshold_ms,
            )
            governance = GovernanceWrapper(
                governance=gov,
                circuit_breaker=sec_cb,
                enable_perf_monitor=sec_cfg.performance.enabled,
            )
            logger.info(
                "安全治理层已启用: phase=%s combo=%s param=%s cb=%s perf=%.1fms",
                sec_cfg.phase_restrictions.enabled,
                sec_cfg.combo_detection.enabled,
                sec_cfg.param_whitelist.enabled,
                sec_cfg.circuit_breaker.enabled,
                sec_cfg.performance.degrade_threshold_ms,
            )
        else:
            logger.info("安全治理层已禁用 (security.enabled = false)")

        self._governance = governance

        # 5. 创建记忆管理器
        self._memory = MemoryManager(self._config.memory.db_path)
        logger.info("MemoryManager 创建完成: %s", self._config.memory.db_path)

        # 6. Token 预算
        token_budget = None
        tb = self._config.memory.token_budget
        if tb.enabled:
            token_budget = TokenBudget(
                model_max_tokens=tb.model_max_tokens,
                safety_factor=tb.safety_factor,
                precision_ratio=tb.precision_ratio,
            )
            logger.info(
                "TokenBudget 启用: max=%s safety=%.1f precision=%.1f",
                tb.model_max_tokens, tb.safety_factor, tb.precision_ratio,
            )

        # 7. 创建 AgentConfig
        agent_cfg = self._config.agent
        agent_config = AgentConfig(
            system_prompt=agent_cfg.system_prompt,
            max_iterations=agent_cfg.max_iterations,
            model=self._config.llm.model,
            verbose=agent_cfg.verbose,
            token_budget=token_budget,
            context_files=list(agent_cfg.context_files or []),
            rules_dir=getattr(agent_cfg, 'rules_dir', '') or '',
        )

        # 8. 创建 Agent
        self._agent = Agent(
            config=agent_config,
            tool_registry=self._registry,
            llm_provider=llm,
            memory=self._memory,
            governance=governance,
        )

        logger.info("Agent 实例初始化完成")

    async def shutdown(self):
        """
        优雅关闭 — 等待进行中的任务完成，释放资源。

        注册为 SIGTERM/SIGINT 的信号处理器。
        """
        logger.info("收到关闭信号，等待进行中的任务完成...")

        # 等待所有进行中的 agent.run() 完成
        if self._running_tasks:
            logger.info("等待 %d 个进行中任务...", len(self._running_tasks))
            await asyncio.gather(*self._running_tasks, return_exceptions=True)

        # 关闭 MCP 连接
        for client in self._mcp_clients:
            await client.disconnect()
        self._mcp_clients.clear()

        # 关闭记忆管理器（确保数据落盘）
        if self._memory:
            await self._memory.close()
            logger.info("MemoryManager 已关闭")

        logger.info("Agent 已关闭")

    # ── 任务跟踪 ────────────────────────────────────

    def track_task(self, task: asyncio.Task):
        """跟踪异步任务，用于优雅关闭时等待完成。"""
        self._running_tasks.add(task)
        task.add_done_callback(self._running_tasks.discard)

    # ── 健康检查 ────────────────────────────────────

    async def health_check(self) -> dict:
        """
        执行健康检查。

        Returns:
            {
                "llm_connected": bool,   # LLM 服务是否可达
                "db_writable": bool,     # 数据库是否可写
            }
        """
        result = {"llm_connected": False, "db_writable": False}

        if not self._agent:
            return result

        # 检查 LLM 连通性（发送最小 ping 请求）
        try:
            from ...core.message import SystemMessage, UserMessage
            response = await self._agent.llm.generate(
                [SystemMessage(content="ping"), UserMessage(content="pong")],
                tools=[],
            )
            result["llm_connected"] = response is not None
        except Exception as e:
            logger.warning("LLM 健康检查失败: %s", e)

        # 检查数据库可写性
        try:
            test_sid = await self._memory.create_session(
                "_health_check",
                metadata={"purpose": "health_check"},
            )
            await self._memory.delete_session(test_sid)
            result["db_writable"] = True
        except Exception as e:
            logger.warning("数据库健康检查失败: %s", e)

        return result


# ── 全局单例 ──────────────────────────────────────────

_agent_manager: Optional[AgentManager] = None
_shutdown_registered: bool = False


def get_agent_manager() -> AgentManager:
    """
    获取全局 AgentManager 单例。

    首次调用时自动创建实例并注册优雅关闭信号处理器。
    """
    global _agent_manager, _shutdown_registered

    if _agent_manager is None:
        _agent_manager = AgentManager()

        # 注册信号处理器（仅一次）
        if not _shutdown_registered:
            _shutdown_registered = True
            _register_signal_handlers(_agent_manager)

    return _agent_manager


def reset_agent_manager():
    """重置全局 AgentManager（测试用）。"""
    global _agent_manager, _shutdown_registered
    _agent_manager = None
    _shutdown_registered = False


def _register_signal_handlers(manager: AgentManager):
    """注册 SIGTERM / SIGINT 处理器。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # 不在事件循环中，跳过

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(manager.shutdown()),
            )
        except (ValueError, NotImplementedError):
            # Windows 不支持 add_signal_handler，降级处理
            pass
