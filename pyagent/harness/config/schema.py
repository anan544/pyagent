"""
YAML 配置 Schema — 使用 Pydantic 进行类型校验和必填项检查。

四层结构：llm / agent / memory / security
所有字段均有默认值，缺失时不报错；类型错误时给出清晰提示。

v0.10.0 新增：
    - security 配置块：安全治理层（前置门控 + 熔断审计）
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class LLMSchema(BaseModel):
    """LLM 连接配置 — 模型名、API Key、温度等。"""

    provider: str = Field(
        default="openai_compat",
        description="LLM Provider 类型，当前仅支持 openai_compat",
    )
    model: str = Field(
        default="deepseek-chat",
        description="模型名称，如 deepseek-chat、gpt-4o",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API Key。留空时自动从环境变量 DEEPSEEK_API_KEY / OPENAI_API_KEY 检测",
    )
    base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="API 基础 URL",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="生成温度，代码场景建议 0.0",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="单次生成最大 Token 数，None 表示不限制",
    )
    timeout: float = Field(
        default=120.0,
        gt=0,
        description="HTTP 请求超时秒数",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="失败重试次数",
    )


class AgentSchema(BaseModel):
    """Agent 行为配置 — system_prompt、max_iterations、工具列表。"""

    system_prompt: str = Field(
        default=(
            "You are an AI coding engineer operating the **PyAgent v1.1.0** framework. "
            "You are an interactive tool that helps users with software engineering tasks.\n\n"
            "## 🛠️ TOOL USAGE STRATEGY\n"
            "1. **Starting Servers / Running Commands**: ALWAYS use `execute_command` (NOT `execute_python`) for: starting dev servers, installing packages, running shell commands. NEVER use `execute_python` for subprocess/os.system/os.popen — these are BLOCKED.\n"
            "2. **Code Execution**: Use `execute_python` ONLY for: data calculations, file I/O, unit tests, code logic verification.\n"
            "3. **Error Recovery (CRITICAL)**: If a tool call is BLOCKED, DO NOT retry the same tool. IMMEDIATELY switch to a different tool. Example: `execute_python` blocked → switch to `execute_command`.\n"
            "4. **File Editing**: Use `read_file` first for context, then `write_file`. Use `search_content` to find patterns. For complex multi-file changes, use `spawn_subagent`."
        ),
        description="系统提示词 — 定义 Agent 的角色和行为规则",
    )
    max_iterations: int = Field(
        default=20,
        ge=1,
        le=100,
        description="ReAct 循环最大轮次，防止死循环",
    )
    verbose: bool = Field(
        default=False,
        description="是否输出详细日志",
    )
    context_files: List[str] = Field(
        default_factory=list,
        description="项目级行为规范文件列表。支持 agent.md、.cursorrules 等。"
                    "相对路径相对于配置文件所在目录，未指定时自动查找项目根目录 agent.md。",
    )
    rules_dir: str = Field(
        default="",
        description="规则目录路径。设置后自动加载该目录下所有 .md 文件，"
                    "拼接到 context_files 之后。相对路径相对于配置文件所在目录。",
    )
    tools: List[str] = Field(
        default_factory=lambda: [
            "read_file",
            "write_file",
            "execute_python",
            "search_content",
        ],
        description="启用的工具名称列表",
    )


class TokenBudgetSchema(BaseModel):
    """Token 预算配置 — 控制滑动窗口行为。"""

    enabled: bool = Field(
        default=True,
        description="是否启用 Token 预算管理",
    )
    model_max_tokens: int = Field(
        default=128000,
        gt=0,
        description="模型原始上下文窗口大小",
    )
    safety_factor: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="安全系数，预留比例给 LLM 回复",
    )
    precision_ratio: float = Field(
        default=0.6,
        gt=0.0,
        le=1.0,
        description="精确区占比（近期消息保留比例）",
    )


class MemorySchema(BaseModel):
    """长期记忆配置 — 数据库路径、检索策略。"""

    db_path: str = Field(
        default="pyagent_memory.db",
        description="SQLite 数据库文件路径",
    )
    load_limit: int = Field(
        default=1000,
        ge=1,
        description="无预算时单次加载最大消息数",
    )
    token_budget: TokenBudgetSchema = Field(
        default_factory=TokenBudgetSchema,
        description="Token 预算子配置",
    )


# ═══════════════════════════════════════════════════════════════
# v0.10.0: Security Governance 配置
# ═══════════════════════════════════════════════════════════════


class PhaseRestrictionConfig(BaseModel):
    """阶段限制配置 — 在 PLANNING/VERIFYING 阶段禁止写入类工具。"""

    enabled: bool = Field(default=True, description="是否启用阶段限制检查")
    blocked_in_readonly_phases: list[str] = Field(
        default_factory=lambda: [
            "write_file", "execute_python", "execute_command", "delete_file",
        ],
        description="在 planning/verifying 阶段禁止的工具名称列表",
    )


class ComboDetectionConfig(BaseModel):
    """组合风险检测配置 — 滑动窗口 + 快路径 set 匹配。"""

    enabled: bool = Field(default=True, description="是否启用组合风险检测")
    sliding_window_seconds: float = Field(
        default=300.0, ge=30.0, le=3600.0,
        description="风险上下文滑动窗口时长（秒），默认 5 分钟",
    )
    max_records: int = Field(
        default=50, ge=10, le=200,
        description="滑动窗口最大记录数（安全阀）",
    )


class ComboRuleSchema(BaseModel):
    """单条字段级精炼规则 — 在快路径命中后执行字段级匹配。"""

    name: str = Field(
        ..., min_length=1,
        description="规则名称，如 'write_then_exec_same_file'",
    )
    sequence: list[str] = Field(
        ..., min_length=2,
        description="触发高危组合的工具序列",
    )
    match_on: Optional[str] = Field(
        default=None,
        description="要求匹配的参数字段名。如 'file_path' 要求前后操作同一文件",
    )
    within_window: int = Field(
        default=5, ge=1, le=50,
        description="在最近 N 次调用内查找匹配",
    )


class ParamWhitelistConfig(BaseModel):
    """参数白名单校验配置。"""

    enabled: bool = Field(default=True, description="是否启用参数白名单校验")
    allowed_command_prefixes: list[str] = Field(
        default_factory=list,
        description="execute_command 允许的命令前缀。空列表使用默认值",
    )
    blocked_command_patterns: list[str] = Field(
        default_factory=list,
        description="execute_command 额外阻止的正则模式（合并到默认列表）",
    )
    domain_allowlist: list[str] = Field(
        default_factory=list,
        description="http_request 允许的域名列表。空列表表示允许所有",
    )
    blocked_ip_patterns: list[str] = Field(
        default_factory=list,
        description="http_request 额外阻止的 IP 正则（合并到默认列表）",
    )


class SecurityCBConfig(BaseModel):
    """安全熔断器配置 — 三态 Circuit Breaker。"""

    enabled: bool = Field(default=True, description="是否启用安全熔断器")
    max_blocks: int = Field(
        default=5, ge=1, le=50,
        description="触发熔断的 BLOCK 事件数阈值",
    )
    window_seconds: float = Field(
        default=60.0, ge=10.0, le=600.0,
        description="BLOCK 事件滑动窗口时长（秒）",
    )
    cooldown_seconds: float = Field(
        default=300.0, ge=60.0, le=3600.0,
        description="OPEN 状态冷却时间（秒），每次 HALF_OPEN→OPEN 乘以 backoff_factor",
    )
    backoff_factor: float = Field(
        default=1.5, ge=1.0, le=10.0,
        description="指数退避因子",
    )
    max_backoff_seconds: float = Field(
        default=3600.0, ge=60.0, le=86400.0,
        description="冷却时间上限（秒）",
    )


class PerformanceConfig(BaseModel):
    """安全校验性能监控配置。"""

    enabled: bool = Field(default=True, description="是否启用性能监控与自动降级")
    degrade_threshold_ms: float = Field(
        default=50.0, ge=10.0, le=500.0,
        description="SecurityGovernance.check() 耗时超过此阈值时自动禁用 combo 检测",
    )


class SecurityConfig(BaseModel):
    """安全治理配置 — 独立于 llm/agent/memory 的第四配置块。

    所有子配置均有 default_factory，缺失 YAML section 不报错。
    enabled: false 可完全关闭安全治理（向后兼容）。
    """

    enabled: bool = Field(default=True, description="安全治理总开关")
    phase_restrictions: PhaseRestrictionConfig = Field(
        default_factory=PhaseRestrictionConfig,
    )
    combo_detection: ComboDetectionConfig = Field(
        default_factory=ComboDetectionConfig,
    )
    combo_rules: list[ComboRuleSchema] = Field(
        default_factory=list,
        description="字段级精炼规则列表（默认空，仅使用 HIGH_RISK_COMBOS 快路径）",
    )
    param_whitelist: ParamWhitelistConfig = Field(
        default_factory=ParamWhitelistConfig,
    )
    circuit_breaker: SecurityCBConfig = Field(
        default_factory=SecurityCBConfig,
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
    )


class SandboxConfig(BaseModel):
    """CubeSandbox 配置（v1.1 新增）。"""

    enabled: bool = Field(default=False, description="是否启用 CubeSandbox 微 VM 执行（替代本地 subprocess）")
    vm_host: str = Field(default="192.168.100.130", description="CubeSandbox 宿主机 IP")
    vm_user: str = Field(default="ananan", description="SSH 用户名")
    template: str = Field(default="tpl-a1ac6a013c6747a5bf64812f", description="沙箱模板 ID")


class MCPServerConfig(BaseModel):
    """MCP 服务器连接配置。"""

    name: str = Field(description="唯一服务器名称")
    command: str = Field(description="启动命令（空格分隔）")
    description: str = Field(default="", description="服务器描述")
    enabled: bool = Field(default=True)


class MCPConfig(BaseModel):
    """MCP 客户端配置。"""

    enabled: bool = Field(default=False, description="启用 MCP 协议替代 SSH 管道")
    servers: list[MCPServerConfig] = Field(default_factory=list)


class HarnessYamlConfig(BaseModel):
    """驾驭工程 YAML 配置的顶层 Schema。

    四层结构：
        llm      — 模型连接参数
        agent    — Agent 行为参数
        memory   — 长期记忆参数
        security — 安全治理参数（v0.10.0 新增）

    使用方式：
        config = HarnessYamlConfig(
            llm={"model": "deepseek-chat"},
            agent={"max_iterations": 15},
        )
    """

    llm: LLMSchema = Field(
        default_factory=LLMSchema,
        description="LLM 连接配置",
    )
    agent: AgentSchema = Field(
        default_factory=AgentSchema,
        description="Agent 行为配置",
    )
    memory: MemorySchema = Field(
        default_factory=MemorySchema,
        description="长期记忆配置",
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="安全治理配置（v0.10.0 新增）",
    )
    sandbox: Optional[SandboxConfig] = Field(
        default=None,
        description="CubeSandbox 沙箱配置（v1.1 新增）",
    )
    mcp: Optional[MCPConfig] = Field(
        default=None,
        description="MCP 协议配置（v1.1 新增）",
    )
