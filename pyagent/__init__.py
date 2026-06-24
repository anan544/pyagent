"""
PyAgent — 一个可扩展的 LLM Agent 框架。

核心架构：ReAct Loop（推理-行动循环）+ PEVR Loop（规划-执行-验收-修补）
LLM 决定何时调用工具、何时返回最终结果。

v0.10.0 新增：
    - SecurityGovernance: 安全治理引擎 — Layer 1 工具前置门控（阶段限制 + 组合检测 + 参数白名单）
    - GovernanceWrapper: Executor 层透明拦截包装器（方案 B + ExecutionContext）
    - SessionRiskContext: 会话级滑动窗口风险上下文（字段级精炼匹配）
    - SecurityCircuitBreaker: 三态安全熔断器（CLOSED/OPEN/HALF_OPEN + 指数退避）
    - ParameterWhitelistValidator: 参数白名单校验器（命令/HTTP/文件路径）
    - ComboRuleEngine: 字段级精炼规则引擎（双层匹配降低误报）
    - SecurityConfig: YAML security 配置块（与 llm/agent/memory 平级）

v0.9.0 新增：
    - RuleRecommender: 高危组合规则推荐引擎（频次异常检测 + 指纹聚类 + 事故关联）
    - RuleRecommendation / RecommendationReport: 推荐报告数据模型
    - FingerprintHotspot: 高频封堵指纹热点模型

v0.8.0 新增：
    - AuditLogReader: 审计日志增量读取器（增量游标 + 防御性解析 + 惰性过滤）
    - ThresholdAnalyzer: 收敛阈值 P75 分位数分析（诊断推荐）
    - ThresholdAdapter: 阈值 DI 适配器（ConvergenceDetector 注入）
    - ConvergenceDetector 支持 threshold_adapter DI（向后兼容）

快速开始：
    from pyagent.core import Agent, AgentConfig
    from pyagent.tools import ToolRegistry, ReadFileTool, CodeExecutorTool
    from pyagent.llm import OpenAICompatProvider
    from pyagent.memory import MemoryManager

    # 可选：启用长期记忆
    memory = MemoryManager("my_project.db")
    await memory.create_session("dev-session")

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(CodeExecutorTool())

    llm = OpenAICompatProvider(model="gpt-4o")
    agent = Agent(AgentConfig(), registry, llm, memory=memory)
    result = await agent.run("帮我执行一段代码", session_id="dev-session")
"""

__version__ = "0.10.0"
