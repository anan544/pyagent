"""
驾驭工程 — 结构化上下文组装器（Context Assembler）。

将散落在 Agent 主循环中的字符串拼接逻辑，抽离为独立的、可替换的
ContextAssembler 类族：

    - ReactContextAssembler: 向后兼容的 ReAct 消息组装（提取自 agent.py）
    - PEVRContextAssembler: PEVR 四阶段上下文组装（四槽位 + 阶段感知 + 裁剪）
    - PEVRRunner: PEVR 一级循环编排器（状态机驱动）

1.5.3 新增：
    - PEVRStateMachine: 枚举驱动状态机，替代手写 if-else 流转
    - PermissionGate: 三级权限门控（计划级预审 + 步骤级检查 + 修补降级）
    - ScopedToolRegistry: 工具注册表装饰器（修补阶段工具隔离）
    - PEVRCheckpoint: 最小可恢复单元（状态持久化）
    - InvalidStateError: 状态合法性异常

1.5.7 新增：
    - RuleRecommender: 高危组合规则推荐引擎（频次异常检测 + 指纹聚类）
    - RuleRecommendation / RecommendationReport: 推荐数据模型
    - FingerprintHotspot: 高频封堵指纹热点模型

v0.10.0 新增：
    - SecurityGovernance: 安全治理引擎（Layer 1 前置门控编排器）
    - GovernanceWrapper: Executor 层透明拦截包装器
    - SecurityDecision / ExecutionContext: 安全决策与执行上下文（不可变值对象）
    - SessionRiskContext: 会话级滑动窗口风险上下文（字段级匹配）
    - SecurityCircuitBreaker: 三态安全熔断器（CLOSED/OPEN/HALF_OPEN + 指数退避）
    - ParameterWhitelistValidator: 参数白名单校验器
    - ComboRule / ComboRuleEngine: 字段级精炼规则引擎

四槽位模型：
    System（系统指令 + agent.md）
    Plan（当前计划快照）
    History（对话历史）
    WorkingMemory（临时产物）

使用方式：
    # 向后兼容（无需改动）
    agent = Agent(config, registry, llm, memory=memory)
    result = await agent.run("任务")

    # PEVR 模式
    from pyagent.harness.context import PEVRRunner, PEVRContextAssembler

    assembler = PEVRContextAssembler(total_budget=64000)
    runner = PEVRRunner(agent, assembler, context_files=["./agent.md"])

    result = await runner.run("审查项目", "所有测试通过")
    print(result.success, result.plan, result.verification)
"""

from .models import (
    PEVRPhase,
    PEVRResult,
    WorkingMemory,
    ContextRequest,
    ContextResult,
    SlotContent,
    SlotSet,
    StepResult,
    Step,
    ExecutionPlan,
    AcceptanceCriteria,
    PlanValidationError,
    InvalidStateError,
)
from .assembler import ContextAssembler, PEVRContextAssembler
from .react_assembler import ReactContextAssembler
from .runner import PEVRRunner
from .state_machine import (
    PEVRStateMachine,
    PEVRState,
    PEVREvent,
    Transition,
    InvalidTransitionError,
    GuardRejectedError,
)
from .permission import (
    PermissionGate,
    ScopedToolRegistry,
    HIGH_RISK_COMBOS,
)
from .checkpoint import (
    PEVRCheckpoint,
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint,
)
from .loader import ContextFileLoader
from .slots import build_slots, allocate_budgets
from .trimmer import trim_slots, estimate_tokens
from .plan_validator import PlanValidator, BusinessRule
# 1.5.4 可观测性
from .observability import (
    ObservabilityContext,
    SecurityAuditEvent,
    AuditLogger,
    Sanitizer,
    TraceContext,
    SENSITIVE_FIELDS,
)
# 1.5.4 修补强化
from .repair_context import (
    RepairContext,
    CircuitBreaker,
    ConvergenceDetector,
    RepairLog,
    ThresholdRecommendation,
    ThresholdAnalyzer,
    ThresholdAdapter,
)
# 审计日志读取器
from .audit_reader import (
    AuditLogReader,
    LogCursor,
)
# 规则推荐引擎
from .rule_recommender import (
    RuleRecommender,
    RuleRecommendation,
    RecommendationReport,
    FingerprintHotspot,
)
# v0.10.0 安全治理
from .security_governance import (
    SecurityGovernance,
    SecurityDecision,
    ExecutionContext,
)
from .governance_wrapper import GovernanceWrapper
from .session_risk_context import SessionRiskContext
from .security_circuit_breaker import SecurityCircuitBreaker
from .parameter_validator import ParameterWhitelistValidator
from .combo_rules import ComboRule, ComboRuleEngine

__all__ = [
    # 核心类
    "ContextAssembler",
    "PEVRContextAssembler",
    "ReactContextAssembler",
    "PEVRRunner",
    "PlanValidator",
    "BusinessRule",
    # 1.5.3 状态机
    "PEVRStateMachine",
    "PEVRState",
    "PEVREvent",
    "Transition",
    "InvalidTransitionError",
    "GuardRejectedError",
    # 1.5.3 权限门控
    "PermissionGate",
    "ScopedToolRegistry",
    "HIGH_RISK_COMBOS",
    # 1.5.3 检查点
    "PEVRCheckpoint",
    "save_checkpoint",
    "load_checkpoint",
    "delete_checkpoint",
    # 1.5.4 可观测性
    "ObservabilityContext",
    "SecurityAuditEvent",
    "AuditLogger",
    "Sanitizer",
    "TraceContext",
    "SENSITIVE_FIELDS",
    # 1.5.4 修补强化
    "RepairContext",
    "CircuitBreaker",
    "ConvergenceDetector",
    "RepairLog",
    "ThresholdRecommendation",
    "ThresholdAnalyzer",
    "ThresholdAdapter",
    # 审计日志读取器
    "AuditLogReader",
    "LogCursor",
    # 规则推荐引擎
    "RuleRecommender",
    "RuleRecommendation",
    "RecommendationReport",
    "FingerprintHotspot",
    # v0.10.0 安全治理
    "SecurityGovernance",
    "SecurityDecision",
    "ExecutionContext",
    "GovernanceWrapper",
    "SessionRiskContext",
    "SecurityCircuitBreaker",
    "ParameterWhitelistValidator",
    "ComboRule",
    "ComboRuleEngine",
    # 数据模型
    "PEVRPhase",
    "PEVRResult",
    "WorkingMemory",
    "ContextRequest",
    "ContextResult",
    "SlotContent",
    "SlotSet",
    "StepResult",
    "Step",
    "ExecutionPlan",
    "AcceptanceCriteria",
    "PlanValidationError",
    "InvalidStateError",
    # 工具
    "ContextFileLoader",
    "build_slots",
    "allocate_budgets",
    "trim_slots",
    "estimate_tokens",
]
