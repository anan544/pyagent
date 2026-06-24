"""
可观测性与安全审计基础设施。

1.5.4 核心新增：为 PEVR 状态机补全"黑盒透视"能力。

设计原则：
    - 零侵入状态机主循环：观测钩子通过 sm.context["observability"] 注入，
      state_machine.py 核心逻辑不变
    - 异步缓冲 + 批量发送：状态机仅写本地 asyncio.Queue，
      独立后台协程消费并推送
    - 安全审计事件在权限门控决策点同步生成（而非事后补记）
    - 敏感数据脱敏在序列化前完成（而非传输时）
    - ObservabilityContext 独立于 WorkingMemory，不污染业务字段

模块结构：
    SecurityAuditEvent  — 结构化审计事件模型
    TraceContext         — Trace ID 生成与管理
    Sanitizer            — 递归脱敏器（支持嵌套路径通配）
    AuditLogger          — 异步缓冲审计日志
    ObservabilityContext — 观测上下文（生命周期管理）
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("pyagent.harness.observability")

# ── 敏感字段配置 ──────────────────────────────────────

SENSITIVE_FIELDS: set[str] = {
    "password", "api_key", "apikey", "token", "secret",
    "authorization", "private_key", "privatekey", "access_key",
    "secret_key", "credential", "passphrase",
}


# ═══════════════════════════════════════════════════════════════
# SecurityAuditEvent
# ═══════════════════════════════════════════════════════════════

class SecurityAuditEvent(BaseModel):
    """
    结构化安全审计事件。

    每次权限门控决策（ALLOW / BLOCK / REPAIR）均生成一条事件。
    包含完整决策链：rule_id → plan_step → risk_score，支持事后溯源。

    强制字段：
        decision:  门控决策结果
        rule_id:   匹配的门控规则 ID（可追溯到具体配置行）
        plan_step_fingerprint:  Plan 步骤指纹（锚定唯一步骤）
        risk_score: 高危组合评分（0-100）
    """

    decision: Literal["ALLOW", "BLOCK", "REPAIR"] = Field(
        ...,
        description="门控决策结果。ALLOW=放行, BLOCK=拦截, REPAIR=进入修补。",
    )
    rule_id: str = Field(
        ...,
        description="匹配的门控规则 ID。如 'HIGH_RISK_COMBOS[0]'、'step_level_check'。",
    )
    plan_step_fingerprint: str = Field(
        default="",
        description="Plan 步骤指纹 = hash(frozenset(params.items()))，锚定唯一步骤。",
    )
    risk_score: int = Field(
        default=0,
        ge=0, le=100,
        description="高危组合评分（0=无风险, 100=极度危险）。",
    )
    tool_name: str = Field(
        default="",
        description="涉及的工具名称。",
    )
    tool_params: dict[str, Any] = Field(
        default_factory=dict,
        description="脱敏后的工具参数。",
    )
    phase: str = Field(
        default="",
        description="当前 PEVR 阶段。",
    )
    state: str = Field(
        default="",
        description="当前状态机状态。",
    )
    trace_id: str = Field(
        default="",
        description="全生命周期追踪 ID。",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="事件时间戳（ISO 8601 UTC）。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据（如 step_id, repair_attempt 等）。",
    )

    @property
    def is_p0(self) -> bool:
        """P0 级事件：BLOCK 或 REPAIR 决策（安全相关）。"""
        return self.decision in ("BLOCK", "REPAIR")

    def to_json(self) -> str:
        """序列化为单行 JSON（便于 JSONL 写入）。"""
        return self.model_dump_json()


# ═══════════════════════════════════════════════════════════════
# TraceContext
# ═══════════════════════════════════════════════════════════════

class TraceContext:
    """
    Trace ID 生成与管理。

    使用方式：
        trace_id = TraceContext.generate()
        # 存入 ExecutionPlan.trace_id
        # 贯穿全生命周期 + 断点恢复
    """

    @staticmethod
    def generate() -> str:
        """生成 UUID4 Trace ID。"""
        return uuid.uuid4().hex

    @staticmethod
    def current(ctx: dict) -> str:
        """
        从状态机上下文提取当前 trace_id。

        优先级：
            1. ctx["observability"].trace_id
            2. ctx["plan"].trace_id（通过 ExecutionPlan）
            3. ctx.get("trace_id", "")
        """
        obs = ctx.get("observability")
        if obs and hasattr(obs, 'trace_id') and obs.trace_id:
            return obs.trace_id
        plan = ctx.get("plan")
        if plan and hasattr(plan, 'trace_id') and plan.trace_id:
            return plan.trace_id
        return ctx.get("trace_id", "")


# ═══════════════════════════════════════════════════════════════
# Sanitizer
# ═══════════════════════════════════════════════════════════════

class SanitizedSerializable:
    """
    脱敏协议：进入观测管道的对象应先调用 .sanitize()。

    子类实现：
        def sanitize(self) -> dict: ...
    """

    def sanitize(self) -> dict:
        raise NotImplementedError


class Sanitizer:
    """
    递归脱敏器 — 在序列化前替换敏感字段。

    特性：
        - 递归遍历 dict / list / Pydantic 模型
        - 支持通配符匹配嵌套路径（如 params.**.secret）
        - 脱敏后返回新对象，原始对象永不修改
        - SENSITIVE_FIELDS 集中配置，支持自定义扩展

    使用方式：
        sanitizer = Sanitizer(extra_fields={"custom_key"})
        clean = sanitizer.sanitize({"api_key": "sk-123", "nested": {"token": "x"}})
        # → {"api_key": "[REDACTED]", "nested": {"token": "[REDACTED]"}}
    """

    REDACTED = "[REDACTED]"

    def __init__(self, extra_fields: set[str] | None = None):
        """
        Args:
            extra_fields: 额外敏感字段名集合（合并到 SENSITIVE_FIELDS）。
        """
        self._sensitive = SENSITIVE_FIELDS.copy()
        if extra_fields:
            self._sensitive.update(extra_fields)

    def sanitize(self, obj: Any) -> Any:
        """
        递归脱敏，返回新对象。

        Args:
            obj: 原始对象（dict / list / Pydantic / 基本类型）。

        Returns:
            脱敏后的新对象（原始对象不变）。
        """
        if isinstance(obj, dict):
            return self._sanitize_dict(obj)
        elif isinstance(obj, list):
            return [self.sanitize(item) for item in obj]
        elif hasattr(obj, 'model_dump') and callable(getattr(obj, 'model_dump', None)):
            return self.sanitize(obj.model_dump())
        elif hasattr(obj, 'sanitize') and callable(obj.sanitize):
            return obj.sanitize()
        return obj

    def _sanitize_dict(self, d: dict) -> dict:
        """递归处理 dict，匹配敏感 key。"""
        result = {}
        for key, value in d.items():
            # 检查 key 是否在敏感字段集合中（不区分大小写）
            if self._is_sensitive(key):
                result[key] = self.REDACTED
            elif isinstance(value, dict):
                result[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                result[key] = [self.sanitize(item) for item in value]
            elif isinstance(value, str):
                # 对字符串值检查是否包含敏感模式（如 Bearer token）
                result[key] = self._redact_sensitive_patterns(value)
            else:
                result[key] = value
        return result

    def _is_sensitive(self, key: str) -> bool:
        """
        判断 key 是否为敏感字段。

        支持：
            - 精确匹配（不区分大小写）
            - `**` 通配符路径段（如 params.**.secret）
        """
        key_lower = key.lower().replace('-', '_').replace(' ', '_')
        for sensitive in self._sensitive:
            sensitive_lower = sensitive.lower().replace('-', '_').replace(' ', '_')
            if key_lower == sensitive_lower:
                return True
            # 支持包含匹配（如 "api_key" 匹配 "x_api_key"）
            if sensitive_lower in key_lower or key_lower in sensitive_lower:
                return True
        return False

    def _redact_sensitive_patterns(self, value: str) -> str:
        """对字符串值中的敏感模式进行替换。"""
        # Bearer token 模式
        value = re.sub(
            r'(Bearer\s+)[A-Za-z0-9\-._~+/]+',
            r'\1[REDACTED]',
            value,
        )
        # API key 模式 (sk-..., ak-...)
        value = re.sub(
            r'\b(sk|ak|pk)-[A-Za-z0-9]{8,}',
            r'\1-[REDACTED]',
            value,
        )
        # JWT token 模式
        value = re.sub(
            r'\beyJ[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]{20,}\.[A-Za-z0-9\-_]{20,}',
            '[REDACTED_JWT]',
            value,
        )
        return value


# ═══════════════════════════════════════════════════════════════
# AuditLogger
# ═══════════════════════════════════════════════════════════════

class AuditLogger:
    """
    异步缓冲审计日志器。

    状态机回调通过 emit() 非阻塞写入事件到本地 asyncio.Queue。
    独立后台协程批量消费，推送到观测后端（默认 logger.info）。
    生产环境可接入 OTLP / Loki / Elasticsearch。

    队列满时：
        - P0 级事件（BLOCK/REPAIR）→ 原子写入本地 JSONL 兜底文件
        - 非 P0 事件（ALLOW）→ 静默丢弃

    使用方式：
        audit = AuditLogger(fallback_path=".claude/audit_fallback.jsonl")
        await audit.start()
        audit.emit(SecurityAuditEvent(decision="BLOCK", ...))
        # ... PEVR 循环 ...
        await audit.stop()
    """

    # 队列最大容量
    MAX_QUEUE_SIZE = 4096
    # 兜底文件最大大小（字节），超过后自动轮转
    MAX_FALLBACK_SIZE = 10 * 1024 * 1024  # 10 MiB
    # 轮转备份保留数
    FALLBACK_BACKUPS = 3

    def __init__(
        self,
        fallback_path: str = ".claude/audit_fallback.jsonl",
        sanitizer: Sanitizer | None = None,
    ):
        """
        Args:
            fallback_path: P0 事件兜底文件路径。
            sanitizer: 脱敏器实例。None 时自动创建。
        """
        self.fallback_path = Path(fallback_path)
        self.sanitizer = sanitizer or Sanitizer()
        self._queue: asyncio.Queue[SecurityAuditEvent | None] = asyncio.Queue(
            maxsize=self.MAX_QUEUE_SIZE
        )
        self._task: asyncio.Task | None = None
        self._running = False
        self._dropped_non_p0: int = 0
        self._flushed_p0: int = 0

    # ── 生命周期 ──────────────────────────────────

    async def start(self):
        """启动后台消费协程。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._consume())
        # 启动时检测并告警未上报 P0 事件
        pending = self.drain_pending()
        if pending:
            logger.warning(
                "[AuditLogger] 检测到 %d 条未上报的 P0 审计事件（兜底文件残留），"
                "将在本次会话中重新上报",
                len(pending),
            )
            for event in pending:
                self.emit(event)
        logger.debug("[AuditLogger] 已启动，队列容量=%d", self.MAX_QUEUE_SIZE)

    async def stop(self):
        """停止后台协程，刷新队列中剩余事件。"""
        if not self._running:
            return
        self._running = False
        # 发送停止信号
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        if self._task:
            await self._task
            self._task = None
        logger.debug(
            "[AuditLogger] 已停止。丢弃=%d 已落盘=%d",
            self._dropped_non_p0, self._flushed_p0,
        )

    # ── 事件写入 ──────────────────────────────────

    def emit(self, event: SecurityAuditEvent):
        """
        非阻塞写入审计事件。

        P0 事件队列满时 → 原子写入兜底文件。
        非 P0 事件队列满时 → 静默丢弃。

        Args:
            event: 审计事件。
        """
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            if event.is_p0:
                self._write_fallback(event)
            else:
                self._dropped_non_p0 += 1
                if self._dropped_non_p0 % 100 == 1:  # 降频告警
                    logger.warning(
                        "[AuditLogger] 队列满，已丢弃 %d 条非 P0 事件",
                        self._dropped_non_p0,
                    )

    def flush(self):
        """同步等待队列清空（仅用于测试）。"""
        # 创建一个临时事件标记 flush 完成
        # 实际通过 busy-wait 等待队列大小降到 0
        import time
        timeout = 5.0
        start = time.monotonic()
        while self._queue.qsize() > 0:
            if time.monotonic() - start > timeout:
                break
            time.sleep(0.01)

    # ── 后台消费 ──────────────────────────────────

    async def _consume(self):
        """后台协程：批量消费队列中的审计事件。"""
        batch: list[SecurityAuditEvent] = []
        while self._running or not self._queue.empty():
            try:
                # 非阻塞获取，超时 0.1s
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                # 超时 → 刷新当前批次
                if batch:
                    await self._flush_batch(batch)
                    batch.clear()
                continue

            if event is None:  # 停止信号
                if batch:
                    await self._flush_batch(batch)
                    batch.clear()
                break

            batch.append(event)
            # 批次大小达到 50 或队列为空时刷新
            if len(batch) >= 50:
                await self._flush_batch(batch)
                batch.clear()

        # 最后刷新
        if batch:
            await self._flush_batch(batch)

    async def _flush_batch(self, batch: list[SecurityAuditEvent]):
        """将批次推送到观测后端。"""
        # 当前默认：logger.info 输出脱敏后的 JSON
        # 生产环境可在此接入 OTLP / Loki / Elasticsearch
        for event in batch:
            sanitized = self.sanitizer.sanitize(event.model_dump())
            logger.info("[AUDIT] %s | rule=%s | risk=%d | decision=%s",
                        sanitized.get("trace_id", "")[:8],
                        sanitized.get("rule_id", ""),
                        sanitized.get("risk_score", 0),
                        sanitized.get("decision", ""),
                        )

    # ── 本地兜底 ──────────────────────────────────

    def _write_fallback(self, event: SecurityAuditEvent):
        """
        P0 事件原子写入兜底文件。

        特性：
            - 原子写入（先写临时文件，再 os.replace）
            - 自动轮转（超过 MAX_FALLBACK_SIZE 时）
            - 0o600 权限（仅 owner 可读写）
        """
        try:
            # 确保目录存在
            self.fallback_path.parent.mkdir(parents=True, exist_ok=True)

            # 轮转检查
            if self.fallback_path.exists() and \
               self.fallback_path.stat().st_size > self.MAX_FALLBACK_SIZE:
                self._rotate()

            # 原子写入（临时文件 + rename）
            sanitized = self.sanitizer.sanitize(event.model_dump())
            line = json.dumps(sanitized, ensure_ascii=False) + "\n"

            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.fallback_path.parent),
                prefix=".audit_tmp_",
            )
            try:
                os.write(fd, line.encode('utf-8'))
                os.fsync(fd)
            finally:
                os.close(fd)

            # 原子替换 + 设权限
            os.replace(tmp_path, str(self.fallback_path))
            try:
                os.chmod(str(self.fallback_path), 0o600)
            except Exception:
                pass  # Windows 下 chmod 行为不同，忽略

            self._flushed_p0 += 1
        except Exception as e:
            logger.error("[AuditLogger] P0 事件落盘失败: %s", e)

    def _rotate(self):
        """轮转兜底文件。"""
        for i in range(self.FALLBACK_BACKUPS - 1, -1, -1):
            src = self.fallback_path if i == 0 else \
                self.fallback_path.with_suffix(f".jsonl.{i}")
            dst = self.fallback_path.with_suffix(f".jsonl.{i + 1}")
            if src.exists():
                try:
                    os.replace(str(src), str(dst))
                except Exception:
                    pass

    def drain_pending(self) -> list[SecurityAuditEvent]:
        """
        读取兜底文件中未上报的 P0 事件。

        用于启动时检测上次会话遗留的未上报事件。

        Returns:
            未上报的 SecurityAuditEvent 列表。
        """
        pending: list[SecurityAuditEvent] = []
        if not self.fallback_path.exists():
            return pending

        try:
            with open(self.fallback_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        pending.append(SecurityAuditEvent(**data))
                    except Exception:
                        logger.debug("[AuditLogger] 跳过无效行: %.100s", line)
        except Exception as e:
            logger.warning("[AuditLogger] 读取兜底文件失败: %s", e)

        # 清空兜底文件（事件已读入内存，将重新上报）
        if pending:
            try:
                self.fallback_path.unlink()
            except Exception:
                pass

        return pending


# ═══════════════════════════════════════════════════════════════
# ObservabilityContext
# ═══════════════════════════════════════════════════════════════

class ObservabilityContext:
    """
    观测上下文 — 独立于 WorkingMemory。

    贯穿 PEVR 全生命周期，但不污染业务字段。
    通过 sm.context["observability"] 注入到状态机。

    使用方式：
        obs = ObservabilityContext()
        obs.trace_id = TraceContext.generate()
        await obs.start()

        # 在状态机回调中：
        obs.emit(SecurityAuditEvent(...))

        await obs.stop()
    """

    def __init__(
        self,
        trace_id: str = "",
        fallback_path: str = ".claude/audit_fallback.jsonl",
    ):
        self.trace_id = trace_id
        self.audit_logger = AuditLogger(fallback_path=fallback_path)
        self.sanitizer = self.audit_logger.sanitizer
        self._started = False

    # ── 生命周期 ──────────────────────────────────

    async def start(self):
        """启动观测上下文（含审计日志后台协程）。"""
        if self._started:
            return
        await self.audit_logger.start()
        self._started = True

    async def stop(self):
        """停止观测上下文，刷新所有缓冲。"""
        if not self._started:
            return
        await self.audit_logger.stop()
        self._started = False

    # ── 便捷方法 ──────────────────────────────────

    def emit(self, event: SecurityAuditEvent):
        """非阻塞写入审计事件。"""
        # 自动填充 trace_id（如果未设置）
        if not event.trace_id and self.trace_id:
            event.trace_id = self.trace_id
        self.audit_logger.emit(event)

    def log_decision(
        self,
        decision: Literal["ALLOW", "BLOCK", "REPAIR"],
        rule_id: str,
        tool_name: str = "",
        tool_params: dict | None = None,
        risk_score: int = 0,
        plan_step_fingerprint: str = "",
        phase: str = "",
        state: str = "",
        metadata: dict | None = None,
    ):
        """
        便捷方法：创建并写入一条审计事件。

        Args:
            decision: 门控决策。
            rule_id: 门控规则 ID。
            tool_name: 工具名称。
            tool_params: 脱敏前工具参数（会自动脱敏）。
            risk_score: 风险评分。
            plan_step_fingerprint: Plan 步骤指纹。
            phase: 当前 PEVR 阶段。
            state: 当前状态机状态。
            metadata: 扩展元数据。
        """
        # 脱敏工具参数
        safe_params = {}
        if tool_params:
            safe_params = self.sanitizer.sanitize(tool_params)

        event = SecurityAuditEvent(
            decision=decision,
            rule_id=rule_id,
            plan_step_fingerprint=plan_step_fingerprint,
            risk_score=risk_score,
            tool_name=tool_name,
            tool_params=safe_params,
            phase=phase,
            state=state,
            trace_id=self.trace_id,
            metadata=metadata or {},
        )
        self.emit(event)
        return event

    # ── 查询 ──────────────────────────────────────

    @property
    def dropped_count(self) -> int:
        """被丢弃的非 P0 事件数。"""
        return self.audit_logger._dropped_non_p0

    @property
    def flushed_count(self) -> int:
        """落盘的 P0 事件数。"""
        return self.audit_logger._flushed_p0


# ── 辅助：Step 指纹计算 ─────────────────────────────

def compute_step_fingerprint(params: dict) -> str:
    """
    计算步骤参数指纹 = SHA256(frozenset(params.items()))[:16]。

    用于审计事件中锚定唯一 Plan 步骤。
    """
    if not params:
        return "no_params"
    try:
        frozen = frozenset(
            (k, str(v)) for k, v in sorted(params.items())
        )
        h = hashlib.sha256(str(frozen).encode()).hexdigest()[:16]
        return h
    except Exception:
        return "fingerprint_error"


# ── 辅助：风险评分 ─────────────────────────────────

def compute_risk_score(matched_combo: set[str]) -> int:
    """
    根据高危工具组合计算风险评分（0-100）。

    评分规则：
        - 单工具高危（如 delete_file）：60
        - 双工具高危（如 write_file + execute_python）：80
        - 三工具及以上：95
        - 未匹配：0
    """
    if not matched_combo:
        return 0
    n = len(matched_combo)
    if n >= 3:
        return 95
    elif n == 2:
        return 80
    else:
        return 60
