"""
修补上下文隔离与熔断机制。

1.5.4 核心新增：将修补视为一个有严格边界和退出条件的子状态机。

关键设计：
    - RepairContext：隔离上下文（不含完整对话历史，避免上下文膨胀）
    - CircuitBreaker：硬性熔断计数器 + 结构化失败报告
    - ConvergenceDetector：编辑距离相似度检测"原地打转"
    - RepairLog：修补审计日志（与主执行日志分离）

设计原则：
    - 纯逻辑：不依赖 Agent/LLM/Memory/IO
    - 可独立单元测试
    - ConvergenceDetector 接口预留，方便后续替换为语义模型
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("pyagent.harness.repair_context")


# ═══════════════════════════════════════════════════════════════
# RepairContext
# ═══════════════════════════════════════════════════════════════

@dataclass
class RepairContext:
    """
    修补上下文 — 仅包含修复所需的最小信息。

    明确不包含：
        - 完整对话历史（History）
        - WorkingMemory 全部内容
        - 其他步骤的 Artifacts

    包含：
        - 失败步骤详情
        - 原始验收标准
        - 可用修复工具白名单
        - 上一轮修补输出摘要（≤500 字符）
    """

    failed_step_description: str = ""
    """失败步骤的描述。"""

    failure_reason: str = ""
    """失败原因摘要（来自验收阶段结论或异常信息）。"""

    acceptance_criteria: str = ""
    """原始验收标准（来自 PLANNING 阶段）。"""

    available_tools: list[str] = field(default_factory=list)
    """当前可用的修复工具白名单。"""

    previous_repair_summary: str = ""
    """上一轮修补输出摘要（截断至 ≤500 字符）。"""

    repair_attempt: int = 0
    """当前修补次数（1-based）。"""

    max_repairs: int = 3
    """熔断阈值。"""

    def to_hint_text(self) -> str:
        """
        转为 observability_hints 槽位的文本内容。

        用于注入 Repair 阶段的 LLM Prompt。
        """
        parts = [
            f"修复尝试: {self.repair_attempt}/{self.max_repairs}",
            f"可用工具: {', '.join(self.available_tools) if self.available_tools else '（受限白名单）' }",
        ]
        if self.previous_repair_summary:
            parts.append(f"上一轮修补摘要: {self.previous_repair_summary[:500]}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# CircuitBreaker
# ═══════════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    硬性熔断计数器。

    修补次数达到阈值后强制转入 FAILED，并生成结构化失败报告。

    使用方式：
        cb = CircuitBreaker(max_repairs=3)
        for attempt in range(3):
            if cb.is_tripped():
                break
            result = await try_repair()
            cb.record_attempt()
            # ...
        if cb.is_tripped():
            report = cb.generate_failure_report()
    """

    def __init__(self, max_repairs: int = 3):
        if max_repairs < 1:
            raise ValueError(f"max_repairs 必须 >= 1，收到: {max_repairs}")
        self.max_repairs = max_repairs
        self.attempt_count: int = 0
        self._last_attempt_detail: str = ""
        self._last_failure_reason: str = ""

    @property
    def remaining(self) -> int:
        """剩余修补次数。"""
        return max(0, self.max_repairs - self.attempt_count)

    def is_tripped(self) -> bool:
        """是否已熔断（达到阈值）。"""
        return self.attempt_count >= self.max_repairs

    def record_attempt(self, detail: str = "", failure_reason: str = ""):
        """
        记录一次修补尝试。

        Args:
            detail: 本次尝试的摘要。
            failure_reason: 本次失败原因。
        """
        self.attempt_count += 1
        self._last_attempt_detail = detail
        self._last_failure_reason = failure_reason
        logger.debug(
            "[CircuitBreaker] 修补 %d/%d (剩余 %d)",
            self.attempt_count, self.max_repairs, self.remaining,
        )

    def generate_failure_report(self) -> dict:
        """
        生成结构化失败报告。

        Returns:
            dict 含:
                - tripped: 是否熔断
                - attempts: 已尝试次数
                - max_repairs: 熔断阈值
                - last_attempt: 最后一次尝试摘要
                - last_failure: 最后一次失败原因
                - suggestion: 建议人工介入点
        """
        return {
            "tripped": self.is_tripped(),
            "attempts": self.attempt_count,
            "max_repairs": self.max_repairs,
            "last_attempt": self._last_attempt_detail or "无",
            "last_failure": self._last_failure_reason or "未知",
            "suggestion": (
                f"修补已尝试 {self.attempt_count} 次（阈值 {self.max_repairs}），"
                f"建议人工介入检查以下方向：\n"
                f"1. 验收标准是否合理（当前标准: 请检查 Plan 快照）\n"
                f"2. 可用工具是否足够（当前白名单: 请检查 allowed_repair_tools）\n"
                f"3. 失败原因是否存在系统性障碍（非模型能力问题）"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ═══════════════════════════════════════════════════════════════
# ConvergenceDetector
# ═══════════════════════════════════════════════════════════════

class ConvergenceDetector:
    """
    修补收敛检测器。

    通过比对连续两次修补的 changes_made 字段，检测"原地打转"。
    使用编辑距离（difflib.SequenceMatcher）计算文本相似度，
    超过阈值（默认 0.7）判定为未收敛。

    接口预留：
        check_semantic(current, model=None) — 未来替换为嵌入向量语义相似度。

    使用方式：
        cd = ConvergenceDetector(threshold=0.7)
        cd.check_convergence("修改了文件路径")  # False（首次，无历史）
        cd.check_convergence("修改了文件路径")  # True（与上次完全相同）
        cd.check_convergence("重构了模块导入")   # False（不同）
    """

    def __init__(self, threshold: float = 0.7, threshold_adapter=None):
        """
        Args:
            threshold: 相似度阈值（0.0-1.0）。当提供了 threshold_adapter 时，
                      此值作为初始值，实际阈值由 adapter.get_threshold() 决定。
            threshold_adapter: 可选的 ThresholdAdapter 实例（DI 注入）。
                              提供时，check_convergence() 每次调用前查询最新阈值。
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold 必须在 [0.0, 1.0]，收到: {threshold}")
        self.threshold = threshold
        self._threshold_adapter = threshold_adapter
        self.previous_changes: str | None = None
        self._last_score: float = 0.0

    @property
    def last_score(self) -> float:
        """最近一次相似度分数（用于埋点调优）。"""
        return self._last_score

    def check_convergence(self, current_changes: str) -> bool:
        """
        检查是否与上一轮修补收敛（原地打转）。

        Args:
            current_changes: 本轮修补的 changes_made 字段。

        Returns:
            True 如果检测到收敛（应立即熔断），False 如果正常。
        """
        normalized = self._normalize(current_changes)

        if self.previous_changes is None:
            # 首次修补，无历史可比较
            self.previous_changes = normalized
            self._last_score = 0.0
            return False

        self._last_score = self._similarity(self.previous_changes, normalized)

        # 解析有效阈值：优先使用 adapter（DI 注入），回退到固定值
        effective = self.threshold
        if self._threshold_adapter is not None:
            effective = self._threshold_adapter.get_threshold()

        is_converged = self._last_score >= effective

        if is_converged:
            logger.warning(
                "[ConvergenceDetector] 检测到原地打转！"
                "上轮='%s' 本轮='%s' 相似度=%.3f (阈值=%.2f)",
                self.previous_changes[:80], normalized[:80],
                self._last_score, effective,
            )

        # 更新历史
        self.previous_changes = normalized
        return is_converged

    def reset(self):
        """重置历史（用于新一轮修补）。"""
        self.previous_changes = None
        self._last_score = 0.0

    def _normalize(self, text: str) -> str:
        """
        归一化预处理。

        步骤：
            1. 去空白（strip + 多空格压缩）
            2. 转小写
            3. 去标点符号
        """
        if not text:
            return ""
        # 多空格压缩
        text = re.sub(r'\s+', ' ', text.strip())
        # 转小写
        text = text.lower()
        # 去标点（保留字母数字和空格）
        text = re.sub(r'[^\w\s]', '', text)
        return text

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """
        计算两段文本的编辑距离相似度。

        使用 difflib.SequenceMatcher（标准库，无外部依赖）。
        """
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    def check_semantic(self, current: str, model=None) -> float:
        """
        [预留接口] 语义相似度检测。

        Args:
            current: 本轮 changes_made。
            model: 嵌入模型实例（未来实现）。

        Returns:
            语义相似度分数 (0.0-1.0)。
        """
        # 当前回退到编辑距离
        if self.previous_changes is None:
            return 0.0
        return self._similarity(
            self._normalize(self.previous_changes),
            self._normalize(current),
        )


# ═══════════════════════════════════════════════════════════════
# RepairLog
# ═══════════════════════════════════════════════════════════════

class RepairLog:
    """
    修补审计日志 — 与主执行日志分离。

    记录每轮修补的完整决策链：attempt → changes_made → fixed → convergence_score。

    使用方式：
        log = RepairLog()
        log.record(attempt=1, changes_made="修改了路径", fixed=True, convergence_score=0.0)
        events = log.to_audit_events(trace_id="abc123")
    """

    def __init__(self):
        self.entries: list[dict[str, Any]] = []

    def record(
        self,
        attempt: int,
        changes_made: str,
        fixed: bool,
        convergence_score: float = 0.0,
        extra: dict | None = None,
    ):
        """
        记录一轮修补。

        Args:
            attempt: 修补次数（1-based）。
            changes_made: 本轮修改摘要（来自 LLM JSON 输出）。
            fixed: 是否修复成功。
            convergence_score: 收敛检测相似度分数。
            extra: 扩展字段。
        """
        entry = {
            "attempt": attempt,
            "changes_made": changes_made,
            "fixed": fixed,
            "convergence_score": convergence_score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            entry.update(extra)
        self.entries.append(entry)
        logger.debug(
            "[RepairLog] 第 %d 轮: fixed=%s convergence=%.3f changes='%s'",
            attempt, fixed, convergence_score, changes_made[:80],
        )

    def to_audit_events(self, trace_id: str = "") -> list:
        """
        将修补日志转换为 SecurityAuditEvent 列表。

        Args:
            trace_id: 追踪 ID。

        Returns:
            SecurityAuditEvent 列表。
        """
        from .observability import SecurityAuditEvent

        events: list[SecurityAuditEvent] = []
        for entry in self.entries:
            decision = "REPAIR"
            rule_id = (
                "convergence_detected"
                if entry.get("convergence_score", 0) >= 0.7
                else f"repair_attempt_{entry['attempt']}"
            )
            events.append(SecurityAuditEvent(
                decision=decision,
                rule_id=rule_id,
                risk_score=40 if entry["fixed"] else 70,
                tool_name="repair",
                phase="repairing",
                trace_id=trace_id,
                metadata={
                    "repair_attempt": entry["attempt"],
                    "changes_made": entry.get("changes_made", ""),
                    "fixed": entry["fixed"],
                    "convergence_score": entry.get("convergence_score", 0),
                },
            ))
        return events

    def last_summary(self, max_chars: int = 500) -> str:
        """返回最近一轮修补的摘要文本。"""
        if not self.entries:
            return ""
        last = self.entries[-1]
        return (
            f"第 {last['attempt']} 轮: "
            f"{'✓ 已修复' if last['fixed'] else '✗ 未修复'} | "
            f"{last.get('changes_made', '')[:max_chars]}"
        )

    def all_failed(self) -> bool:
        """所有修补轮次是否均未修复。"""
        if not self.entries:
            return True
        return all(not e.get("fixed", False) for e in self.entries)


# ═══════════════════════════════════════════════════════════════
# ThresholdRecommendation
# ═══════════════════════════════════════════════════════════════

@dataclass
class ThresholdRecommendation:
    """
    阈值分析推荐结果。

    Attributes:
        recommended: 推荐阈值（已钳制到 [min, max] 范围内）。
        confidence: 置信度 — "high" (≥50样本) / "medium" (≥20) / "low" (≥10) / "insufficient" (<10)。
        sample_count: 有效样本数。
        distribution: 分位数分布 {min, p25, p50, p75, p90, max}。
        is_active: 校准是否激活（样本不足时为 False）。
        fallback_reason: 未激活或钳制的原因。
    """

    recommended: float = 0.7
    confidence: str = "insufficient"
    sample_count: int = 0
    distribution: dict = field(default_factory=dict)
    is_active: bool = False
    fallback_reason: str = ""


# ═══════════════════════════════════════════════════════════════
# ThresholdAnalyzer
# ═══════════════════════════════════════════════════════════════

class ThresholdAnalyzer:
    """
    基于历史 REPAIR 事件相似度分布的阈值分析器。

    从 RepairLog 条目或 AuditLogReader 提取的 REPAIR 事件中计算
    convergence_score 的 P75 分位数作为推荐阈值，并提供置信度评估。

    设计原则：
        - 纯诊断：只输出推荐值，不修改系统状态
        - 安全边界：推荐值自动钳制到 [min_threshold, max_threshold]
        - 样本门控：不足 MIN_SAMPLES 条时标记为 "insufficient"

    使用方式：
        analyzer = ThresholdAnalyzer()
        recommendation = analyzer.analyze(repair_log.entries)
        print(f"推荐阈值: {recommendation.recommended} "
              f"(置信度: {recommendation.confidence}, "
              f"样本: {recommendation.sample_count})")

        # 将推荐值写入 YAML 配置，或注入 ThresholdAdapter
        if recommendation.is_active:
            adapter = ThresholdAdapter()
            adapter.update(recommendation)
            cd = ConvergenceDetector(threshold_adapter=adapter)
    """

    # 最小样本数 — 低于此值时校准不激活
    MIN_SAMPLES: int = 10
    # 安全边界（类常量，可通过 __init__ 覆盖）
    DEFAULT_MIN_THRESHOLD: float = 0.6
    DEFAULT_MAX_THRESHOLD: float = 0.85

    def __init__(
        self,
        min_threshold: float = 0.6,
        max_threshold: float = 0.85,
    ):
        """
        Args:
            min_threshold: 推荐阈值下界（防止阈值过低导致误判）。
            max_threshold: 推荐阈值上界（防止阈值过高导致漏判）。
        """
        if not 0.0 <= min_threshold < max_threshold <= 1.0:
            raise ValueError(
                f"阈值范围非法: [{min_threshold}, {max_threshold}]，"
                f"需满足 0.0 ≤ min < max ≤ 1.0"
            )
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    # ── 主入口 ──────────────────────────────────

    def analyze(
        self,
        repair_entries: list[dict],
    ) -> ThresholdRecommendation:
        """
        分析修补事件列表，输出阈值推荐。

        仅使用 convergence_score > 0 的条目（排除首次修补，
        因为首次无历史可比较，分数为 0.0）。

        Args:
            repair_entries: RepairLog 条目列表，每项至少含:
                - convergence_score: float — 连续修补的相似度分数

        Returns:
            ThresholdRecommendation — 含推荐阈值、置信度、分布。
        """
        # 提取有效分数（排除 0.0 —— 即首次修补或无可比数据）
        scores = [
            e["convergence_score"]
            for e in repair_entries
            if e.get("convergence_score", 0) > 0.0
        ]

        n = len(scores)

        # 样本不足 → 校准未激活
        if n < self.MIN_SAMPLES:
            return ThresholdRecommendation(
                recommended=0.7,
                confidence="insufficient",
                sample_count=n,
                distribution={},
                is_active=False,
                fallback_reason=(
                    f"样本不足（{n} < {self.MIN_SAMPLES}），"
                    f"需至少 {self.MIN_SAMPLES} 条有效修补记录"
                ),
            )

        # 计算分位数分布
        sorted_scores = sorted(scores)
        dist = self._compute_distribution(sorted_scores)
        p75 = dist["p75"]

        # 钳制到安全边界
        recommended = round(
            max(self.min_threshold, min(self.max_threshold, p75)), 3
        )
        clamped = recommended != round(p75, 3)

        # 置信度评估
        if n >= 50:
            confidence = "high"
        elif n >= 20:
            confidence = "medium"
        else:
            confidence = "low"

        return ThresholdRecommendation(
            recommended=recommended,
            confidence=confidence,
            sample_count=n,
            distribution=dist,
            is_active=True,
            fallback_reason=(
                f"推荐值已钳制: P75={p75:.3f} → {recommended}"
                if clamped else ""
            ),
        )

    # ── 内部分位数计算 ──────────────────────────

    @staticmethod
    def _compute_distribution(sorted_scores: list[float]) -> dict:
        """
        计算分位数分布。

        使用线性插值法（与 numpy.percentile 的 linear 模式一致）。

        Args:
            sorted_scores: 已排序的分数列表。

        Returns:
            {min, p25, p50, p75, p90, max} — 所有值保留 3 位小数。
        """
        n = len(sorted_scores)

        def pct(p: float) -> float:
            """第 p 百分位数（0-100）。"""
            k = (n - 1) * p / 100.0
            f = int(k)
            c = k - f
            if f + 1 < n:
                return sorted_scores[f] + c * (
                    sorted_scores[f + 1] - sorted_scores[f]
                )
            return sorted_scores[f]

        return {
            "min": round(sorted_scores[0], 3),
            "p25": round(pct(25), 3),
            "p50": round(pct(50), 3),
            "p75": round(pct(75), 3),
            "p90": round(pct(90), 3),
            "max": round(sorted_scores[-1], 3),
        }


# ═══════════════════════════════════════════════════════════════
# ThresholdAdapter
# ═══════════════════════════════════════════════════════════════

class ThresholdAdapter:
    """
    阈值适配器 — DI 注入 ConvergenceDetector。

    封装阈值来源（固定值 / 校准推荐），向 ConvergenceDetector
    暴露统一的 get_threshold() 接口。支持运行时更新推荐值。

    设计原则：
        - 状态机零修改：ConvergenceDetector 通过 threshold_adapter
          参数接收，check_convergence() 对外接口不变
        - 安全边界：get_threshold() 返回值始终在 [MIN, MAX] 内
        - 透明回退：未校准时自动使用 fallback 值

    使用方式：
        # 方式 1：固定阈值（向后兼容，无需 adapter）
        cd = ConvergenceDetector(threshold=0.7)

        # 方式 2：DI 注入（校准后自动调节）
        adapter = ThresholdAdapter(fallback=0.7)
        cd = ConvergenceDetector(threshold_adapter=adapter)

        # 运行时更新阈值
        recommendation = analyzer.analyze(repair_log.entries)
        adapter.update(recommendation)
        # 下一次 cd.check_convergence() 自动使用新阈值
    """

    DEFAULT_THRESHOLD: float = 0.7
    MIN_THRESHOLD: float = 0.6
    MAX_THRESHOLD: float = 0.85

    def __init__(self, fallback: float = 0.7):
        """
        Args:
            fallback: 回退阈值（校准未激活时使用）。
        """
        self.fallback = max(
            self.MIN_THRESHOLD,
            min(self.MAX_THRESHOLD, fallback),
        )
        self._recommendation: ThresholdRecommendation | None = None
        self._update_count: int = 0

    # ── 阈值查询 ──────────────────────────────────

    def get_threshold(self) -> float:
        """
        返回当前有效阈值。

        优先级：校准推荐（is_active=True）→ fallback。

        Returns:
            float: 阈值，始终在 [MIN_THRESHOLD, MAX_THRESHOLD] 范围内。
        """
        if self._recommendation is not None and self._recommendation.is_active:
            value = self._recommendation.recommended
            # 二次钳制（防御性编程）
            return max(
                self.MIN_THRESHOLD,
                min(self.MAX_THRESHOLD, value),
            )
        return self.fallback

    # ── 校准更新 ──────────────────────────────────

    def update(self, recommendation: ThresholdRecommendation) -> None:
        """
        更新校准推荐值。

        传入 is_active=False 的推荐不会影响 get_threshold()——
        此时继续使用 fallback。

        Args:
            recommendation: ThresholdAnalyzer.analyze() 的输出。
        """
        self._recommendation = recommendation
        self._update_count += 1
        if recommendation.is_active:
            logger.info(
                "[ThresholdAdapter] 阈值已校准: %.3f (置信度=%s, 样本=%d)",
                recommendation.recommended,
                recommendation.confidence,
                recommendation.sample_count,
            )
        else:
            logger.debug(
                "[ThresholdAdapter] 收到推荐但未激活: %s",
                recommendation.fallback_reason,
            )

    def reset(self) -> None:
        """清除校准状态，恢复为 fallback。"""
        self._recommendation = None
        logger.debug("[ThresholdAdapter] 已重置为 fallback=%.3f", self.fallback)

    # ── 查询 ──────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        """当前是否使用校准阈值（而非 fallback）。"""
        return (
            self._recommendation is not None
            and self._recommendation.is_active
        )

    @property
    def current_threshold(self) -> float:
        """get_threshold() 的便捷属性。"""
        return self.get_threshold()

    @property
    def recommendation(self) -> ThresholdRecommendation | None:
        """最近一次校准推荐（可能未激活）。"""
        return self._recommendation
