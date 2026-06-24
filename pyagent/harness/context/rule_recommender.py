"""
高危组合规则推荐引擎 — 从审计日志中识别过度封堵规则，生成白名单推荐。

核心能力：
    - 频次异常检测：同一 rule_id 在窗口内 BLOCK 次数超过阈值时触发分析
    - 指纹聚类：按 plan_step_fingerprint 分组，识别被反复封堵的特定操作模式
    - 事故关联：检查同一 trace_id 是否存在 REPAIR 事件（真实安全事故），
      无事故的频繁 BLOCK 视为「疑似过度封堵」
    - 推荐输出：生成 RuleRecommendation，写入 .claude/rule_recommendations.json

设计原则：
    - 纯分析模块，不修改 permission.py 或 HIGH_RISK_COMBOS
    - 所有推荐标记为 review_for_whitelist，绝不自动应用
    - 依赖 AuditLogReader 读取事件（组合，非继承）
    - 使用 SecurityAuditEvent 已有字段，不引入新的事件类型

使用方式：
    from pyagent.harness.context.rule_recommender import RuleRecommender

    recommender = RuleRecommender()
    report = recommender.analyze(events)
    if report.recommendations:
        recommender.save_report(report, ".claude/rule_recommendations.json")
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .observability import SecurityAuditEvent

logger = logging.getLogger("pyagent.harness.rule_recommender")


# ═══════════════════════════════════════════════════════════════
# 推荐阈值常量
# ═══════════════════════════════════════════════════════════════

# 同一 rule_id 至少产生此数量的 BLOCK 事件才进入分析
DEFAULT_MIN_BLOCK_COUNT = 5
# 同一 fingerprint 至少出现此次数才列为热点指纹
DEFAULT_MIN_FINGERPRINT_COUNT = 3
# 默认推荐输出路径
DEFAULT_RECOMMENDATIONS_PATH = ".claude/rule_recommendations.json"


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

class FingerprintHotspot(BaseModel):
    """
    被反复封堵的特定操作指纹热点。

    Attributes:
        fingerprint: Plan 步骤指纹（SHA256[:16]）。
        count: 该指纹在窗口内的 BLOCK 次数。
        tool_params_example: 脱敏后的工具参数示例（首次出现）。
    """

    fingerprint: str = Field(
        ...,
        description="Plan 步骤指纹 = hash(frozenset(params.items()))[:16]。",
    )
    count: int = Field(
        ...,
        ge=0,
        description="该指纹的 BLOCK 次数。",
    )
    tool_params_example: dict[str, Any] = Field(
        default_factory=dict,
        description="脱敏后的工具参数示例（首次出现时的参数）。",
    )


class RuleRecommendation(BaseModel):
    """
    单条规则推荐 — 描述一条可能过度封堵的规则。

    Attributes:
        rule_id: 匹配的门控规则 ID（如 HIGH_RISK_COMBOS[0]）。
        combo: 触发的高危工具组合（去重排序）。
        block_count: 总 BLOCK 次数。
        unique_fingerprints: 独立指纹数（操作多样性指标）。
        top_fingerprints: 高频封堵指纹热点列表。
        avg_risk_score: 平均风险评分。
        has_related_incidents: 是否有关联安全事故（REPAIR）。
        recommendation: 推荐动作（固定为 review_for_whitelist）。
        reason: 推荐理由（人类可读）。
    """

    rule_id: str = Field(
        ...,
        description="门控规则 ID，如 'HIGH_RISK_COMBOS[0]'。",
    )
    combo: list[str] = Field(
        default_factory=list,
        description="触发的高危工具组合。",
    )
    block_count: int = Field(
        ...,
        ge=0,
        description="总 BLOCK 次数。",
    )
    unique_fingerprints: int = Field(
        default=0,
        ge=0,
        description="独立操作指纹数。",
    )
    top_fingerprints: list[FingerprintHotspot] = Field(
        default_factory=list,
        description="高频封堵指纹热点（按次数降序）。",
    )
    avg_risk_score: float = Field(
        default=0.0,
        ge=0.0, le=100.0,
        description="该规则下 BLOCK 事件的平均风险评分。",
    )
    has_related_incidents: bool = Field(
        default=False,
        description="是否存在关联安全事故（同一 trace_id 的 REPAIR 事件）。",
    )
    recommendation: str = Field(
        default="review_for_whitelist",
        description="推荐动作。固定为 review_for_whitelist。",
    )
    reason: str = Field(
        default="",
        description="推荐理由（人类可读）。",
    )


class RecommendationReport(BaseModel):
    """
    规则推荐报告 — 一次分析的完整输出。

    Attributes:
        generated_at: 报告生成时间（ISO 8601 UTC）。
        source_file: 分析的审计日志文件路径。
        analysis_window: 分析窗口 {start, end} ISO 时间。
        total_events_analyzed: 分析的事件总数。
        total_block_events: BLOCK 事件总数。
        recommendations: 推荐列表（按 block_count 降序）。
    """

    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="报告生成时间（ISO 8601 UTC）。",
    )
    source_file: str = Field(
        default="",
        description="分析的审计日志文件路径。",
    )
    analysis_window: dict[str, str] = Field(
        default_factory=dict,
        description="分析窗口 {start, end} ISO 时间。",
    )
    total_events_analyzed: int = Field(
        default=0,
        ge=0,
        description="分析的事件总数。",
    )
    total_block_events: int = Field(
        default=0,
        ge=0,
        description="BLOCK 事件总数。",
    )
    recommendations: list[RuleRecommendation] = Field(
        default_factory=list,
        description="推荐列表（按 block_count 降序）。",
    )


# ═══════════════════════════════════════════════════════════════
# RuleRecommender
# ═══════════════════════════════════════════════════════════════

class RuleRecommender:
    """
    规则推荐引擎 — 从审计事件中识别过度封堵规则。

    分析流程：
        1. 收集所有 BLOCK 事件，按 rule_id 分组
        2. 过滤：同一 rule_id 的 BLOCK 次数 < min_block_count 则跳过
        3. 事故关联：检查同一 rule_id 涉及的 trace_id 中是否存在 REPAIR 事件
           （真实安全事故会抑制推荐）
        4. 指纹聚类：按 plan_step_fingerprint 分组，识别高频热点
        5. 生成推荐：为每条超标规则生成 RuleRecommendation

    使用方式：
        recommender = RuleRecommender(min_block_count=10)
        report = recommender.analyze(events)
        recommender.save_report(report)
    """

    def __init__(
        self,
        min_block_count: int = DEFAULT_MIN_BLOCK_COUNT,
        min_fingerprint_count: int = DEFAULT_MIN_FINGERPRINT_COUNT,
    ):
        """
        Args:
            min_block_count: 同一 rule_id 至少产生此数量的 BLOCK 事件
                            才进入分析。默认 5。
            min_fingerprint_count: 同一 fingerprint 至少出现此次数
                                   才列入热点列表。默认 3。
        """
        self.min_block_count = min_block_count
        self.min_fingerprint_count = min_fingerprint_count

    # ── 主入口 ────────────────────────────────────

    def analyze(
        self,
        events: list[SecurityAuditEvent],
        source_file: str = "",
    ) -> RecommendationReport:
        """
        分析审计事件，生成规则推荐报告。

        Args:
            events: SecurityAuditEvent 列表（可由 AuditLogReader 提供）。
            source_file: 事件来源文件路径（写入报告的 source_file 字段）。

        Returns:
            RecommendationReport。无推荐时 recommendations 为空列表。
        """
        if not events:
            return RecommendationReport(
                source_file=source_file,
                total_events_analyzed=0,
                total_block_events=0,
            )

        # ── 1. 分类事件 ──
        block_events: list[SecurityAuditEvent] = []
        repair_trace_ids: set[str] = set()
        time_min: str = ""
        time_max: str = ""

        for event in events:
            # 时间窗口
            if event.timestamp:
                if not time_min or event.timestamp < time_min:
                    time_min = event.timestamp
                if not time_max or event.timestamp > time_max:
                    time_max = event.timestamp

            # BLOCK 事件
            if event.decision == "BLOCK":
                block_events.append(event)

            # REPAIR 事件 → 收集关联 trace_id（安全事故标记）
            if event.decision == "REPAIR" and event.trace_id:
                repair_trace_ids.add(event.trace_id)

        if not block_events:
            return RecommendationReport(
                source_file=source_file,
                analysis_window={"start": time_min, "end": time_max},
                total_events_analyzed=len(events),
                total_block_events=0,
            )

        # ── 2. 按 rule_id 分组 BLOCK 事件 ──
        by_rule: dict[str, list[SecurityAuditEvent]] = defaultdict(list)
        for event in block_events:
            by_rule[event.rule_id].append(event)

        # ── 3. 逐规则分析 ──
        recommendations: list[RuleRecommendation] = []

        for rule_id, rule_events in sorted(by_rule.items()):
            if len(rule_events) < self.min_block_count:
                continue

            rec = self._analyze_rule(
                rule_id=rule_id,
                rule_events=rule_events,
                repair_trace_ids=repair_trace_ids,
            )
            if rec is not None:
                recommendations.append(rec)

        # 按 block_count 降序排列
        recommendations.sort(key=lambda r: r.block_count, reverse=True)

        return RecommendationReport(
            source_file=source_file,
            analysis_window={"start": time_min, "end": time_max},
            total_events_analyzed=len(events),
            total_block_events=len(block_events),
            recommendations=recommendations,
        )

    # ── 单规则分析 ─────────────────────────────────

    def _analyze_rule(
        self,
        rule_id: str,
        rule_events: list[SecurityAuditEvent],
        repair_trace_ids: set[str],
    ) -> Optional[RuleRecommendation]:
        """
        对单条规则的 BLOCK 事件进行深度分析。

        Args:
            rule_id: 规则 ID。
            rule_events: 该规则的所有 BLOCK 事件。
            repair_trace_ids: 全部 REPAIR 事件的 trace_id 集合。

        Returns:
            RuleRecommendation 或 None（不满足推荐条件时）。
        """
        # ── 工具组合（从事件 tool_name 去重聚合） ──
        tools: set[str] = set()
        for e in rule_events:
            if e.tool_name:
                # tool_name 可能包含多个工具（逗号分隔），如 "write_file, execute_python"
                for t in e.tool_name.split(","):
                    t = t.strip()
                    if t:
                        tools.add(t)

        # ── 事故关联检查 ──
        # 同一 trace_id 下存在 REPAIR 事件 → 真实安全事故
        related_trace_ids: set[str] = set()
        for e in rule_events:
            if e.trace_id and e.trace_id in repair_trace_ids:
                related_trace_ids.add(e.trace_id)

        has_incidents = len(related_trace_ids) > 0

        # 如果所有 BLOCK 都关联了安全事故，不推荐白名单
        incident_block_count = sum(
            1 for e in rule_events
            if e.trace_id in repair_trace_ids
        )
        if incident_block_count == len(rule_events):
            logger.debug(
                "[RuleRecommender] 规则 %s: 全部 %d 次 BLOCK 均关联安全事故，跳过",
                rule_id, len(rule_events),
            )
            return None

        # ── 指纹聚类 ──
        fp_counts: dict[str, int] = defaultdict(int)
        fp_examples: dict[str, dict[str, Any]] = {}
        risk_scores: list[int] = []

        for e in rule_events:
            risk_scores.append(e.risk_score)
            fp = e.plan_step_fingerprint or "no_params"
            fp_counts[fp] += 1
            if fp not in fp_examples and e.tool_params:
                fp_examples[fp] = dict(e.tool_params)

        # 热点指纹（≥ min_fingerprint_count）
        hotspots = sorted(
            [
                FingerprintHotspot(
                    fingerprint=fp,
                    count=cnt,
                    tool_params_example=fp_examples.get(fp, {}),
                )
                for fp, cnt in fp_counts.items()
                if cnt >= self.min_fingerprint_count
            ],
            key=lambda h: h.count,
            reverse=True,
        )

        avg_risk = round(sum(risk_scores) / len(risk_scores), 1) if risk_scores else 0.0

        # ── 构件推荐理由 ──
        reason_parts = [
            f"{len(rule_events)} 次 BLOCK 决策",
        ]
        if has_incidents:
            reason_parts.append(
                f"{incident_block_count} 次关联安全事故（{len(related_trace_ids)} 个独立 trace）"
            )
        else:
            reason_parts.append("无关联安全事故")
        reason_parts.append(f"{len(fp_counts)} 种不同操作指纹")
        reason_parts.append(f"平均风险评分 {avg_risk}")
        if hotspots:
            reason_parts.append(
                f"{len(hotspots)} 个高频热点指纹（≥{self.min_fingerprint_count} 次）"
            )

        full_reason = (
            f"{'，'.join(reason_parts)}。"
            f"建议人工审查是否可对特定参数模式加白。"
        )

        return RuleRecommendation(
            rule_id=rule_id,
            combo=sorted(tools),
            block_count=len(rule_events),
            unique_fingerprints=len(fp_counts),
            top_fingerprints=hotspots,
            avg_risk_score=avg_risk,
            has_related_incidents=has_incidents,
            recommendation="review_for_whitelist",
            reason=full_reason,
        )

    # ── 报告持久化 ─────────────────────────────────

    @staticmethod
    def save_report(
        report: RecommendationReport,
        path: str = DEFAULT_RECOMMENDATIONS_PATH,
    ) -> bool:
        """
        将推荐报告写入 JSON 文件。

        Args:
            report: RecommendationReport 实例。
            path: 输出文件路径。默认 .claude/rule_recommendations.json。

        Returns:
            True 如果写入成功。
        """
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report.model_dump_json(indent=2, ensure_ascii=False))
            logger.info(
                "[RuleRecommender] 报告已写入 %s: %d 条推荐",
                path, len(report.recommendations),
            )
            return True
        except OSError as e:
            logger.error("[RuleRecommender] 报告写入失败: %s — %s", path, e)
            return False

    @staticmethod
    def load_report(
        path: str = DEFAULT_RECOMMENDATIONS_PATH,
    ) -> Optional[RecommendationReport]:
        """
        从 JSON 文件加载推荐报告。

        Args:
            path: 报告文件路径。

        Returns:
            RecommendationReport 或 None（文件不存在 / 解析失败）。
        """
        file_path = Path(path)
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return RecommendationReport(**data)
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning(
                "[RuleRecommender] 报告加载失败: %s — %s",
                path, str(e)[:120],
            )
            return None

    @staticmethod
    def delete_report(path: str = DEFAULT_RECOMMENDATIONS_PATH) -> bool:
        """
        删除推荐报告文件。

        Args:
            path: 报告文件路径。

        Returns:
            True 如果已删除或不存在。
        """
        file_path = Path(path)
        try:
            if file_path.exists():
                file_path.unlink()
            return True
        except OSError as e:
            logger.error("[RuleRecommender] 报告删除失败: %s — %s", path, e)
            return False
