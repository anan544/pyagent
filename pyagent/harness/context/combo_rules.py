"""
高危组合精炼规则引擎 — 字段级匹配（优化 2）。

ComboRuleEngine 在快路径（set-based HIGH_RISK_COMBOS）命中后执行精炼过滤：
    - 若 combo 有对应的 ComboRule 且定义了 match_on 字段，
      则要求前后操作的指定字段值相同才触发拦截。
    - 若 match_on 不匹配 → 放行（降低误报）。
    - 若 combo 无对应规则 → 维持快路径判断（拦截）。

典型场景：
    快路径: {write_file, execute_python} ⊆ recent_set → 命中！
    精炼: ComboRule(sequence=[write_file, execute_python], match_on="file_path")
    → write_file("log.txt") + execute_python("data_process.py"):
       file_path 不同 → 放行 ✓（合法工作流）
    → write_file("a.py") + execute_python("a.py"):
       file_path 相同 → 拦截 ✗（疑似恶意）

使用方式：
    engine = ComboRuleEngine([
        ComboRule(name="w_e_same_file", sequence=["write_file", "execute_python"],
                  match_on="file_path", within_window=5),
    ])
    result = engine.match(
        combo={"write_file", "execute_python"},
        current_tool="execute_python",
        current_params={"code": "..."},
        recent_records=[...],
    )
    # → "w_e_same_file"（命中）, False（不匹配）, 或 None（无规则）
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("pyagent.security.combo_rules")


# ═══════════════════════════════════════════════════════════════
# ComboRule 模型
# ═══════════════════════════════════════════════════════════════

class ComboRule(BaseModel):
    """单条字段级高危组合规则。

    Attributes:
        name: 规则名称（如 "write_then_exec_same_file"）。
        sequence: 触发高危组合的工具序列（顺序无关，使用 set 匹配）。
        match_on: 要求匹配的参数字段名（如 "file_path"）。
                  None 表示仅依赖快路径（无精炼过滤）。
        within_window: 在最近 N 次调用内匹配。默认 5。
    """

    name: str = Field(
        ...,
        min_length=1,
        description="规则名称，用于审计日志。",
    )
    sequence: list[str] = Field(
        ...,
        min_length=2,
        description="触发高危组合的工具序列。",
    )
    match_on: Optional[str] = Field(
        default=None,
        description="要求匹配的参数字段名。如 'file_path' 要求前后操作操作同一文件。",
    )
    within_window: int = Field(
        default=5,
        ge=1,
        le=50,
        description="在最近 N 次调用内查找匹配。",
    )


# ═══════════════════════════════════════════════════════════════
# ComboRuleEngine
# ═══════════════════════════════════════════════════════════════

class ComboRuleEngine:
    """字段级组合规则引擎。

    双层匹配的第二层：快路径命中后，检查是否有精炼规则，
    若有则要求字段级匹配通过才拦截。

    索引结构：{(frozenset(sequence), match_on): rule_name}
    确保 O(1) 查找。
    """

    def __init__(self, rules: Optional[list[ComboRule]] = None):
        """
        Args:
            rules: ComboRule 列表。空列表或 None 表示无精炼规则。
        """
        self._rules: list[ComboRule] = list(rules) if rules else []
        # 索引: {(frozenset(tool_names), match_on): rule_index}
        self._index: dict[tuple, int] = {}
        for i, rule in enumerate(self._rules):
            key = (frozenset(rule.sequence), rule.match_on)
            self._index[key] = i

    def match(
        self,
        combo: set[str],
        current_tool: str,
        current_params: dict,
        recent_records: list[dict],
    ) -> Optional[str | bool]:
        """精炼匹配检查。

        Args:
            combo: 快路径命中的工具组合（set）。
            current_tool: 当前即将调用的工具名。
            current_params: 当前工具的参数。
            recent_records: SessionRiskContext.recent_records() 返回的历史记录。

        Returns:
            - str (rule name): 精炼匹配通过，应拦截。
            - False: 精炼匹配不通过，应放行。
            - None: 无对应的精炼规则，维持快路径判断（拦截）。
        """
        # 查找匹配的规则
        matched_rule: Optional[ComboRule] = None
        for rule in self._rules:
            rule_combo = frozenset(rule.sequence)
            if rule_combo == combo or rule_combo.issubset(combo):
                matched_rule = rule
                break

        if matched_rule is None:
            # 无精炼规则 → 维持快路径判断
            return None

        if matched_rule.match_on is None:
            # 规则存在但无 match_on → 命中（等同于快路径）
            return matched_rule.name

        # ── 字段级匹配 ──
        match_field = matched_rule.match_on

        # 提取当前调用的字段值
        current_value = self._extract_field(current_params, match_field)
        if current_value is None:
            # 当前调用缺少 match_on 字段 → 无法精炼，维持快路径判断
            logger.debug(
                "[ComboRuleEngine] 当前工具 '%s' 缺少 match_on 字段 '%s'，维持快路径",
                current_tool, match_field,
            )
            return None

        # 在窗口内查找匹配的其他工具
        window = matched_rule.within_window
        recent_slice = recent_records[-window:] if len(recent_records) > window else recent_records

        other_tools_in_combo = combo - {current_tool}
        for record in recent_slice:
            rec_tool = record.get("tool_name", "")
            if rec_tool not in other_tools_in_combo:
                continue
            rec_summary = record.get("params_summary", {})
            rec_value = ComboRuleEngine._extract_field(rec_summary, match_field)
            if rec_value and rec_value == current_value:
                # 字段匹配 → 精炼通过，应拦截
                logger.debug(
                    "[ComboRuleEngine] '%s' 精炼命中: %s=%s 匹配 (工具: %s + %s)",
                    matched_rule.name, match_field, current_value,
                    current_tool, rec_tool,
                )
                return matched_rule.name

        # 字段不匹配 → 精炼拒绝，应放行
        logger.debug(
            "[ComboRuleEngine] '%s' 精炼不匹配: %s=%s 无对应记录",
            matched_rule.name, match_field, current_value,
        )
        return False

    @staticmethod
    def _extract_field(params: dict, field: str) -> Optional[str]:
        """从参数中提取指定字段值（支持别名）。

        file_path 字段：尝试 "file_path", "path", "filename"。
        command 字段：尝试 "command", "cmd"。
        """
        # 直接匹配
        if field in params:
            return str(params[field])

        # 别名表
        aliases = {
            "file_path": ["path", "filename", "file"],
            "command": ["cmd"],
            "url": ["uri", "link"],
        }
        for alias in aliases.get(field, []):
            if alias in params:
                return str(params[alias])

        return None

    @property
    def rule_count(self) -> int:
        """已注册的规则数量。"""
        return len(self._rules)
