"""
优先级感知的 Token 裁剪器。

当四槽位总 Token 超限时，按优先级降级：
    1. 保留最近 N 轮交互
    2. 保留计划关键节点
    3. 压缩早期工具输出
    4. 截断无关规范片段

与现有 TokenBudget 的区别：
    TokenBudget 是粗粒度的"压缩区/精确区"二分裁剪。
    本模块是细粒度的分层裁剪，区分消息类型和阶段相关性。
"""

import re
from typing import Optional
from .models import SlotContent, SlotSet


# 最大 Token 预算（字符估算）
DEFAULT_MAX_TOKENS = 64000  # 约 64K tokens
TOKEN_ESTIMATE_RATIO = 4  # 字符/Token


def estimate_tokens(text: str) -> int:
    """简单 Token 估算（字符 / 4）。"""
    return max(1, len(text) // TOKEN_ESTIMATE_RATIO)


def trim_slots(
    slots: SlotSet,
    max_total_tokens: int = DEFAULT_MAX_TOKENS,
    phase: Optional[str] = None,
) -> SlotSet:
    """
    对四槽位执行优先级裁剪。

    裁剪顺序（先用低优先级尝试）：
        1. WorkingMemory（priority=3）— 截断旧产物
        2. History（priority=5）— 保留最近 N 轮
        3. Plan（priority=7）— 保留关键节点
        4. System（priority=10）— 最后才裁

    Args:
        slots: 原始四槽位。
        max_total_tokens: 总 Token 上限。
        phase: 当前 PEVR 阶段，影响裁剪策略。

    Returns:
        裁剪后的 SlotSet（不修改原对象）。
    """
    current = slots.total_tokens()
    if current <= max_total_tokens:
        return slots

    # 计算需要削减的 Token
    excess = current - max_total_tokens

    # 按优先级升序裁剪（低优先级先裁）— 1.5.4 五槽位
    slot_names = ["observability_hints", "working_memory", "history", "plan", "system"]
    trimmed = SlotSet(
        system=slots.system.model_copy(),
        plan=slots.plan.model_copy(),
        history=slots.history.model_copy(),
        working_memory=slots.working_memory.model_copy(),
        observability_hints=slots.observability_hints.model_copy(),
    )

    for name in slot_names:
        if excess <= 0:
            break
        slot = getattr(trimmed, name)
        if not slot.content:
            continue
        excess = _trim_slot(slot, excess, phase)

    return trimmed


def _trim_slot(slot: SlotContent, excess: int, phase: Optional[str] = None) -> int:
    """对单个槽位执行裁剪，返回剩余超限 Token 数。"""
    if slot.name == "history":
        return _trim_history(slot, excess)
    else:
        return _trim_generic(slot, excess)


def _trim_history(slot: SlotContent, excess: int) -> int:
    """
    裁剪历史槽位：保留最近 N 轮交互。

    历史格式假设为多段落，每段以换行分隔。
    从最早的消息开始丢弃，保留最近的消息。
    """
    if not slot.content:
        return excess

    # 按轮次分割（假设每轮以 "---" 或空行 + 角色标记分隔）
    rounds = _split_rounds(slot.content)
    if len(rounds) <= 1:
        return _trim_generic(slot, excess)

    # 从最早轮次开始丢弃
    removed = 0
    kept: list[str] = []
    for r in reversed(rounds):  # 从最新开始保留
        r_tokens = estimate_tokens(r)
        if excess > 0 and kept:  # 保留最新的一轮
            excess -= r_tokens
            removed += 1
        else:
            kept.append(r)

    if removed > 0:
        slot.content = "\n\n".join(reversed(kept))
        slot.content += f"\n\n[已裁剪 {removed} 轮早期交互]"

    return max(0, excess)


def _trim_generic(slot: SlotContent, excess: int) -> int:
    """通用裁剪：从尾部截断，保留头部。"""
    if not slot.content or excess <= 0:
        return excess

    current_tokens = estimate_tokens(slot.content)
    if current_tokens <= excess:
        return excess

    # 按字符比例截断
    keep_chars = max(
        TOKEN_ESTIMATE_RATIO,  # 最少保留 4 字符
        len(slot.content) - (excess * TOKEN_ESTIMATE_RATIO),
    )
    keep_chars = int(keep_chars)

    if keep_chars >= len(slot.content):
        return excess

    slot.content = slot.content[:keep_chars] + "\n\n[内容已截断]"
    new_tokens = estimate_tokens(slot.content)
    return max(0, excess - (current_tokens - new_tokens))


def _split_rounds(history: str) -> list[str]:
    """
    将历史文本拆分为交互轮次。

    启发式判断：以 User/Assistant 交替为分界。
    """
    # 尝试按 "user:" 或 "User:" 分割（简化版）
    parts = re.split(r"\n(?=[Uu]ser[: ])", history)
    if len(parts) > 1:
        return parts

    # 回退：按双换行分割
    parts = [p.strip() for p in history.split("\n\n") if p.strip()]
    if len(parts) > 1:
        return parts

    return [history]


# ── 阶段感知的规范裁剪 ─────────────────────────────

def trim_context_files(
    files: list[str],
    max_tokens: int,
    phase: str,
) -> list[str]:
    """
    根据阶段裁剪上下文规范文件。

    - plan: 全量保留
    - execute: 按相关度保留片段
    - verify: 仅保留验收标准相关
    - repair: 仅保留相关片段

    Args:
        files: 格式为 ["[来源: path]\ncontent", ...]。
        max_tokens: 规范文件总 Token 上限。
        phase: PEVR 阶段。

    Returns:
        裁剪后的文件列表。
    """
    if phase == "plan":
        # 全量保留，超限时尾部截断
        return _trim_files_by_length(files, max_tokens)

    elif phase in ("execute", "repair"):
        # 每个文件截断到合理大小
        per_file = max_tokens // max(1, len(files))
        return [
            f[: per_file * TOKEN_ESTIMATE_RATIO] + "\n[截断]"
            if estimate_tokens(f) > per_file
            else f
            for f in files
        ]

    elif phase == "verify":
        # 仅保留与验收标准可能相关的（包含关键判断词的文件）
        verify_keywords = ["accept", "criteria", "test", "spec", "require",
                           "should", "must", "验收", "标准", "测试"]
        relevant = []
        for f in files:
            content_lower = f.lower()
            if any(kw in content_lower for kw in verify_keywords):
                relevant.append(f)
        return _trim_files_by_length(relevant, max_tokens)

    return files


def _trim_files_by_length(files: list[str], max_tokens: int) -> list[str]:
    """按 Token 上限裁剪文件列表（保留前面的）。"""
    result = []
    used = 0
    for f in files:
        t = estimate_tokens(f)
        if used + t <= max_tokens:
            result.append(f)
            used += t
        else:
            remaining = max_tokens - used
            if remaining > 100:  # 至少保留有意义的一段
                truncate_chars = remaining * TOKEN_ESTIMATE_RATIO
                result.append(f[:truncate_chars] + "\n[已截断]")
            break
    return result
