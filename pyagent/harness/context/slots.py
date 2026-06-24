"""
槽位预算分配器。

根据 PEVR 阶段和总 Token 预算，为四槽位分配独立 Token 配额。

各阶段分配权重不同：
    plan:    System 40% | Plan 30% | History 0% | WM 30%
    execute: System 15% | Plan 20% | History 40% | WM 25%
    verify:  System 10% | Plan 25% | History 10% | WM 55%
    repair:  System 10% | Plan 15% | History 20% | WM 55%
"""

from .models import SlotSet, SlotContent


# ── 各阶段槽位预算权重 ─────────────────────────

PHASE_BUDGETS: dict[str, dict[str, float]] = {
    "plan": {
        "system": 0.38,
        "plan": 0.28,
        "history": 0.00,
        "working_memory": 0.28,
        "observability_hints": 0.06,
    },
    "execute": {
        "system": 0.14,
        "plan": 0.18,
        "history": 0.38,
        "working_memory": 0.23,
        "observability_hints": 0.07,
    },
    "verify": {
        "system": 0.09,
        "plan": 0.23,
        "history": 0.09,
        "working_memory": 0.52,
        "observability_hints": 0.07,
    },
    "repair": {
        "system": 0.08,
        "plan": 0.13,
        "history": 0.17,
        "working_memory": 0.52,
        "observability_hints": 0.10,
    },
}


def allocate_budgets(
    slots: SlotSet,
    total_budget: int,
    phase: str,
) -> SlotSet:
    """
    根据阶段为四槽位分配 Token 预算。

    Args:
        slots: 初始 SlotSet（已填充内容，max_tokens 待分配）。
        total_budget: 总可用 Token 数。
        phase: PEVR 阶段名（plan/execute/verify/repair）。

    Returns:
        已分配 max_tokens 的 SlotSet 副本。
    """
    ratios = PHASE_BUDGETS.get(phase, PHASE_BUDGETS["execute"])

    result = slots.model_copy()
    for slot_name, ratio in ratios.items():
        slot = getattr(result, slot_name)
        slot.max_tokens = int(total_budget * ratio)

    return result


def build_slots(
    system_prompt: str = "",
    context_files: list[str] | None = None,
    plan: str = "",
    history_text: str = "",
    working_memory_artifacts: dict[str, str] | None = None,
    step_results_text: str = "",
    observability_hints_text: str = "",
) -> SlotSet:
    """
    从原始素材构建五槽位初始内容。

    此函数负责将分散的输入聚合为 SlotSet，不涉及 Token 预算。

    Args:
        system_prompt: Agent 系统提示词。
        context_files: 已加载的规范文件内容列表（含来源标记）。
        plan: 当前计划快照文本。
        history_text: 对话历史文本。
        working_memory_artifacts: 工作记忆产物 dict。
        step_results_text: 步骤执行记录文本。
        observability_hints_text: 1.5.4 观测提示文本（修复建议、工具白名单等）。

    Returns:
        填充了内容的 SlotSet。
    """
    # System 槽位：系统指令 + 规范文件
    system_content = system_prompt or ""
    if context_files:
        system_content += "\n\n## 项目规范\n" + "\n\n".join(context_files)

    # WorkingMemory 槽位：产物 + 步骤记录
    wm_content = ""
    if working_memory_artifacts:
        for key, value in working_memory_artifacts.items():
            wm_content += f"### {key}\n{value}\n\n"
    if step_results_text:
        wm_content += f"\n## 执行记录\n{step_results_text}"

    return SlotSet(
        system=SlotContent(name="system", content=system_content, priority=10),
        plan=SlotContent(name="plan", content=plan, priority=7),
        history=SlotContent(name="history", content=history_text, priority=5),
        working_memory=SlotContent(
            name="working_memory",
            content=wm_content.strip(),
            priority=3,
        ),
        observability_hints=SlotContent(
            name="observability_hints",
            content=observability_hints_text,
            priority=2,
        ),
    )
