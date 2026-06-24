"""
上下文组装器 — 基类 + PEVR 实现。

ContextAssembler 负责将所有碎片信息（系统指令、规范、计划、历史、工作记忆）
组装为 LLM 可消费的标准消息列表。PEVRContextAssembler 使用四槽位模型 +
Jinja2 模板实现阶段感知的动态 Prompt 渲染。

设计原则：
    - 无状态：不持有对话历史，所有数据通过 ContextRequest 传入
    - 模板分离：Prompt 模板存放在 templates/pevr/ 目录下
    - 可测试：纯函数式的 assemble() 方法，输入确定则输出确定
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, BaseLoader, TemplateNotFound

from .models import (
    PEVRPhase,
    ContextRequest,
    ContextResult,
    SlotSet,
    SlotContent,
    InvalidStateError,
)
from .slots import build_slots, allocate_budgets
from .trimmer import trim_slots, trim_context_files, estimate_tokens
from ...core.message import SystemMessage, UserMessage

logger = logging.getLogger("pyagent.harness.context")


# ── 模板环境 ──────────────────────────────────────

_TEMPLATE_DIR = Path(__file__).parent / "templates"

_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _render_template(name: str, context: dict) -> str:
    """渲染 Jinja2 模板，文件不存在时降级。"""
    try:
        template = _jinja_env.get_template(name)
        return template.render(context)
    except TemplateNotFound:
        logger.warning("模板 %s 未找到，使用降级渲染", name)
        return _fallback_render(name, context)


def _fallback_render(name: str, context: dict) -> str:
    """无模板时的降级渲染 — 将槽位内容直接拼接。"""
    slots = context.get("slots", SlotSet())
    phase = context.get("phase", "execute")
    parts = [
        f"# PEVR Phase: {phase}",
        f"## System\n{slots.system.content}" if slots.system.content else "",
        f"## Plan\n{slots.plan.content}" if slots.plan.content else "",
        f"## History\n{slots.history.content}" if slots.history.content else "",
        f"## Working Memory\n{slots.working_memory.content}" if slots.working_memory.content else "",
        f"## Observability Hints\n{slots.observability_hints.content}" if slots.observability_hints.content else "",
    ]
    return "\n\n".join(p for p in parts if p)


# ── 基类 ──────────────────────────────────────────

class ContextAssembler(ABC):
    """
    上下文组装器抽象基类。

    子类实现 assemble() 方法，将 ContextRequest 转换为 ContextResult。
    """

    @abstractmethod
    async def assemble(self, request: ContextRequest) -> ContextResult:
        """
        组装上下文为消息列表。

        Args:
            request: 包含阶段、系统指令、历史、计划等信息的请求对象。

        Returns:
            ContextResult: 组装后的消息列表 + Token 统计。
        """
        ...


# ── PEVR 实现 ─────────────────────────────────────

class PEVRContextAssembler(ContextAssembler):
    """
    PEVR 上下文组装器 — 四槽位 + 阶段感知 + Token 裁剪。

    工作流程：
        1. 从 ContextRequest 提取素材
        2. 构建四槽位初始内容
        3. 按阶段分配 Token 预算
        4. 如需裁剪，按优先级降级
        5. 渲染 Jinja2 模板 → 消息列表

    使用方式：
        assembler = PEVRContextAssembler(total_budget=64000)
        result = await assembler.assemble(request)
        # result.messages 可直接传给 LLM
    """

    # ── 模板文件映射 ──────────────────────────────
    TEMPLATES = {
        PEVRPhase.PLAN: "plan.j2",
        PEVRPhase.EXECUTE: "execute.j2",
        PEVRPhase.VERIFY: "verify.j2",
        PEVRPhase.REPAIR: "repair.j2",
    }

    def __init__(self, total_budget: int = 64000):
        """
        Args:
            total_budget: 四槽位总 Token 预算上限。
        """
        self.total_budget = total_budget

    # ── 主入口 ────────────────────────────────────

    async def assemble(self, request: ContextRequest) -> ContextResult:
        """
        组装 PEVR 上下文。

        Args:
            request: 上下文请求（含阶段、计划、历史、工作记忆等）。

        Returns:
            ContextResult: 组装好的消息列表。
        """
        phase = request.phase

        # ── 1.5.3: VERIFY 阶段断言 ──
        # 强制从 WorkingMemory.plan 和 WorkingMemory.acceptance_criteria 取值，
        # 不接受外部传入的 criteria 参数，防止静默降级导致误判通过。
        if phase == PEVRPhase.VERIFY:
            if request.working_memory.plan is None:
                raise InvalidStateError(
                    "VERIFY 阶段需要 Plan 快照，但 WorkingMemory.plan 为空。"
                    "这表明 Plan 快照可能已损坏或未正确冻结——请检查 PLANNING 阶段流程。"
                )
            wm_criteria = request.working_memory.acceptance_criteria
            if not wm_criteria or not wm_criteria.strip():
                raise InvalidStateError(
                    "VERIFY 阶段需要验收标准，但 WorkingMemory.acceptance_criteria 为空。"
                    "验收标准应在 run() 入口时存入 WorkingMemory，并在 VERIFY 阶段强制从此读取。"
                )

        # 1. 阶段感知的规范文件裁剪
        trimmed_files = trim_context_files(
            request.context_files,
            max_tokens=self.total_budget // 4,  # 规范最多占 1/4
            phase=phase.value,
        )

        # 2. 构建工作记忆文本
        wm_text = _build_working_memory_text(request)

        # 3. 构建历史文本
        history_text = _build_history_text(request)

        # 4. 构建五槽位（1.5.4: observability_hints）
        slots = build_slots(
            system_prompt=request.system_prompt,
            context_files=trimmed_files,
            plan=request.plan,
            history_text=history_text,
            working_memory_artifacts=request.working_memory.artifacts,
            step_results_text=wm_text,
            observability_hints_text=_build_observability_hints(request),
        )

        # 5. 分配 Token 预算
        slots = allocate_budgets(slots, self.total_budget, phase.value)

        # 6. 裁剪（如需）
        was_trimmed = slots.total_tokens() > self.total_budget
        if was_trimmed:
            slots = trim_slots(slots, self.total_budget, phase.value)

        # 7. 渲染模板
        context = {
            "phase": phase.value,
            "system": {
                "content": slots.system.content,
                "max_tokens": slots.system.max_tokens,
            },
            "plan": {
                "content": slots.plan.content,
                "max_tokens": slots.plan.max_tokens,
            },
            "history": {
                "content": slots.history.content,
                "max_tokens": slots.history.max_tokens,
            },
            "observability_hints": {
                "content": slots.observability_hints.content,
                "max_tokens": slots.observability_hints.max_tokens,
            },
            "working_memory": _working_memory_to_template(request),
            "slots": slots,
            # 1.5.3: VERIFY 阶段强制从 WorkingMemory 读取验收标准
            "acceptance_criteria": (
                request.working_memory.acceptance_criteria
                if phase == PEVRPhase.VERIFY
                else request.acceptance_criteria
            ),
            "failure_summary": request.failure_summary,
            "few_shot_examples": request.few_shot_examples,
            "relevant_specs": _relevant_specs(trimmed_files, phase),
            "context_files": [
                {"source": _extract_source(f), "content": _extract_body(f)}
                for f in trimmed_files
            ],
        }

        template_name = self.TEMPLATES.get(phase, "execute.j2")
        rendered = _render_template(template_name, context)

        # 8. 构建消息列表
        messages = [
            SystemMessage(content=rendered),
            UserMessage(content=self._phase_user_message(phase, request)),
        ]

        # 9. 统计 Token
        slot_tokens = {
            "system": estimate_tokens(slots.system.content),
            "plan": estimate_tokens(slots.plan.content),
            "history": estimate_tokens(slots.history.content),
            "working_memory": estimate_tokens(slots.working_memory.content),
            "observability_hints": estimate_tokens(slots.observability_hints.content),
        }

        return ContextResult(
            messages=messages,
            total_tokens=sum(slot_tokens.values()),
            slot_tokens=slot_tokens,
            was_trimmed=was_trimmed,
            trimmed_details=_trim_detail(slots) if was_trimmed else "",
        )

    # ── 阶段用户消息 ─────────────────────────────

    def _phase_user_message(self, phase: PEVRPhase, request: ContextRequest) -> str:
        """根据阶段生成不同的用户提示前缀。"""
        if phase == PEVRPhase.PLAN:
            return "请制定执行计划。"
        elif phase == PEVRPhase.EXECUTE:
            return "请执行当前步骤。"
        elif phase == PEVRPhase.VERIFY:
            return "请逐条对照验收标准进行验收检查。"
        elif phase == PEVRPhase.REPAIR:
            return "上一次执行失败，请修复问题并重新执行。"
        return ""


# ── 辅助函数 ──────────────────────────────────────

def _build_observability_hints(request: ContextRequest) -> str:
    """从 ContextRequest 提取观测提示文本（1.5.4）。"""
    wm = request.working_memory
    hints = []

    # 修补阶段提示
    repair_ctx = wm.metadata.get("repair_context") if wm else None
    if repair_ctx and hasattr(repair_ctx, 'to_hint_text'):
        hints.append(repair_ctx.to_hint_text())

    # 审计摘要（最近 3 条 P0 事件）
    audit_events = wm.metadata.get("recent_audit_events", []) if wm else []
    if audit_events:
        hints.append("## 最近安全审计")
        for ev in audit_events[-3:]:
            if isinstance(ev, dict):
                hints.append(
                    f"- [{ev.get('decision', '?')}] {ev.get('rule_id', '?')} "
                    f"({ev.get('tool_name', '?')})"
                )

    return "\n".join(hints)


def _build_working_memory_text(request: ContextRequest) -> str:
    """将 WorkingMemory 序列化为文本。"""
    parts = []
    for sr in request.working_memory.step_results:
        icon = "✓" if sr.status == "success" else "✗" if sr.status == "failed" else "○"
        parts.append(f"[{icon}] {sr.step}: {sr.result[:200]}")
    return "\n".join(parts)


def _build_history_text(request: ContextRequest) -> str:
    """将历史消息列表转为文本。"""
    if not request.history:
        return ""
    lines = []
    for msg in request.history[-20:]:  # 最多 20 条
        if hasattr(msg, 'role') and hasattr(msg, 'content'):
            role = msg.role.upper()
            content = str(msg.content or "")
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"[{role} → {tc.function_name}] {json.dumps(tc.arguments, ensure_ascii=False)[:200]}")
            elif role == "TOOL":
                lines.append(f"[{role}] {content[:300]}")
            else:
                lines.append(f"[{role}] {content[:500]}")
        elif isinstance(msg, str):
            lines.append(msg[:500])
    return "\n".join(lines)


def _working_memory_to_template(request: ContextRequest) -> dict:
    """为模板准备 working_memory 数据。"""
    return {
        "content": _build_working_memory_text(request),
        "artifacts": request.working_memory.artifacts,
        "step_results": [
            {"step": sr.step, "result": sr.result, "status": sr.status}
            for sr in request.working_memory.step_results
        ],
        "notes": request.working_memory.notes,
    }


def _relevant_specs(files: list[str], phase: PEVRPhase) -> list[dict]:
    """从规范文件列表中提取相关片段。"""
    if phase == PEVRPhase.PLAN:
        return []  # plan 阶段全量注入，不走 relevant_specs
    results = []
    for f in files:
        source = _extract_source(f)
        body = _extract_body(f)
        # execute/repair 阶段：截断大文件，保留关键片段
        if len(body) > 2000:
            body = body[:2000] + "\n[已截断，完整规范见 Plan 阶段]"
        results.append({"source": source, "content": body})
    return results


def _extract_source(file_text: str) -> str:
    """从 [来源: path] 格式中提取来源路径。"""
    if file_text.startswith("[来源:"):
        end = file_text.index("]")
        return file_text[4:end]
    return "unknown"


def _extract_body(file_text: str) -> str:
    """从 [来源: path]\ncontent 格式中提取正文。"""
    if file_text.startswith("[来源:"):
        end = file_text.index("]")
        return file_text[end + 2:]  # 跳过 ]\n
    return file_text


def _trim_detail(slots: SlotSet) -> str:
    """生成裁剪详情说明。"""
    parts = []
    for s in [slots.system, slots.plan, slots.history, slots.working_memory, slots.observability_hints]:
        if "[已截断]" in s.content or "[已裁剪" in s.content:
            parts.append(f"{s.name}: {estimate_tokens(s.content)} tokens")
    return "; ".join(parts)
