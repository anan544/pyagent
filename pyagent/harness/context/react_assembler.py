"""
ReactContextAssembler — 向后兼容的上下文组装器。

完全复现原有 Agent.run() 中的消息拼装逻辑（原 agent.py 103-143 行），
确保所有 84 个现有测试无需修改即可通过。

行为等价：
    SystemMessage(config.system_prompt)
    + load_history(session_id, budget, compressor)
    + UserMessage(user_prompt)

迁移后的 Agent.run() 通过注入 ReactContextAssembler 保持行为完全不变。
"""

import logging
from typing import Optional

from .models import ContextRequest, ContextResult, PEVRPhase
from .assembler import ContextAssembler
from ...core.message import SystemMessage, UserMessage

logger = logging.getLogger("pyagent.harness.context")


class ReactContextAssembler(ContextAssembler):
    """
    ReAct 模式的上下文组装器 — 等价于原 Agent._build_messages() 逻辑。

    不引入任何新行为，仅将原 Agent 中的消息拼装代码提取为独立类。
    """

    def __init__(
        self,
        memory: Optional[object] = None,
        context_compressor: Optional[object] = None,
        log_callback=None,
    ):
        """
        Args:
            memory: MemoryManager 实例（可选）。
            context_compressor: ContextCompressor 实例（可选）。
            log_callback: 日志回调，用于输出 [MEMORY] 等信息。
        """
        self.memory = memory
        self.context_compressor = context_compressor
        self._log = log_callback or (lambda _: None)

    # ── 主入口 ────────────────────────────────────

    async def assemble(self, request: ContextRequest) -> ContextResult:
        """
        组装 ReAct 上下文（完全复现原逻辑）。

        Args:
            request: ContextRequest，需包含：
                - system_prompt
                - history (由调用方从 memory 加载后填入，或留空由本方法自行加载)
                - user_prompt (放在 working_memory.notes 或 plan 中)
                - token_budget (可选)
                - 额外属性 session_id (通过 ContextRequest.model_extra)

        Returns:
            ContextResult: 组装好的 [SystemMessage, ...history, UserMessage] 列表。
        """
        system_prompt = request.system_prompt
        user_prompt = request.plan or request.working_memory.notes or ""
        token_budget = request.token_budget
        session_id = getattr(request, '_session_id', None)
        # ContextRequest 支持 extra fields via model_config
        # 这里从 model_extra 或直接属性获取 session_id
        if hasattr(request, 'model_extra') and request.model_extra:
            session_id = request.model_extra.get('session_id', session_id)

        # ── 1. SystemMessage ────────────────────
        system_msg = SystemMessage(content=system_prompt)
        messages: list = [system_msg]
        system_tokens = max(1, len(system_prompt or "") // 4)

        # ── 2. 加载历史 ─────────────────────────
        history_info = ""
        if session_id and self.memory:
            budget = token_budget
            if budget:
                budget.system_prompt_tokens = system_tokens

            loaded = await self.memory.load_messages(
                session_id,
                budget=budget,
                compressor=self.context_compressor,
            )

            if isinstance(loaded, list):
                history_msgs = loaded
            else:
                history_msgs = loaded.messages
                history_info = (
                    f" (原始 {loaded.original_count} 条, "
                    f"Token: {loaded.total_tokens}/"
                    f"{budget.available_budget if budget else 'N/A'})"
                )
                if loaded.was_compressed:
                    history_info += f" [压缩: {loaded.compressed_count} 条]"
                elif loaded.was_trimmed:
                    history_info += f" [裁剪: {loaded.trimmed_count} 条]"

            if history_msgs:
                messages.extend(history_msgs)
                self._log(
                    f"[MEMORY] 加载了 {len(history_msgs)} 条消息{history_info}"
                )

            # 保存 session_id 供后续使用
            request._session_id = session_id  # type: ignore

        # ── 3. UserMessage ──────────────────────
        user_msg = UserMessage(content=user_prompt)
        messages.append(user_msg)

        # 增量保存 user 消息
        if session_id and self.memory:
            await self.memory.save_message(session_id, user_msg)

        # ── 4. Token 统计 ────────────────────────
        total = sum(max(1, len(str(m.content or "")) // 4) for m in messages)

        return ContextResult(
            messages=messages,
            total_tokens=total,
        )
