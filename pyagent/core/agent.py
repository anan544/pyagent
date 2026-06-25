"""
Agent 主循环 — ReAct Loop 的核心实现。

ReAct（Reasoning + Acting）循环：
    ┌──────────────────────────────────────────┐
    │  user_prompt                              │
    │     ↓                                     │
    │  ContextAssembler.assemble(request)       │  ← v0.4.1: 可注入
    │     ↓                                     │
    │  messages = [system, ...history, user]    │
    │     ↓                                     │
    │  ┌─────────────────────┐                  │
    │  │  while True:         │ ← 循环         │
    │  │    response = llm()  │                 │
    │  │    save to memory    │  ← 增量持久化  │
    │  │    if tool_calls:    │                 │
    │  │      execute tools   │                 │
    │  │      save results    │  ← 增量持久化  │
    │  │    else:             │                 │
    │  │      return content  │                 │
    │  │    if too_many > MAX │                 │
    │  └─────────────────────┘                  │
    │     ↓                                     │
    │  最终回复                                  │
    └──────────────────────────────────────────┘
"""

import json
import re
from typing import Optional, List
from .config import AgentConfig
from .message import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
    Message,
)


class MaxIterationsExceeded(Exception):
    """Agent 循环次数超过最大限制时抛出。"""

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        super().__init__(
            f"Agent 循环次数超过限制 ({max_iterations})，可能存在死循环。"
        )


class Agent:
    """
    LLM Agent — 封装 ReAct 循环。

    支持 ContextAssembler DI（v0.4.1）：
        from pyagent.harness.context import ReactContextAssembler

        assembler = ReactContextAssembler(memory=memory)
        agent = Agent(config, registry, llm,
                      context_assembler=assembler)

    向后兼容：未提供 assembler 时自动使用 ReactContextAssembler（内建）。
    """

    def __init__(self, config: AgentConfig, tool_registry, llm_provider,
                 logger=None, memory=None, context_compressor=None,
                 context_assembler=None, governance=None):
        """
        Args:
            config: Agent 配置（可含 token_budget）。
            tool_registry: ToolRegistry 实例，管理所有可用工具。
            llm_provider: LLMProvider 实例，封装 LLM API 调用。
            logger: 可选的 Logger 实例。
            memory: 可选的 MemoryManager 实例，提供长期记忆持久化。
            context_compressor: 可选的 ContextCompressor 实例。
            context_assembler: 可选的 ContextAssembler 实例（v0.4.1）。
                               用于自定义消息拼装逻辑。未提供时使用内建 ReAct 组装。
            governance: 可选的 GovernanceWrapper 实例（v0.10.0）。
                        用于安全治理前置门控。None 时跳过安全检查（向后兼容）。
        """
        self.config = config
        self.tool_registry = tool_registry
        self.llm = llm_provider
        self.log = logger
        self.memory = memory
        self.context_compressor = context_compressor
        self.context_assembler = context_assembler
        self._governance = governance
        self._captured_tool_calls: list[dict] = []  # 每次 run() 重置
        self._workspace: str | None = None  # ★ v2.0: 用户工作区路径
        self._dynamic_context: str = ""     # ★ v2.1: 动态上下文（时间/位置等）
        self._current_thought: str = ""     # ★ v2.2: 当前思考内容（文本 ReAct 模式）
        self._tool_output_compressor = None  # ★ v2.4: 工具输出压缩器（懒加载）

    @property
    def workspace(self) -> str | None:
        """用户工作区路径。设置后，所有文件/命令工具将在该目录下执行。"""
        return self._workspace

    @workspace.setter
    def workspace(self, path: str | None):
        self._workspace = path

    @property
    def dynamic_context(self) -> str:
        """动态上下文块，注入到 system prompt 末尾。"""
        return self._dynamic_context

    @dynamic_context.setter
    def dynamic_context(self, context: str):
        self._dynamic_context = context

    @property
    def current_thought(self) -> str:
        """返回当前轮次的思考内容（文本 ReAct 模式下的 <thought> 标签内容）。"""
        return self._current_thought

    @property
    def captured_tool_calls(self) -> list[dict]:
        """返回本轮 run() 中所有工具调用记录。"""
        return list(self._captured_tool_calls)

    # ── 公开 API ──────────────────────────────────

    def reset_security_state(self):
        """v0.10.1: 重置当前会话的安全状态。

        每个新会话/请求开始前调用，确保：
        - 风险上下文（SessionRiskContext）窗口清空
        - 熔断器（SecurityCircuitBreaker）恢复 CLOSED
        - 前一会话的 BLOCK 计数不影响当前会话
        """
        if self._governance is not None:
            self._governance.reset_session()

    async def run(self, user_prompt: str, session_id: str = None) -> str:
        """
        执行 Agent 的完整 ReAct 循环。

        Args:
            user_prompt: 用户输入的问题/任务。
            session_id: 可选的会话 ID。

        Returns:
            LLM 的最终文本回复。

        Raises:
            MaxIterationsExceeded: 循环次数超过 max_iterations。
        """
        # v0.10.1: 新会话开始，重置安全状态
        self.reset_security_state()
        self._captured_tool_calls = []  # 重置工具调用记录

        # 确保会话存在（CLI 直接调用无 API 层的 create_session）
        if session_id and self.memory:
            await self.memory.create_session(session_id)

        # ── 消息组装 ──
        if self.context_assembler is not None:
            messages = await self._assemble_via_di(user_prompt, session_id)
        else:
            messages = await self._assemble_builtin(user_prompt, session_id)

        # ── 启动日志 ──
        self._log("=" * 60)
        self._log(f"[START] Agent 启动")
        self._log(f"[INPUT] 用户输入: {user_prompt}")
        self._log(f"[SESSION] {session_id or '(无持久化)'}")
        self._log(f"[TOOLS] 可用工具: {self.tool_registry.list_names()}")
        self._log("=" * 60)

        # ── ReAct 循环 ──
        return await self._react_loop(messages, session_id)

    async def run_with_messages(
        self, messages: List[Message], session_id: str = None
    ) -> str:
        """
        使用预组装的消息列表执行 ReAct 循环。

        PEVRRunner 等编排器使用此方法：先通过 ContextAssembler 组装消息，
        再传入此方法执行 ReAct 循环。跳过内建的消息拼装和启动日志。

        Args:
            messages: 预组装的消息列表（需已包含 SystemMessage + UserMessage）。
            session_id: 可选的会话 ID。

        Returns:
            LLM 的最终文本回复。
        """
        # 增量保存 UserMessage（最后一条）
        if session_id and self.memory:
            for msg in reversed(messages):
                if isinstance(msg, UserMessage):
                    await self.memory.save_message(session_id, msg)
                    break

        return await self._react_loop(messages, session_id)

    # ── 消息组装（内建，向后兼容）─────────────────

    async def _assemble_builtin(
        self, user_prompt: str, session_id: str = None
    ) -> List[Message]:
        """
        内建消息组装 — 与 v0.3.0 行为完全一致。

        提取自原有 run() 方法 103-143 行。
        v2.3: 支持外部规则文件（context_files + rules_dir）动态注入。
        """
        # SystemMessage — ★ v2.1: 追加动态上下文（时间/位置）
        system_content = self.config.system_prompt
        if self._dynamic_context:
            system_content += "\n\n" + self._dynamic_context

        # ★ v2.3: 加载外部规则文件（agent.md、rules/*.md 等）
        rules_content = await self._load_rules(user_input=user_prompt)
        if rules_content:
            system_content += "\n\n" + rules_content

        system_msg = SystemMessage(content=system_content)
        messages: list[Message] = [system_msg]

        # 计算系统提示的 Token 消耗（供 budget 使用）
        system_tokens = max(1, len(system_msg.content or "") // 4)

        # 加载历史消息（带滑动窗口 + 可选压缩）
        if session_id and self.memory:
            budget = self.config.token_budget
            if budget:
                budget.system_prompt_tokens = system_tokens

            loaded = await self.memory.load_messages(
                session_id,
                budget=budget,
                compressor=self.context_compressor,
            )

            if isinstance(loaded, list):
                history_msgs = loaded
                history_info = ""
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

        # 添加当前用户消息
        user_msg = UserMessage(content=user_prompt)
        messages.append(user_msg)

        # 增量保存用户消息
        if session_id and self.memory:
            await self.memory.save_message(session_id, user_msg)

        return messages

    async def _assemble_via_di(
        self, user_prompt: str, session_id: str = None
    ) -> List[Message]:
        """
        通过注入的 ContextAssembler 组装消息（v0.4.1 DI 路径）。
        """
        # 构造 ContextRequest
        from ..harness.context.models import ContextRequest, WorkingMemory

        request = ContextRequest(
            system_prompt=self.config.system_prompt,
            plan=user_prompt,
            working_memory=WorkingMemory(notes=user_prompt),
            token_budget=self.config.token_budget,
        )

        # 通过 model_extra 传递 session_id（保持 assembler 接口纯净）
        request._session_id = session_id  # type: ignore

        result = await self.context_assembler.assemble(request)

        # 更新 session_id（assembler 可能创建了新会话）
        sid = getattr(request, '_session_id', session_id)

        # 增量保存 UserMessage（如果 assembler 未处理）
        if sid and self.memory:
            for msg in reversed(result.messages):
                if isinstance(msg, UserMessage):
                    await self.memory.save_message(sid, msg)
                    break

        return result.messages

    # ── 外部规则加载（v2.3）───────────────────────

    async def _load_rules(self, user_input: str = "") -> str:
        """加载外部规则文件，支持 YAML Frontmatter 条件加载。

        每次 run() 调用时重新读取磁盘文件，实现热更新。

        规则加载策略：
            - 无 paths Frontmatter → 全局规则，始终注入
            - 有 paths Frontmatter → 条件规则，仅当用户输入中的文件路径匹配 glob 时注入
            - context_files → 显式指定文件，始终注入
        """
        if not self.config.context_files and not self.config.rules_dir:
            return ""

        from ..harness.context.rules_loader import RuleLoader

        loader = RuleLoader(
            rules_dir=self.config.rules_dir,
            context_files=list(self.config.context_files) if self.config.context_files else None,
        )

        workspace = self._workspace or ""
        result = loader.load(user_input=user_input, workspace=workspace)

        if result:
            self._log(f"[RULES] 已注入规则 ({len(result)} 字符)")

        return result

    # ── ReAct 循环核心 ─────────────────────────────

    async def _react_loop(
        self, messages: List[Message], session_id: str = None
    ) -> str:
        """
        ReAct 循环核心（提取自原 run() 方法 156-201 行）。

        v2.2: 支持两种模式 —
          - 文本 ReAct 模式：system prompt 含 <action> 标签时自动启用，
            解析 <thought> / <action> / <final_answer>，不发送 function 定义。
          - 函数调用模式：传统 OpenAI function calling（向后兼容）。
        """
        use_text_react = "<action>" in (self.config.system_prompt or "")

        for iteration in range(1, self.config.max_iterations + 1):
            self._log(f"\n-- Round {iteration}/{self.config.max_iterations} --")

            # ── 1. 调用 LLM ──
            if use_text_react:
                tools_schema = []  # 文本模式：不发送 function 定义
                self._log("[LLM] 调用 LLM (文本 ReAct 模式)...")
            else:
                tools_schema = self.tool_registry.get_all_schemas()
                self._log(f"[LLM] 调用 LLM (携带 {len(tools_schema)} 个工具定义)...")

            response: AssistantMessage = await self.llm.generate(
                messages, tools_schema
            )

            # ── 2. 把 LLM 回复加入消息历史 ──
            messages.append(response)
            if session_id and self.memory:
                await self.memory.save_message(session_id, response)

            # ═══════════════════════════════════════════════
            # 文本 ReAct 模式
            # ═══════════════════════════════════════════════
            if use_text_react:
                content = response.content or ""
                parsed = self._parse_react_response(content)
                self._current_thought = parsed["thought"] or ""

                if parsed["thought"]:
                    self._log(f"[THOUGHT] {parsed['thought'][:200]}")

                # 有 final_answer → 直接返回
                if parsed["final_answer"] is not None:
                    self._log("[DONE] Agent 完成（<final_answer>），返回文本回复")
                    self._log(f"[OUTPUT] 最终回复: {parsed['final_answer'][:200]}")
                    if session_id and self.memory:
                        await self.memory.update_session(session_id)
                        count = await self.memory.message_count(session_id)
                        self._log(f"[MEMORY] 会话 {session_id} 共 {count} 条消息")
                    return parsed["final_answer"]

                # 有 action → 解析 JSON 并执行工具
                if parsed["action"] is not None:
                    action = parsed["action"]
                    tool_name = action.get("tool", "")
                    tool_args = action.get("arguments", {})
                    self._log(f"[TOOL] 文本 ReAct 请求: {tool_name}({tool_args})")

                    # 构造 ToolCall 并执行
                    tc = ToolCall(
                        id=f"call_{iteration}",
                        function_name=tool_name,
                        arguments=tool_args,
                    )
                    tool_msg = await self._execute_tool(tc)
                    # 文本模式下将工具结果包装为 UserMessage（标准 ReAct 做法）
                    result_text = (
                        f"Tool '{tool_name}' execution result:\n"
                        f"{tool_msg.content}"
                    )
                    user_result = UserMessage(content=result_text)
                    messages.append(user_result)
                    if session_id and self.memory:
                        await self.memory.save_message(session_id, user_result)
                    continue

                # 既无 action 也无 final_answer → 视为最终回复
                self._log("[DONE] 无 ReAct 标签，视为最终回复")
                if session_id and self.memory:
                    await self.memory.update_session(session_id)
                return content

            # ═══════════════════════════════════════════════
            # 函数调用模式（原始逻辑）
            # ═══════════════════════════════════════════════
            else:
                # 3. 判断是否有工具调用
                if response.has_tool_calls():
                    self._log(
                        f"[TOOL] LLM 请求调用 {len(response.tool_calls)} 个工具:"
                    )
                    for tc in response.tool_calls:
                        self._log(f"   -> {tc.function_name}({tc.arguments})")

                    # 4. 依次执行每个工具调用
                    for tc in response.tool_calls:
                        tool_msg = await self._execute_tool(tc)
                        messages.append(tool_msg)
                        if session_id and self.memory:
                            await self.memory.save_message(session_id, tool_msg)

                    continue  # 回到循环
                else:
                    # 5. 没有工具调用 — 最终回复
                    self._log("[DONE] Agent 完成，返回文本回复")
                    self._log(f"[OUTPUT] 最终回复: {response.content}")

                    if session_id and self.memory:
                        await self.memory.update_session(session_id)
                        count = await self.memory.message_count(session_id)
                        self._log(f"[MEMORY] 会话 {session_id} 共 {count} 条消息")

                    return response.content or ""

        raise MaxIterationsExceeded(self.config.max_iterations)

    # ── 工具执行 ──────────────────────────────────

    def _get_tool_output_compressor(self):
        """懒加载工具输出压缩器（纯本地，不调 LLM）。

        与对话历史压缩不同：工具输出压缩在每个工具执行完立即触发，
        使用同步压缩（正则/AST/去重），不额外消耗 LLM Token。

        Returns:
            ContextCompressor 实例，用于 compress_sync()。
        """
        if self._tool_output_compressor is None:
            from ..memory.compressor import ContextCompressor
            # 不传 llm_provider → compress_sync() 走纯本地路径
            self._tool_output_compressor = ContextCompressor(
                llm_provider=None,
                max_summary_tokens=3000,  # 工具输出截断阈值
            )
        return self._tool_output_compressor

    async def _execute_tool(self, tool_call: ToolCall) -> ToolMessage:
        """执行单个工具调用，并包装为 ToolMessage。

        v0.10.0: 若配置了 governance，通过 GovernanceWrapper 执行安全前置检查。
        v2.0: 注入 workspace 为工具的 cwd，使文件/命令操作在用户项目目录执行。
        v2.4: 工具输出压缩 — 执行结果在喂给 LLM 之前先过压缩器，减少 Token 消耗。
        """
        # ★ v2.0: 注入工作区路径为 cwd
        if self._workspace:
            tool_call.arguments["cwd"] = self._workspace

        # 记录工具调用（供 API 结构化响应）
        self._captured_tool_calls.append({
            "id": tool_call.id,
            "name": tool_call.function_name,
            "arguments": tool_call.arguments,
        })

        # ── 安全治理路径（v0.10.0）──
        if self._governance is not None:
            ctx = self._governance.get_active_context()
            result = await self._governance.execute_tool(
                tool_call, self.tool_registry, ctx,
            )
        else:
            # ── 原始路径（向后兼容）──
            try:
                result = await self.tool_registry.execute(
                    name=tool_call.function_name,
                    call_id=tool_call.id,
                    arguments=tool_call.arguments,
                )
            except Exception as e:
                self._log(f"   [WARN] 工具执行失败: {e}")
                return ToolMessage(
                    content=f"Error executing tool '{tool_call.function_name}': {e}",
                    tool_call_id=tool_call.id,
                    name=tool_call.function_name,
                )

        # ★ v2.4: 压缩工具输出（阈值：超过 2000 字符才压缩）
        result = self._compress_tool_output(result, tool_call.function_name)
        return result

    def _compress_tool_output(
        self, result: ToolMessage, tool_name: str
    ) -> ToolMessage:
        """压缩工具执行结果，减少喂给 LLM 的 Token 量。

        双引擎混合压缩：
            1. Headroom 优先 — Code (AST) + JSON (SmartCrusher) + Log
            2. PyAgent 兜底 — 如果 Headroom 不可用，用自带的 ContextCompressor
            - Kompress ML 不加载（2G 服务器省内存）
            - 小于 2000 字符原样返回
        """
        content = result.content or ""
        if len(content) <= 2000:
            return result

        # ── 引擎 1: Headroom ──
        try:
            import headroom
            compressed_text = headroom.compress(content)
            if isinstance(compressed_text, str) and len(compressed_text) < len(content):
                saved_pct = f"{(1 - len(compressed_text) / len(content)) * 100:.0f}%"
                new_content = (
                    f"[工具输出已压缩 — 节省 {saved_pct}，"
                    f"原文 {len(content)} → {len(compressed_text)} 字符]\n"
                    f"{compressed_text}"
                )
                self._log(
                    f"   [COMPRESS] {tool_name}: "
                    f"{len(content)} → {len(compressed_text)} 字符 "
                    f"(Headroom, 省 {saved_pct})"
                )
                return ToolMessage(
                    content=new_content,
                    tool_call_id=result.tool_call_id,
                    name=result.name,
                )
        except Exception as e:
            self._log(f"   [COMPRESS] Headroom 失败 ({tool_name}): {e}，降级 PyAgent")

        # ── 引擎 2: PyAgent 兜底 ──
        compressor = self._get_tool_output_compressor()
        try:
            compressed = compressor.compress_sync(content)
            saved = compressed.savings
            new_content = (
                f"[工具输出已压缩 — 节省 {saved} Token，"
                f"原文 {len(content)} → {len(compressed.text)} 字符]\n"
                f"{compressed.text}"
            )
            self._log(
                f"   [COMPRESS] {tool_name}: "
                f"{len(content)} → {len(compressed.text)} 字符 (PyAgent, 节省 {saved})"
            )
            return ToolMessage(
                content=new_content,
                tool_call_id=result.tool_call_id,
                name=result.name,
            )
        except Exception as e:
            self._log(f"   [COMPRESS] PyAgent 也失败 ({tool_name}): {e}，原样返回")
            return result

    # ── 文本 ReAct 解析（v2.2）────────────────────

    def _parse_react_response(self, content: str) -> dict:
        """解析文本 ReAct 响应中的 <thought> / <action> / <final_answer> 标签。

        返回:
            {
                "thought": str | None,
                "action": {"tool": str, "arguments": dict} | None,
                "final_answer": str | None,
            }

        容错策略：
            - 标签不完整 → 尽力提取
            - <action> JSON 畸形 → 尝试修复（Windows 路径反斜杠），
              修复仍失败时构造一条错误消息让模型重试，而非原始内容。
            - 多个 <action> → 只取第一个
        """
        result: dict = {
            "thought": None,
            "action": None,
            "final_answer": None,
        }

        if not content:
            return result

        # ── 提取 <thought> ──
        thought_m = re.search(
            r"<thought>\s*(.*?)\s*</thought>", content, re.DOTALL | re.IGNORECASE
        )
        if thought_m:
            result["thought"] = thought_m.group(1).strip()

        # ── 提取 <final_answer>（优先级最高）──
        fa_m = re.search(
            r"<final_answer>\s*(.*?)\s*</final_answer>",
            content, re.DOTALL | re.IGNORECASE,
        )
        if fa_m:
            result["final_answer"] = fa_m.group(1).strip()
            return result

        # ── 提取 <action> ──
        # 支持 <action> 和 <action type="..."> 等变体
        action_m = re.search(
            r"<action[^>]*>\s*(.*?)\s*</action>", content, re.DOTALL | re.IGNORECASE,
        )
        if action_m:
            json_str = action_m.group(1).strip()
            # 清理常见的 LLM 输出问题：反引号包裹、尾部逗号
            if json_str.startswith("```"):
                json_str = re.sub(r"^```\w*\s*", "", json_str)
                json_str = re.sub(r"\s*```$", "", json_str)

            action_data = self._try_parse_action_json(json_str)
            if action_data is not None:
                result["action"] = {
                    "tool": action_data["tool"],
                    "arguments": action_data.get("arguments", {}),
                }
            else:
                # JSON 完全无法解析 → 返回错误提示让模型重试，不污染上下文
                self._log(
                    f"[WARN] <action> JSON 解析失败，将提示模型重试。"
                    f" 原始内容前 200 字符: {json_str[:200]}"
                )
                result["final_answer"] = (
                    "ERROR: Your <action> tag contained invalid JSON. "
                    "Please ensure all backslashes in Windows paths are escaped "
                    "(use \\\\ or forward slashes /). "
                    "Please retry with properly formatted JSON."
                )

        return result

    def _try_parse_action_json(self, json_str: str) -> dict | None:
        """尝试解析 <action> 中的 JSON，自动修复常见问题。

        常见 LLM 输出问题：
            1. Windows 路径反斜杠未转义: "E:\\dir" → "E:\\\\dir"
            2. 尾部逗号: {"a": 1,}
            3. 单引号代替双引号
            4. Python 风格布尔: True/False/None

        Returns:
            解析成功的 dict（含 tool + arguments），失败返回 None。
        """
        # ── 尝试 1：直接解析 ──
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

        # ── 尝试 2：修复 Windows 路径反斜杠 ──
        # 检测非法转义序列（如 \t \s \a \d 等出现在不该出现的位置）
        try:
            fixed = self._fix_json_backslashes(json_str)
            data = json.loads(fixed)
            if isinstance(data, dict) and "tool" in data:
                self._log(f"[FIX] JSON 反斜杠修复后解析成功")
                return data
        except json.JSONDecodeError:
            pass

        # ── 尝试 3：Python 风格 → JSON（单引号、True/False/None）──
        try:
            py_fixed = json_str
            py_fixed = py_fixed.replace("'", '"')
            py_fixed = py_fixed.replace("True", "true")
            py_fixed = py_fixed.replace("False", "false")
            py_fixed = py_fixed.replace("None", "null")
            # 再次尝试反斜杠修复
            try:
                data = json.loads(py_fixed)
            except json.JSONDecodeError:
                py_fixed = self._fix_json_backslashes(py_fixed)
                data = json.loads(py_fixed)
            if isinstance(data, dict) and "tool" in data:
                self._log(f"[FIX] JSON Python→标准 转换后解析成功")
                return data
        except json.JSONDecodeError:
            pass

        return None

    @staticmethod
    def _fix_json_backslashes(json_str: str) -> str:
        r"""修复 JSON 字符串中未正确转义的 Windows 路径反斜杠。

        策略：在 JSON 字符串值内部，将 \ 替换为 \\，
        但保留已正确转义的序列（\" \\ \/ \b \f \n \r \t \uXXXX）。

        具体做法：
            1. 识别 JSON 字符串值（用正则匹配引号包裹的内容）
            2. 在字符串值内部，将 \X 替换为 \\X（X 不是合法转义字符）
        """
        # 匹配 JSON 字符串值：双引号包裹的内容（处理 \\" 转义）
        # 使用简单但实用的方法：逐字符扫描
        result: list[str] = []
        in_string = False
        i = 0
        while i < len(json_str):
            ch = json_str[i]
            if not in_string:
                result.append(ch)
                if ch == '"':
                    in_string = True
                i += 1
            else:
                if ch == '\\':
                    # 检查下一个字符
                    if i + 1 < len(json_str):
                        next_ch = json_str[i + 1]
                        if next_ch in '"\\/bfnrtu':
                            # 合法转义序列，原样保留
                            result.append(ch)
                            result.append(next_ch)
                            i += 2
                        else:
                            # 非法转义序列（如 \t 中的 \ 表示路径分隔符）
                            # 额外加一个反斜杠：\X → \\X
                            result.append('\\')
                            result.append('\\')
                            result.append(next_ch)
                            i += 2
                    else:
                        result.append(ch)
                        i += 1
                elif ch == '"':
                    result.append(ch)
                    in_string = False
                    i += 1
                else:
                    result.append(ch)
                    i += 1
        return ''.join(result)

    # ── 日志 ──────────────────────────────────────

    def _log(self, msg: str):
        """内部日志输出。"""
        if self.log:
            self.log(msg)
        elif self.config.verbose:
            print(msg)
