"""
测试上下文管理 — TokenBudget + 滑动窗口 + ContextCompressor + Agent 集成。

覆盖：
    - TokenBudget 分区计算
    - 纯滑动窗口裁剪（无 LLM）
    - ContextCompressor 的结构化摘要生成
    - MemoryManager.load_messages() 的三种模式
    - Agent 集成的 Token 预算控制
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyagent.core.message import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
)
from pyagent.core.config import AgentConfig
from pyagent.memory import MemoryManager, TokenBudget, ContextCompressor


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _temp_db_path():
    tmp = tempfile.mkdtemp()
    return os.path.join(tmp, "test.db")


# ═══════════════════════════════════════════════════════════════
# TokenBudget 测试
# ═══════════════════════════════════════════════════════════════

class TestTokenBudget:
    """测试 Token 预算计算和分区逻辑。"""

    def test_default_budget(self):
        """默认 128K 模型 + 0.7 安全系数。"""
        budget = TokenBudget(model_max_tokens=128_000, safety_factor=0.7)
        assert budget.total_budget == 89_600
        assert budget.available_budget == 89_600
        assert budget.precision_budget == 53_760   # 60% of available
        assert budget.compression_budget == 35_840  # 40% of available

    def test_with_system_prompt(self):
        """扣除系统提示后的预算分配。"""
        budget = TokenBudget(model_max_tokens=100_000, safety_factor=0.8)
        budget.system_prompt_tokens = 1_000
        assert budget.total_budget == 80_000
        assert budget.available_budget == 79_000
        assert budget.precision_budget == 47_400  # 60% of 79000
        assert budget.compression_budget == 31_600

    def test_for_model(self):
        """根据模型名称自动选择上下文窗口。"""
        budget = TokenBudget.for_model("deepseek-chat", safety_factor=0.7)
        assert budget.model_max_tokens == 128_000
        assert budget.total_budget == 89_600

        budget = TokenBudget.for_model("gpt-4")
        assert budget.model_max_tokens == 8_192

        budget = TokenBudget.for_model("unknown-model")
        assert budget.model_max_tokens == 128_000  # default

    def test_split_messages_all_in_precision(self):
        """消息总数在精确区预算内 → 全部在精确区。"""
        budget = TokenBudget(model_max_tokens=10_000, safety_factor=1.0)
        budget.system_prompt_tokens = 0
        # precision_budget = 6000

        messages = [
            UserMessage(content="a" * 400),  # ~100 tokens
            AssistantMessage(content="b" * 400),  # ~100 tokens
            UserMessage(content="c" * 400),  # ~100 tokens
        ]
        # total = 300 tokens << 6000

        comp, prec = budget.split_messages(messages)
        assert len(comp) == 0
        assert len(prec) == 3

    def test_split_messages_partial(self):
        """消息总数超预算 → 分割为压缩区 + 精确区。"""
        budget = TokenBudget(model_max_tokens=10_000, safety_factor=1.0)
        # precision_budget = 6000

        # 创建 10 条消息，每条 ~1000 tokens（估算值由 split 使用）
        messages = []
        for i in range(10):
            msg = UserMessage(content=f"msg{i}_" + "x" * 4000)  # ~1000 tokens
            messages.append(msg)

        comp, prec = budget.split_messages(messages)
        # 每条 ~1000 tokens，precision_budget = 6000 → 约 6 条在精确区
        assert len(prec) >= 4  # 至少 4 条在精确区
        assert len(prec) + len(comp) == 10
        # 压缩区消息的索引小于精确区
        assert messages.index(prec[0]) > messages.index(comp[-1]) if comp else True

    def test_split_empty(self):
        """空消息列表。"""
        budget = TokenBudget()
        comp, prec = budget.split_messages([])
        assert comp == []
        assert prec == []

    def test_info(self):
        """info() 返回完整预算信息。"""
        budget = TokenBudget.for_model("deepseek-chat")
        info = budget.info()
        assert "total_budget" in info
        assert "precision_budget" in info
        assert "compression_budget" in info
        assert info["model_max_tokens"] == 128_000


# ═══════════════════════════════════════════════════════════════
# ContextCompressor 测试
# ═══════════════════════════════════════════════════════════════

class TestContextCompressor:
    """测试压缩器的摘要生成和降级逻辑。"""

    def test_compress_empty_messages(self):
        """空消息列表应返回空摘要模板。"""
        async def run():
            compressor = ContextCompressor(llm_provider=None)
            result = await compressor.compress([])
            assert "已达成共识" in result
            assert "无" in result
        asyncio.run(run())

    def test_fallback_summary(self):
        """
        LLM 不可用时降级为简单提取。
        (compressor 传入 None 作为 llm，模拟不可用场景)
        """
        async def run():
            compressor = ContextCompressor(llm_provider=None)
            messages = [
                UserMessage(content="请读取 agent.py 文件"),
                AssistantMessage(
                    content="我来读取文件",
                    tool_calls=[
                        ToolCall(id="c1", function_name="read_file",
                                 arguments={"path": "agent.py"})
                    ],
                ),
                ToolMessage(
                    content="def run(): ...",
                    tool_call_id="c1",
                    name="read_file",
                ),
            ]
            # llm=None 时 compress() 会抛出异常 → 触发降级
            try:
                result = await compressor.compress(messages)
                assert "降级" in result or "agent.py" in result
            except AttributeError:
                # llm=None 没有 generate 方法 → 异常被捕获 → 降级
                result = compressor._fallback_summary(messages)
                assert "agent.py" in result
                assert "read_file" in result
        asyncio.run(run())

    def test_messages_to_text_format(self):
        """消息转换为文本的格式正确。"""
        compressor = ContextCompressor(llm_provider=None)
        messages = [
            UserMessage(content="帮我审查 agent.py"),
            AssistantMessage(
                content="读取文件中",
                tool_calls=[
                    ToolCall(id="c1", function_name="read_file",
                             arguments={"path": "agent.py"})
                ],
            ),
            ToolMessage(
                content="def run(self): pass",
                tool_call_id="c1",
                name="read_file",
            ),
        ]
        text = compressor._messages_to_text(messages)
        assert "[用户]" in text
        assert "[助手]" in text
        assert "[调用工具: read_file]" in text
        assert "[工具 read_file 返回]" in text
        assert "agent.py" in text

    def test_messages_to_text_truncates_long_tool_output(self):
        """过长工具输出应截断。"""
        compressor = ContextCompressor(llm_provider=None)
        messages = [
            ToolMessage(
                content="x" * 1000,
                tool_call_id="c1",
                name="search_content",
            ),
        ]
        text = compressor._messages_to_text(messages)
        # 500 字符限制 + 截断标记
        assert len(text) < 800
        assert "截断" in text

    def test_compression_prompt_contains_sections(self):
        """压缩 Prompt 应包含所有必需的结构化输出段落。"""
        from pyagent.memory.compressor import (
            COMPRESSION_SYSTEM_PROMPT,
            COMPRESSION_USER_TEMPLATE,
        )
        assert "技术负责人" in COMPRESSION_SYSTEM_PROMPT
        assert "已达成共识" in COMPRESSION_USER_TEMPLATE
        assert "待解决问题" in COMPRESSION_USER_TEMPLATE
        assert "关键文件路径" in COMPRESSION_USER_TEMPLATE
        assert "工具调用结论" in COMPRESSION_USER_TEMPLATE
        assert "关键实体" in COMPRESSION_USER_TEMPLATE
        assert "{conversation}" in COMPRESSION_USER_TEMPLATE


# ═══════════════════════════════════════════════════════════════
# MemoryManager + 滑动窗口/压缩 测试
# ═══════════════════════════════════════════════════════════════

class TestMemoryManagerWithBudget:
    """测试 MemoryManager.load_messages() 的三种模式。"""

    def test_mode_no_budget_backward_compat(self):
        """无 budget 时保持 v0.2.0 兼容行为 — 返回 list[Message]。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")
                await mgr.save_message(sid, UserMessage(content="hello"))
                await mgr.save_message(sid, AssistantMessage(content="world"))

                result = await mgr.load_messages(sid, limit=50)
                # v0.2.0 兼容：无 budget 时返回 list[Message]
                assert isinstance(result, list)
                assert len(result) == 2
                assert isinstance(result[0], UserMessage)
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_mode_budget_within_limit(self):
        """有 budget 但消息在预算内 → 全部返回。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")
                await mgr.save_message(sid, UserMessage(content="hi"))
                await mgr.save_message(sid, AssistantMessage(content="hey"))

                budget = TokenBudget(model_max_tokens=100_000, safety_factor=1.0)
                result = await mgr.load_messages(sid, budget=budget)
                assert len(result.messages) == 2
                assert result.compressed_count == 0
                assert result.trimmed_count == 0
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_mode_sliding_window_trimming(self):
        """
        纯滑动窗口：消息超预算且无 compressor → 裁剪早期消息。
        """
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")

                # 插入大量消息（每条 ~100 tokens 估算，总 ~5000 tokens）
                for i in range(50):
                    msg = UserMessage(content=f"msg{i:03d}_" + "x" * 380)
                    await mgr.save_message(sid, msg)

                # 设置很小的可用预算 → 触发裁剪
                # available_budget = 2000, precision = 1200
                budget = TokenBudget(model_max_tokens=3_000, safety_factor=1.0)
                result = await mgr.load_messages(sid, budget=budget)

                # 应裁剪了部分消息
                assert len(result.messages) < 50
                assert result.was_trimmed
                assert result.trimmed_count > 0
                assert result.compressed_count == 0
                assert result.summary is None

                # 保留的消息应是最新的（最后插入的）
                last_content = result.messages[-1].content
                assert "msg049_" in last_content
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_mode_compression_with_llm(self):
        """
        LLM 压缩：有 compressor 时压缩早期消息为摘要。
        使用 mock LLM 验证流程。
        """
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")

                # 插入大量消息（总 ~5000 tokens）
                for i in range(50):
                    msg = UserMessage(content=f"msg{i:03d}_" + "x" * 380)
                    await mgr.save_message(sid, msg)

                # Mock LLM — 返回结构化摘要
                class MockCompressionLLM:
                    async def generate(self, messages, tools):
                        return AssistantMessage(
                            content=(
                                "【已达成共识】\n- 测试项目\n\n"
                                "【待解决问题】\n无\n\n"
                                "【关键文件路径】\n- test.py\n\n"
                                "【工具调用结论】\n无\n\n"
                                "【关键实体】\n无"
                            )
                        )

                compressor = ContextCompressor(MockCompressionLLM())
                # available_budget = 3000 << 5000 total → 触发压缩
                budget = TokenBudget(model_max_tokens=4_000, safety_factor=1.0)

                result = await mgr.load_messages(
                    sid, budget=budget, compressor=compressor
                )

                # 应执行了压缩
                assert result.was_compressed
                assert result.compressed_count > 0
                assert result.summary is not None
                assert "已达成共识" in result.summary
                # 返回的消息中第一条应是摘要
                assert "上下文摘要" in result.messages[0].content

                # 验证摘要已持久化
                last_summary = await mgr.db.get_last_summary(sid)
                assert last_summary is not None
                assert last_summary["is_summary"] == 1
                assert "已达成共识" in last_summary["content"]
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_compression_called_only_when_over_budget(self):
        """消息在预算内时不触发压缩。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")
                await mgr.save_message(sid, UserMessage(content="hello"))

                compress_called = False

                class TrackedLLM:
                    async def generate(self, messages, tools):
                        nonlocal compress_called
                        compress_called = True
                        return AssistantMessage(content="summary")

                compressor = ContextCompressor(TrackedLLM())
                budget = TokenBudget(model_max_tokens=100_000, safety_factor=1.0)

                result = await mgr.load_messages(
                    sid, budget=budget, compressor=compressor
                )

                assert not compress_called  # 不应该调用 LLM
                assert not result.was_compressed
                assert len(result.messages) == 1
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_summary_persisted_and_reused(self):
        """压缩后的摘要持久化，下次加载时复用。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")

                # 插入消息（总 ~3000 tokens）
                for i in range(30):
                    await mgr.save_message(
                        sid, UserMessage(content=f"msg{i:03d}_" + "x" * 380)
                    )

                # 第一次压缩
                class MockLLM:
                    async def generate(self, messages, tools):
                        return AssistantMessage(
                            content="【已达成共识】\n- 第一轮压缩\n\n"
                                    "【待解决问题】\n无\n\n"
                                    "【关键文件路径】\n无\n\n"
                                    "【工具调用结论】\n无\n\n"
                                    "【关键实体】\n无"
                        )

                compressor = ContextCompressor(MockLLM())
                # 2,880 tokens > 2,000 available → 触发压缩
                budget = TokenBudget(model_max_tokens=2_000, safety_factor=1.0)

                result1 = await mgr.load_messages(
                    sid, budget=budget, compressor=compressor
                )
                assert result1.was_compressed

                # 验证摘要已写入 DB
                summary_row = await mgr.db.get_last_summary(sid)
                assert summary_row is not None
                assert "第一轮压缩" in summary_row["content"]

                # 第二次加载（不再超预算，因为旧消息已被删除+替换为摘要）
                result2 = await mgr.load_messages(
                    sid, budget=budget, compressor=compressor
                )
                # 不应该再次压缩（消息已在预算内）
                assert not result2.was_compressed, (
                    "二次加载不应再次压缩：旧消息已删除+摘要替换，总 Token 应在预算内"
                )
                # 摘要消息应出现在返回结果中（以 UserMessage 形式从 DB 加载）
                has_summary = any(
                    "第一轮压缩" in (m.content or "")
                    for m in result2.messages
                    if hasattr(m, 'content') and m.content
                )
                assert has_summary, (
                    "二次加载的消息列表中应包含 DB 中持久化的压缩摘要"
                )
            finally:
                await mgr.close()
        asyncio.run(run())


# ═══════════════════════════════════════════════════════════════
# Agent 集成测试
# ═══════════════════════════════════════════════════════════════

class TestAgentWithBudget:
    """测试 Agent 对 TokenBudget 和 Compressor 的集成。"""

    def test_agent_uses_budget_when_configured(self):
        """Agent 使用 config.token_budget 控制加载的消息数。"""
        async def run():
            from pyagent.core.agent import Agent

            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")

                # 插入大量消息（总 ~10000 tokens）
                for i in range(100):
                    await mgr.save_message(
                        sid, UserMessage(content=f"msg{i:03d}_" + "x" * 380)
                    )

                # 使用极小预算 → available = 2000 << 10000 → 触发裁剪
                budget = TokenBudget(model_max_tokens=3_000, safety_factor=1.0)

                class SimpleLLM:
                    async def generate(self, messages, tools):
                        return AssistantMessage(content="done")

                class SimpleRegistry:
                    def get_all_schemas(self):
                        return []
                    def list_names(self):
                        return []

                config = AgentConfig(
                    system_prompt="test",
                    max_iterations=3,
                    verbose=False,
                    token_budget=budget,
                )

                agent = Agent(
                    config=config,
                    tool_registry=SimpleRegistry(),
                    llm_provider=SimpleLLM(),
                    memory=mgr,
                )

                result = await agent.run("test question", session_id=sid)
                assert result == "done"

                # 验证发送给 LLM 的消息数被裁剪了
                # (通过 SimpleLLM 无法直接验证，但流程不报错即可)
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_agent_with_compressor(self):
        """Agent 使用 ContextCompressor 压缩早期消息。"""
        async def run():
            from pyagent.core.agent import Agent

            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")

                # 插入大量消息触发压缩（总 ~5000 tokens）
                for i in range(50):
                    await mgr.save_message(
                        sid, UserMessage(content=f"msg{i:03d}_" + "x" * 380)
                    )

                # available_budget = 3000 << 5000 → 触发压缩
                budget = TokenBudget(model_max_tokens=4_000, safety_factor=1.0)

                # 记录 LLM 收到的消息
                captured = []

                class CaptureLLM:
                    async def generate(self, messages, tools):
                        captured.append(list(messages))
                        return AssistantMessage(content="ok")

                class MockCompressLLM:
                    async def generate(self, messages, tools):
                        return AssistantMessage(
                            content="【已达成共识】\n- 压缩\n\n"
                                    "【待解决问题】\n无\n\n"
                                    "【关键文件路径】\n无\n\n"
                                    "【工具调用结论】\n无\n\n"
                                    "【关键实体】\n无"
                        )

                class SimpleRegistry:
                    def get_all_schemas(self):
                        return []
                    def list_names(self):
                        return []

                compressor = ContextCompressor(MockCompressLLM())

                config = AgentConfig(
                    system_prompt="You are helpful.",
                    max_iterations=3,
                    verbose=False,
                    token_budget=budget,
                )

                agent = Agent(
                    config=config,
                    tool_registry=SimpleRegistry(),
                    llm_provider=CaptureLLM(),
                    memory=mgr,
                    context_compressor=compressor,
                )

                await agent.run("question", session_id=sid)

                # 验证 LLM 收到的消息包含压缩摘要
                llm_messages = captured[0]
                # 第一条是 system，后面应有压缩摘要 + 部分原始消息
                has_summary = any(
                    "上下文摘要" in str(m.content)
                    for m in llm_messages
                    if hasattr(m, 'content') and m.content
                )
                assert has_summary, "LLM 收到的消息应包含压缩摘要"
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_agent_without_budget_still_works(self):
        """没有 token_budget 时 Agent 正常工作（v0.2.0 兼容）。"""
        async def run():
            from pyagent.core.agent import Agent

            config = AgentConfig(verbose=False)

            class SimpleLLM:
                async def generate(self, messages, tools):
                    return AssistantMessage(content="ok")

            class SimpleRegistry:
                def get_all_schemas(self):
                    return []
                def list_names(self):
                    return []

            agent = Agent(
                config=config,
                tool_registry=SimpleRegistry(),
                llm_provider=SimpleLLM(),
                # 不传 memory, compressor, budget
            )

            result = await agent.run("test")
            assert result == "ok"
        asyncio.run(run())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
