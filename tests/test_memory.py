"""
测试长期记忆系统 — Database + MemoryManager + Agent 集成。

覆盖：
    - Database 层的 session/message CRUD
    - MemoryManager 的序列化/反序列化
    - Agent 与 MemoryManager 的集成点
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
from pyagent.memory.database import Database
from pyagent.memory.manager import MemoryManager


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _temp_db_path():
    """在临时目录中创建数据库路径。"""
    tmp = tempfile.mkdtemp()
    return os.path.join(tmp, "test.db")


# ═══════════════════════════════════════════════════════════════
# Database 层测试
# ═══════════════════════════════════════════════════════════════

class TestDatabase:
    """测试纯 SQLite 操作层。"""

    def test_create_and_get_session(self):
        """创建会话并读取。"""
        async def run():
            db_path = _temp_db_path()
            db = Database(db_path)
            try:
                await db.create_session("s1", {"project": "test"})
                session = await db.get_session("s1")
                assert session is not None
                assert session["session_id"] == "s1"
            finally:
                await db.close()
        asyncio.run(run())

    def test_create_session_upsert(self):
        """重复创建同一 session_id 应更新而非报错。"""
        async def run():
            db_path = _temp_db_path()
            db = Database(db_path)
            try:
                await db.create_session("s1", {"v": 1})
                await db.create_session("s1", {"v": 2})  # 不应抛异常
                session = await db.get_session("s1")
                assert session is not None
            finally:
                await db.close()
        asyncio.run(run())

    def test_list_sessions(self):
        """列出会话，按更新时间降序。"""
        async def run():
            db_path = _temp_db_path()
            db = Database(db_path)
            try:
                await db.create_session("s1")
                await db.create_session("s2")
                await db.create_session("s3")
                sessions = await db.list_sessions(limit=10)
                assert len(sessions) >= 3
            finally:
                await db.close()
        asyncio.run(run())

    def test_delete_session_cascade(self):
        """删除会话应级联删除所有消息。"""
        async def run():
            db_path = _temp_db_path()
            db = Database(db_path)
            try:
                await db.create_session("s1")
                await db.insert_message({
                    "session_id": "s1", "role": "user", "content": "hello",
                    "token_count": 1,
                })
                await db.delete_session("s1")
                assert await db.get_session("s1") is None
                msgs = await db.load_messages("s1")
                assert len(msgs) == 0
            finally:
                await db.close()
        asyncio.run(run())

    def test_insert_and_load_messages(self):
        """插入消息后能正确加载。"""
        async def run():
            db_path = _temp_db_path()
            db = Database(db_path)
            try:
                await db.create_session("s1")
                await db.insert_message({
                    "session_id": "s1", "role": "user", "content": "Q1",
                    "token_count": 1,
                })
                await db.insert_message({
                    "session_id": "s1", "role": "assistant", "content": "A1",
                    "token_count": 1,
                })
                msgs = await db.load_messages("s1")
                assert len(msgs) == 2
                assert msgs[0]["role"] == "user"
                assert msgs[0]["content"] == "Q1"
                assert msgs[1]["role"] == "assistant"
                assert msgs[1]["content"] == "A1"
            finally:
                await db.close()
        asyncio.run(run())

    def test_load_messages_with_limit(self):
        """limit 参数应限制返回的消息数量。"""
        async def run():
            db_path = _temp_db_path()
            db = Database(db_path)
            try:
                await db.create_session("s1")
                for i in range(10):
                    await db.insert_message({
                        "session_id": "s1", "role": "user",
                        "content": f"msg{i}", "token_count": 1,
                    })
                msgs = await db.load_messages("s1", limit=3)
                assert len(msgs) == 3
                # 应是最后 3 条（id 最大的 3 条）
                assert msgs[0]["content"] == "msg7"
                assert msgs[2]["content"] == "msg9"
            finally:
                await db.close()
        asyncio.run(run())

    def test_message_count(self):
        """统计消息数量。"""
        async def run():
            db_path = _temp_db_path()
            db = Database(db_path)
            try:
                await db.create_session("s1")
                assert await db.message_count("s1") == 0
                await db.insert_message({
                    "session_id": "s1", "role": "user", "content": "x",
                    "token_count": 1,
                })
                await db.insert_message({
                    "session_id": "s1", "role": "assistant", "content": "y",
                    "token_count": 1,
                })
                assert await db.message_count("s1") == 2
            finally:
                await db.close()
        asyncio.run(run())

    def test_update_session_timestamp(self):
        """更新时间戳不应报错。"""
        async def run():
            db_path = _temp_db_path()
            db = Database(db_path)
            try:
                await db.create_session("s1")
                await db.update_session_timestamp("s1")
            finally:
                await db.close()
        asyncio.run(run())

    def test_insert_message_with_tool_calls(self):
        """插入包含 tool_calls JSON 的 assistant 消息。"""
        async def run():
            db_path = _temp_db_path()
            db = Database(db_path)
            try:
                await db.create_session("s1")
                tc_json = '[{"id":"c1","function_name":"search","arguments":{"q":"x"}}]'
                await db.insert_message({
                    "session_id": "s1",
                    "role": "assistant",
                    "content": "let me check",
                    "tool_calls": tc_json,
                    "token_count": 5,
                })
                msgs = await db.load_messages("s1")
                assert len(msgs) == 1
                assert msgs[0]["tool_calls"] is not None
            finally:
                await db.close()
        asyncio.run(run())

    def test_insert_tool_message(self):
        """插入 ToolMessage（包含 tool_call_id 和 tool_name）。"""
        async def run():
            db_path = _temp_db_path()
            db = Database(db_path)
            try:
                await db.create_session("s1")
                await db.insert_message({
                    "session_id": "s1",
                    "role": "tool",
                    "content": "result here",
                    "tool_call_id": "call_123",
                    "tool_name": "search_content",
                    "token_count": 2,
                })
                msgs = await db.load_messages("s1")
                assert len(msgs) == 1
                assert msgs[0]["tool_call_id"] == "call_123"
                assert msgs[0]["tool_name"] == "search_content"
            finally:
                await db.close()
        asyncio.run(run())


# ═══════════════════════════════════════════════════════════════
# MemoryManager 测试
# ═══════════════════════════════════════════════════════════════

class TestMemoryManager:
    """测试 MemoryManager 的消息序列化/反序列化。"""

    def test_create_session_auto_id(self):
        """不传 session_id 时自动生成。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session()
                assert len(sid) == 12  # UUID hex[:12]
                session = await mgr.get_session(sid)
                assert session is not None
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_create_session_custom_id(self):
        """传入自定义 session_id。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("my-session")
                assert sid == "my-session"
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_save_and_load_user_message(self):
        """保存并加载 UserMessage。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")
                msg = UserMessage(content="hello world")
                await mgr.save_message(sid, msg)

                loaded = await mgr.load_messages(sid)
                assert len(loaded) == 1
                assert isinstance(loaded[0], UserMessage)
                assert loaded[0].content == "hello world"
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_save_and_load_assistant_message(self):
        """保存并加载包含 tool_calls 的 AssistantMessage。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")
                tc = ToolCall(id="call_1", function_name="read_file",
                              arguments={"path": "/f.py"})
                msg = AssistantMessage(content="reading file", tool_calls=[tc])
                await mgr.save_message(sid, msg)

                loaded = await mgr.load_messages(sid)
                assert len(loaded) == 1
                assert isinstance(loaded[0], AssistantMessage)
                assert loaded[0].content == "reading file"
                assert loaded[0].has_tool_calls()
                assert loaded[0].tool_calls[0].function_name == "read_file"
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_save_and_load_tool_message(self):
        """保存并加载 ToolMessage。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")
                msg = ToolMessage(
                    content="file content here",
                    tool_call_id="call_1",
                    name="read_file",
                )
                await mgr.save_message(sid, msg)

                loaded = await mgr.load_messages(sid)
                assert len(loaded) == 1
                assert isinstance(loaded[0], ToolMessage)
                assert loaded[0].content == "file content here"
                assert loaded[0].tool_call_id == "call_1"
                assert loaded[0].name == "read_file"
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_skips_system_messages_on_load(self):
        """加载时跳过 SystemMessage（由 Agent 重新注入 config.system_prompt）。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")
                await mgr.save_message(sid, SystemMessage(content="old system prompt"))
                await mgr.save_message(sid, UserMessage(content="user question"))

                loaded = await mgr.load_messages(sid)
                assert len(loaded) == 1
                assert isinstance(loaded[0], UserMessage)
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_load_respects_limit(self):
        """加载消息时遵守 limit 限制。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")
                for i in range(10):
                    await mgr.save_message(sid, UserMessage(content=f"msg{i}"))

                loaded = await mgr.load_messages(sid, limit=3)
                assert len(loaded) == 3
                # 应是最新的 3 条
                assert loaded[0].content == "msg7"
                assert loaded[2].content == "msg9"
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_message_count(self):
        """消息计数。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")
                assert await mgr.message_count(sid) == 0
                await mgr.save_message(sid, UserMessage(content="a"))
                await mgr.save_message(sid, AssistantMessage(content="b"))
                assert await mgr.message_count(sid) == 2
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_list_sessions(self):
        """列出所有会话。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                await mgr.create_session("a")
                await mgr.create_session("b")
                sessions = await mgr.list_sessions()
                assert len(sessions) >= 2
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_delete_session(self):
        """删除会话及其消息。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")
                await mgr.save_message(sid, UserMessage(content="test"))
                await mgr.delete_session(sid)
                assert await mgr.get_session(sid) is None
                assert await mgr.message_count(sid) == 0
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_roundtrip_empty_tool_calls(self):
        """AssistantMessage 无 tool_calls 的往返测试。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")
                msg = AssistantMessage(content="final answer")
                await mgr.save_message(sid, msg)

                loaded = await mgr.load_messages(sid)
                assert len(loaded) == 1
                assert isinstance(loaded[0], AssistantMessage)
                assert loaded[0].content == "final answer"
                assert not loaded[0].has_tool_calls()
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_token_estimation(self):
        """Token 估算应 > 0。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")
                await mgr.save_message(sid, UserMessage(content="a" * 100))
                rows = await mgr.db.load_messages(sid)
                assert rows[0]["token_count"] > 0
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_multiple_tool_calls_roundtrip(self):
        """多个 tool_calls 的序列化往返。"""
        async def run():
            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                sid = await mgr.create_session("s1")
                tcs = [
                    ToolCall(id="c1", function_name="search",
                             arguments={"pattern": "foo"}),
                    ToolCall(id="c2", function_name="read_file",
                             arguments={"path": "a.py"}),
                ]
                msg = AssistantMessage(content="checking", tool_calls=tcs)
                await mgr.save_message(sid, msg)

                loaded = await mgr.load_messages(sid)
                assert len(loaded[0].tool_calls) == 2
                assert loaded[0].tool_calls[0].function_name == "search"
                assert loaded[0].tool_calls[1].function_name == "read_file"
            finally:
                await mgr.close()
        asyncio.run(run())


# ═══════════════════════════════════════════════════════════════
# Agent 集成测试
# ═══════════════════════════════════════════════════════════════

class MockLLMForMemory:
    """模拟 LLM：第一轮调用工具，第二轮返回最终回复。"""
    def __init__(self):
        self.call_count = 0

    async def generate(self, messages, tools):
        self.call_count += 1
        if self.call_count == 1:
            return AssistantMessage(
                content="Let me check",
                tool_calls=[
                    ToolCall(id="c1", function_name="mock_tool", arguments={})
                ],
            )
        else:
            return AssistantMessage(content="Task completed.")


class MockRegistryForMemory:
    """模拟工具注册表。"""
    def get_all_schemas(self):
        return [{"type": "function", "function": {"name": "mock_tool"}}]

    def list_names(self):
        return ["mock_tool"]

    async def execute(self, name, call_id, arguments):
        return ToolMessage(
            content="mock result",
            tool_call_id=call_id,
            name=name,
        )


class TestAgentMemoryIntegration:
    """测试 Agent 与 MemoryManager 的集成。"""

    def test_agent_saves_messages_during_run(self):
        """Agent 运行时增量保存所有消息。"""
        async def run():
            from pyagent.core.agent import Agent

            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                config = AgentConfig(
                    system_prompt="You are helpful.",
                    max_iterations=10,
                    verbose=False,
                )
                agent = Agent(
                    config=config,
                    tool_registry=MockRegistryForMemory(),
                    llm_provider=MockLLMForMemory(),
                    memory=mgr,
                )
                sid = await mgr.create_session("test-session")
                result = await agent.run("do something", session_id=sid)

                assert result == "Task completed."
                # 应有: UserMessage + AssistantMessage(tool_calls) + ToolMessage + AssistantMessage(final)
                count = await mgr.message_count(sid)
                assert count == 4
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_agent_loads_history_on_second_run(self):
        """第二次调用同一 session_id 时加载历史消息。"""
        async def run():
            from pyagent.core.agent import Agent

            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                config = AgentConfig(
                    system_prompt="You are helpful.",
                    max_iterations=10,
                    verbose=False,
                )
                agent = Agent(
                    config=config,
                    tool_registry=MockRegistryForMemory(),
                    llm_provider=MockLLMForMemory(),
                    memory=mgr,
                )
                sid = await mgr.create_session("test-session")

                # 第一次运行
                await agent.run("first question", session_id=sid)
                first_count = await mgr.message_count(sid)

                # 第二次运行（同一个 session）
                agent.llm = MockLLMForMemory()  # 重置 mock
                await agent.run("second question", session_id=sid)

                second_count = await mgr.message_count(sid)
                # 第二次运行又增加了 4 条消息
                assert second_count == first_count + 4
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_agent_without_memory_still_works(self):
        """不提供 memory 时 Agent 应正常工作（向后兼容）。"""
        async def run():
            from pyagent.core.agent import Agent

            config = AgentConfig(
                system_prompt="You are helpful.",
                max_iterations=10,
                verbose=False,
            )
            agent = Agent(
                config=config,
                tool_registry=MockRegistryForMemory(),
                llm_provider=MockLLMForMemory(),
            )
            result = await agent.run("do something")
            assert result == "Task completed."
        asyncio.run(run())

    def test_agent_with_session_id_but_no_memory(self):
        """提供了 session_id 但没有 memory 时应正常工作（兼容模式）。"""
        async def run():
            from pyagent.core.agent import Agent

            config = AgentConfig(
                system_prompt="You are helpful.",
                max_iterations=10,
                verbose=False,
            )
            agent = Agent(
                config=config,
                tool_registry=MockRegistryForMemory(),
                llm_provider=MockLLMForMemory(),
            )
            result = await agent.run("do something", session_id="no-memory")
            assert result == "Task completed."
        asyncio.run(run())

    def test_history_included_in_prompt(self):
        """验证历史消息真的被包含在发送给 LLM 的消息列表中。"""
        async def run():
            from pyagent.core.agent import Agent

            captured_messages = []

            class CaptureLLM:
                async def generate(self, messages, tools):
                    captured_messages.extend(messages)
                    return AssistantMessage(content="done")

            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                config = AgentConfig(
                    system_prompt="SYSTEM: be helpful",
                    max_iterations=10,
                    verbose=False,
                )
                agent = Agent(
                    config=config,
                    tool_registry=MockRegistryForMemory(),
                    llm_provider=CaptureLLM(),
                    memory=mgr,
                )
                sid = await mgr.create_session("hist-test")

                # 先保存一些"历史"消息
                await mgr.save_message(sid, UserMessage(content="old question"))
                await mgr.save_message(sid, AssistantMessage(content="old answer"))

                # 运行新请求
                await agent.run("new question", session_id=sid)

                # 验证发送给 LLM 的消息包含历史
                roles = [m.role for m in captured_messages if hasattr(m, 'role')]
                assert roles[0] == "system"
                assert roles[1] == "user"      # old question (from history)
                assert roles[2] == "assistant"  # old answer (from history)
                assert roles[3] == "user"      # new question
                assert captured_messages[1].content == "old question"
                assert captured_messages[2].content == "old answer"
                assert captured_messages[3].content == "new question"
            finally:
                await mgr.close()
        asyncio.run(run())

    def test_agent_creates_session_automatically(self):
        """Agent 不负责创建 session（由调用者先创建），但可以用已有 session。"""
        async def run():
            from pyagent.core.agent import Agent

            db_path = _temp_db_path()
            mgr = MemoryManager(db_path)
            try:
                config = AgentConfig(verbose=False)
                agent = Agent(
                    config=config,
                    tool_registry=MockRegistryForMemory(),
                    llm_provider=MockLLMForMemory(),
                    memory=mgr,
                )
                sid = await mgr.create_session("pre-created")
                result = await agent.run("task", session_id=sid)
                assert result == "Task completed."
                assert await mgr.message_count(sid) > 0
            finally:
                await mgr.close()
        asyncio.run(run())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
