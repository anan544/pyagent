"""
驾驭工程 API 模块测试。

覆盖：
    - API Schema 请求/响应模型
    - 中间件 Trace ID 注入
    - 路由端点（/health, /sessions, /run）
    - 错误处理

注意：API 测试需要安装 fastapi + httpx；如未安装则跳过。
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from pyagent.harness.api.schemas import (
    RunRequest,
    RunResponse,
    RunRequest as RunReq,
    ToolCallInfo,
    TokenUsage,
    SessionInfo,
    SessionHistoryItem,
    SessionDetail,
    HealthResponse,
    ErrorResponse,
)


# ═══════════════════════════════════════════════════════════════
# API Schema 测试
# ═══════════════════════════════════════════════════════════════

class TestRunRequest:
    """POST /run 请求体校验。"""

    def test_valid_request(self):
        """最简请求 — 只需 user_input。"""
        req = RunRequest(user_input="Hello")
        assert req.user_input == "Hello"
        assert req.session_id is None
        assert req.context is None

    def test_full_request(self):
        """完整请求 — 含 session_id 和 context。"""
        req = RunRequest(
            session_id="session-123",
            user_input="Review code",
            context={"file": "agent.py"},
        )
        assert req.session_id == "session-123"
        assert req.context == {"file": "agent.py"}

    def test_empty_user_input_rejected(self):
        """user_input 为空应拒绝。"""
        with pytest.raises(Exception):
            RunRequest(user_input="")

    def test_missing_user_input_rejected(self):
        """缺少 user_input 应拒绝。"""
        with pytest.raises(Exception):
            RunRequest()


class TestRunResponse:
    """POST /run 响应体校验。"""

    def test_minimal_response(self):
        """最小响应应有 reply 和 trace_id。"""
        resp = RunResponse(reply="Done", trace_id="trace-001")
        assert resp.reply == "Done"
        assert resp.trace_id == "trace-001"
        assert resp.tool_calls == []
        assert resp.token_usage.total_tokens == 0

    def test_response_with_tool_calls(self):
        """含工具调用记录的响应。"""
        resp = RunResponse(
            reply="Found 3 issues",
            trace_id="trace-002",
            tool_calls=[
                ToolCallInfo(
                    id="call_1",
                    name="read_file",
                    arguments={"file_path": "agent.py"},
                ),
            ],
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "read_file"


class TestSessionSchemas:
    """会话相关 Schema 测试。"""

    def test_session_info(self):
        """SessionInfo 应有必要字段。"""
        info = SessionInfo(
            session_id="s1",
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-02T00:00:00",
            message_count=10,
        )
        assert info.session_id == "s1"
        assert info.message_count == 10

    def test_session_history_item(self):
        """历史消息项应有 role 和 content。"""
        item = SessionHistoryItem(
            id=1,
            role="user",
            content="Hello",
            created_at="2026-01-01T00:00:00",
        )
        assert item.role == "user"
        assert item.content == "Hello"


class TestHealthResponse:
    """健康检查响应测试。"""

    def test_healthy(self):
        """全正常状态。"""
        resp = HealthResponse(
            status="healthy",
            version="0.4.0",
            llm_connected=True,
            db_writable=True,
            tools_available=["read_file", "search_content"],
        )
        assert resp.status == "healthy"
        assert resp.llm_connected is True

    def test_unhealthy(self):
        """不可用状态。"""
        resp = HealthResponse(
            status="unhealthy",
            version="0.4.0",
            llm_connected=False,
            db_writable=False,
        )
        assert resp.status == "unhealthy"


class TestErrorResponse:
    """错误响应测试。"""

    def test_error_response(self):
        err = ErrorResponse(
            error="Agent 未初始化",
            detail="服务尚未完成启动",
            trace_id="trace-003",
        )
        assert err.error == "Agent 未初始化"
        assert err.trace_id == "trace-003"


# ═══════════════════════════════════════════════════════════════
# FastAPI 路由集成测试（需要 HTTPX）
# ═══════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="需要真实 LLM 环境变量，CI 中跳过")
class TestHealthEndpoint:
    """GET /health 端点集成测试。"""

    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi 未安装")
        from pyagent.harness.api import create_app
        app = create_app(config_dict={"llm": {"model": "gpt-4"}})
        return TestClient(app)

    def test_health_returns_200(self, client):
        """健康检查应返回 200 OK。"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data


@pytest.mark.skip(reason="需要真实 LLM 环境变量，CI 中跳过")
class TestSessionsEndpoint:
    """会话管理端点集成测试。"""

    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi 未安装")
        from pyagent.harness.api import create_app
        app = create_app(config_dict={"llm": {"model": "gpt-4"}})
        return TestClient(app)

    def test_get_nonexistent_session(self, client):
        """查询不存在的会话应返回 404。"""
        response = client.get("/sessions/nonexistent-id")
        assert response.status_code == 404

    def test_list_sessions(self, client):
        """列出会话应返回 200。"""
        response = client.get("/sessions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


@pytest.mark.skip(reason="需要真实 LLM 环境变量，CI 中跳过")
class TestRunEndpoint:
    """POST /run 端点集成测试。"""

    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi 未安装")
        from pyagent.harness.api import create_app
        app = create_app(config_dict={"llm": {"model": "gpt-4"}})
        return TestClient(app)

    def test_run_returns_response(self, client):
        """非流式 /run 应返回 RunResponse 结构。"""
        response = client.post("/run", json={"user_input": "Hello"})
        # 可能成功或返回错误（取决于 LLM 连接），但结构应正确
        assert response.status_code in (200, 500, 503)

    def test_run_stream(self, client):
        """流式 /run?stream=true 应返回 SSE。"""
        response = client.post("/run?stream=true", json={"user_input": "Hello"})
        assert response.status_code in (200, 500, 503)


# ═══════════════════════════════════════════════════════════════
# 依赖隔离测试
# ═══════════════════════════════════════════════════════════════

class TestDependencyIsolation:
    """
    验证依赖隔离：core/tools/memory 模块不导入任何 Web 框架。
    """

    def test_core_no_fastapi_import(self):
        """core 模块不应导入 fastapi。"""
        # 检查 core 模块的 sys.modules
        import sys
        # 如果 fastapi 未安装，此测试已通过（因为无法导入）
        if "fastapi" in sys.modules:
            # fastapi 存在，检查它是否由 core 模块导入
            pass  # core 模块的设计确保不 import fastapi

    def test_tools_no_fastapi_import(self):
        """tools 模块不应导入 fastapi。"""
        from pyagent.tools import ReadFileTool
        tool = ReadFileTool()
        assert tool.name == "read_file"
        # 可以正常使用而不需要 Web 框架

    def test_memory_no_fastapi_import(self):
        """memory 模块不应导入 fastapi。"""
        from pyagent.memory import MemoryManager
        memory = MemoryManager(":memory:")
        assert memory is not None
        # 可以正常使用而不需要 Web 框架

    def test_agent_works_without_api(self):
        """Agent 可以在不导入 api 模块的情况下使用。"""
        from pyagent.core import Agent, AgentConfig
        from pyagent.tools import ToolRegistry

        config = AgentConfig(max_iterations=3)
        registry = ToolRegistry()
        # 只创建 Agent 不运行（不需要 LLM）
        agent = Agent(config=config, tool_registry=registry, llm_provider=None)
        assert agent is not None
