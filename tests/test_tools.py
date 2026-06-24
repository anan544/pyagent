"""
测试工具系统 — ToolRegistry、file_ops、code_executor。
"""

import asyncio
import os
import tempfile
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyagent.tools import (
    Tool,
    ToolRegistry,
    ReadFileTool,
    WriteFileTool,
    CodeExecutorTool,
)
from pyagent.core.message import ToolMessage


# ═══════════════════════════════════════════════════════════════
# ToolRegistry 测试
# ═══════════════════════════════════════════════════════════════

class TestToolRegistry:
    """测试工具注册表的基本功能。"""

    def test_register_and_list(self):
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())

        names = registry.list_names()
        assert "read_file" in names
        assert "write_file" in names
        assert len(names) == 2

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        with pytest.raises(ValueError, match="已经注册过"):
            registry.register(ReadFileTool())

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        registry.unregister("read_file")
        assert "read_file" not in registry.list_names()

    def test_get_all_schemas(self):
        registry = ToolRegistry()
        registry.register(ReadFileTool())

        schemas = registry.get_all_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "read_file"
        assert "parameters" in schemas[0]["function"]

    def test_execute_unknown_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(Exception):
            asyncio.run(
                registry.execute("nonexistent", "call_1", {})
            )

    def test_execute_returns_tool_message(self):
        registry = ToolRegistry()
        registry.register(ReadFileTool())

        # 创建一个临时文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("hello world")
            tmp = f.name

        try:
            msg = asyncio.run(
                registry.execute("read_file", "call_123", {"path": tmp})
            )
            assert isinstance(msg, ToolMessage)
            assert msg.tool_call_id == "call_123"
            assert msg.name == "read_file"
            assert "hello world" in msg.content
        finally:
            os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════
# ReadFileTool 测试
# ═══════════════════════════════════════════════════════════════

class TestReadFileTool:
    """测试文件读取工具。"""

    def test_read_existing_file(self):
        tool = ReadFileTool()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("line1\nline2\nline3")
            tmp = f.name

        try:
            result = asyncio.run(tool.execute(path=tmp))
            assert "line1" in result
            assert "line2" in result
            assert "line3" in result
            # 检查行号
            assert "   1|" in result
            assert "   3|" in result
        finally:
            os.unlink(tmp)

    def test_read_nonexistent_file(self):
        tool = ReadFileTool()
        result = asyncio.run(tool.execute(path="/nonexistent/file.txt"))
        assert "错误" in result or "不存在" in result

    def test_read_directory_not_file(self):
        tool = ReadFileTool()
        result = asyncio.run(tool.execute(path=os.path.dirname(__file__)))
        assert "不是文件" in result


# ═══════════════════════════════════════════════════════════════
# WriteFileTool 测试
# ═══════════════════════════════════════════════════════════════

class TestWriteFileTool:
    """测试文件写入工具。"""

    def test_write_and_verify(self):
        tool = WriteFileTool()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            tmp = f.name

        try:
            result = asyncio.run(
                tool.execute(path=tmp, content="Hello from test!")
            )
            assert "写入成功" in result or "成功" in result

            # 验证内容
            with open(tmp, "r", encoding="utf-8") as f:
                assert f.read() == "Hello from test!"
        finally:
            os.unlink(tmp)

    def test_creates_parent_directory(self):
        tool = WriteFileTool()
        tmp_dir = tempfile.mkdtemp()
        new_path = os.path.join(tmp_dir, "subdir", "test.txt")

        try:
            result = asyncio.run(
                tool.execute(path=new_path, content="nested file")
            )
            assert os.path.exists(new_path)
            with open(new_path, "r") as f:
                assert f.read() == "nested file"
        finally:
            # 递归清理
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# CodeExecutorTool 测试
# ═══════════════════════════════════════════════════════════════

class TestCodeExecutorTool:
    """测试代码执行工具。"""

    def test_simple_execution(self):
        tool = CodeExecutorTool()
        result = asyncio.run(
            tool.execute(code="print('hello from subprocess')")
        )
        assert "hello from subprocess" in result
        assert "退出码: 0" in result

    def test_stderr_capture(self):
        tool = CodeExecutorTool()
        result = asyncio.run(
            tool.execute(code="import sys; sys.stderr.write('error msg')")
        )
        assert "error msg" in result

    def test_exception_handling(self):
        """代码抛出异常应该体现在退出码和 stderr 中。"""
        tool = CodeExecutorTool()
        result = asyncio.run(
            tool.execute(code="raise ValueError('test error')")
        )
        assert "ValueError" in result
        # 异常导致退出码非 0
        assert "退出码: 1" in result

    def test_timeout(self):
        """无限循环应该被超时机制终止。"""
        tool = CodeExecutorTool(default_timeout=2)
        result = asyncio.run(
            tool.execute(
                code="while True: pass",
                timeout=1,  # 1 秒超时
            )
        )
        assert "超时" in result

    def test_multiline_code(self):
        tool = CodeExecutorTool()
        code = """
result = 0
for i in range(10):
    result += i
print(f"Sum: {result}")
"""
        result = asyncio.run(tool.execute(code=code))
        assert "Sum: 45" in result
        assert "退出码: 0" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
