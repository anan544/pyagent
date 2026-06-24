"""
测试 SearchTool — 验证三层降级（rg → findstr → Python）和输出安全。
"""

import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyagent.tools.search import SearchTool


# ═══════════════════════════════════════════════════════════════
# 纯 Python 搜索测试（第 3 层兜底，始终可用）
# ═══════════════════════════════════════════════════════════════

class TestPythonSearch:
    """测试纯 Python pathlib + re 搜索。"""

    def test_basic_search(self):
        """基本搜索：在测试文件中查找匹配行。"""
        tool = SearchTool(max_results=50)
        with tempfile.TemporaryDirectory() as tmp:
            # 创建测试文件
            test_file = Path(tmp) / "test.py"
            test_file.write_text(
                "def hello():\n    pass\n\ndef world():\n    pass\n",
                encoding="utf-8",
            )

            result = asyncio.run(
                tool.execute(pattern="def ", path=tmp, include="*.py")
            )
            assert "def hello()" in result or "def hello" in result
            assert "def world()" in result or "def world" in result
            assert "匹配" in result  # 统计信息

    def test_no_match(self):
        """搜索不存在的模式应返回提示信息。"""
        tool = SearchTool(max_results=50)
        # 在临时目录中搜索，确保没有任何文件包含目标模式
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.py").write_text("hello world", encoding="utf-8")
            result = asyncio.run(
                tool.execute(
                    pattern="completely_missing_pattern_99999",
                    path=tmp,
                    include="*.py",
                )
            )
        assert "未找到" in result

    def test_invalid_regex(self):
        """非法正则表达式应返回错误信息，不能崩溃。"""
        tool = SearchTool(max_results=50)
        # (?P<name... 未闭合 = 必定触发 re.error，确保测试 Python 错误处理路径
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.py").write_text("anything", encoding="utf-8")
            result = asyncio.run(
                tool.execute(pattern=r"(?P<foo", path=tmp, include="*.py")
            )
        assert "错误" in result or "无效" in result

    def test_include_filter(self):
        """include 过滤：只搜 *.txt，不搜 *.py。"""
        tool = SearchTool(max_results=50)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a.py").write_text("hello world", encoding="utf-8")
            Path(tmp, "b.txt").write_text("hello world", encoding="utf-8")

            result = asyncio.run(
                tool.execute(pattern="hello", path=tmp, include="*.txt")
            )
            assert "b.txt" in result
            assert "a.py" not in result

    def test_exclude_filter(self):
        """exclude 过滤：排除 *.log 文件。
        注意：exclude 条件会触发 Python 兜底路径（findstr 不支持 exclude）。"""
        tool = SearchTool(max_results=50)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "keep.py").write_text("needle_xyz", encoding="utf-8")
            Path(tmp, "skip.log").write_text("needle_xyz", encoding="utf-8")

            result = asyncio.run(
                tool.execute(pattern="needle_xyz", path=tmp, exclude="*.log")
            )
            assert "keep.py" in result
            assert "skip.log" not in result

    def test_truncation(self):
        """结果超过 max_results 时应截断并给出提示。"""
        tool = SearchTool(max_results=3)
        with tempfile.TemporaryDirectory() as tmp:
            # 创建包含很多匹配行的文件
            lines = [f"keyword_{i}" for i in range(20)]
            Path(tmp, "many.txt").write_text("\n".join(lines), encoding="utf-8")

            result = asyncio.run(
                tool.execute(pattern="keyword", path=tmp, include="*.txt")
            )
            assert "截断" in result
            assert "前 3 行" in result and "20 行" in result

    def test_binary_file_skipped(self):
        """二进制文件应被跳过，不报错。"""
        tool = SearchTool(max_results=50)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "data.bin").write_bytes(b"\x00\x01\x02\x00")

            result = asyncio.run(
                tool.execute(pattern="anything", path=tmp)
            )
            # 不应崩溃，返回未找到
            assert "未找到" in result or "匹配" in result

    def test_skip_common_dirs(self):
        """应跳过 __pycache__、.git 等目录。
        注意：添加 exclude 条件触发 Python 兜底（findstr 不跳过这些目录）。"""
        tool = SearchTool(max_results=50)
        with tempfile.TemporaryDirectory() as tmp:
            pycache = Path(tmp, "__pycache__")
            pycache.mkdir()
            (pycache / "cached.py").write_text("top_secret_xyz", encoding="utf-8")

            normal = Path(tmp, "normal.py")
            normal.write_text("top_secret_xyz", encoding="utf-8")

            # 用 exclude 触发 Python 路径 → 跳过 __pycache__
            result = asyncio.run(
                tool.execute(
                    pattern="top_secret_xyz", path=tmp,
                    include="*.py", exclude="*.pyc"
                )
            )
            assert "normal.py" in result
            assert "cached" not in result

    def test_nonexistent_path(self):
        """搜索不存在的路径应返回错误。"""
        tool = SearchTool(max_results=50)
        result = asyncio.run(
            tool.execute(pattern="test", path="/nonexistent/path")
        )
        assert "不存在" in result or "错误" in result


# ═══════════════════════════════════════════════════════════════
# 命令探测测试
# ═══════════════════════════════════════════════════════════════

class TestCommandDetection:
    """测试 rg / findstr 的探测逻辑。"""

    def test_is_literal_detection(self):
        """纯字面搜索 vs 正则搜索的判断。"""
        assert SearchTool._is_literal("hello world") is True
        assert SearchTool._is_literal("def function") is True
        assert SearchTool._is_literal("class.*Service") is False
        assert SearchTool._is_literal(r"import\s+os") is False
        # + 是正则量词（一个或多个），不是字面字符
        assert SearchTool._is_literal("a+b") is False
        # \ 也是正则特殊字符（转义符），_is_literal 会保守地认为含 \ 的模式不是纯字面
        assert SearchTool._is_literal(r"a\+b") is False
        assert SearchTool._is_literal("[a-z]") is False

    def test_parse_glob_list(self):
        """Glob 列表解析。"""
        result = SearchTool._parse_glob_list("*.py")
        assert result == ["*.py"]

        result = SearchTool._parse_glob_list("*.py, *.js")
        assert result == ["*.py", "*.js"]

    def test_binary_detection(self):
        """二进制文件检测。"""
        with tempfile.TemporaryDirectory() as tmp:
            text_file = Path(tmp, "test.txt")
            text_file.write_text("hello", encoding="utf-8")
            assert SearchTool._is_binary(text_file) is False

            bin_file = Path(tmp, "test.bin")
            bin_file.write_bytes(b"\x00\x01\x02")
            assert SearchTool._is_binary(bin_file) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
