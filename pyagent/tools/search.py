"""
SearchTool — 代码库内容搜索工具。

实现策略（优雅降级）：
    1. 首选 rg (ripgrep) — Rust 实现，速度极快，天然忽略 .gitignore
    2. 降级 findstr   — Windows 原生，无需额外安装
    3. 兜底纯 Python  — pathlib + re，保证在任何环境都能跑

输出安全：
    - 默认最多返回 100 行，超出自动截断并提示
    - 防止 LLM 上下文被大量匹配结果撑爆
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from .base import Tool


MAX_RESULTS = 100
TIMEOUT_SECONDS = 30


class SearchTool(Tool):
    """
    在代码库中搜索匹配指定模式的文件内容。

    相当于 grep/rg 的 Agent 友好封装。支持正则表达式，
    可限定搜索目录和文件类型。
    """

    name = "search_content"
    risk_level = "low"
    description = (
        "使用正则表达式在代码库中搜索特定文本模式或符号。\n"
        "\n"
        "当您知道要查找的确切函数名、变量或符号时使用此工具。对于精确匹配，这比语义搜索更快更准。\n"
        "始终转义特殊正则字符（. * + ? ^ $ ( ) [ ] { } |）。\n"
        "搜索结果最多 100 行，超出会被截断。如果结果被截断，请缩小搜索范围。\n"
        "适用于：查找函数定义、类声明、import 语句、错误信息、特定模式的代码片段。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "要搜索的正则表达式模式。示例: 'def calculate', 'class.*Service', 'import os'",
            },
            "path": {
                "type": "string",
                "description": "限定搜索的目录路径。默认从当前工作目录开始搜索。",
                "default": ".",
            },
            "include": {
                "type": "string",
                "description": "只搜索匹配此 glob 模式的文件，如 '*.py' 或 '*.{js,ts}'。不填则搜索所有文本文件。",
            },
            "exclude": {
                "type": "string",
                "description": "排除匹配此 glob 模式的文件或目录，如 '*.log' 或 '*.pyc'。",
            },
            "explanation": {
                "type": "string",
                "description": "一句话说明你在搜索什么以及为什么。",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, max_results: int = MAX_RESULTS):
        self.max_results = max_results
        # 缓存命令探测结果
        self._rg_path: str | None = None
        self._findstr_available: bool | None = None

    # ── 主入口 ──

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        include: str = "",
        exclude: str = "",
        cwd: str | None = None,
        **kwargs,
    ) -> str:
        """
        执行内容搜索。按 rg → findstr → Python 的顺序尝试。
        cwd: 工作目录。path 为相对路径时基于此目录解析。
        """
        # ★ 解析搜索路径：相对路径基于 cwd 而非后端目录
        search_path = Path(path)
        if cwd and not search_path.is_absolute():
            search_path = Path(cwd) / search_path
        search_path = search_path.expanduser().resolve()
        if not search_path.exists():
            return f"[错误] 搜索路径不存在: {search_path}"

        # ── 第 1 层：ripgrep ──
        result = await self._try_ripgrep(pattern, search_path, include, exclude)
        if result is not None:
            return result

        # ── 第 2 层：Windows findstr ──
        result = await self._try_findstr(pattern, search_path, include, exclude)
        if result is not None:
            return result

        # ── 第 3 层：纯 Python 兜底 ──
        return await self._python_search(pattern, search_path, include, exclude)

    # ═══════════════════════════════════════════════════════════
    # 第 1 层：ripgrep (rg)
    # ═══════════════════════════════════════════════════════════

    async def _try_ripgrep(self, pattern, search_path, include, exclude) -> str | None:
        """尝试用 rg 搜索，不可用时返回 None。"""
        if self._rg_path is None:
            self._rg_path = shutil.which("rg") or shutil.which("ripgrep") or ""

        if not self._rg_path:
            return None  # rg 未安装，透明降级

        cmd = [
            self._rg_path,
            "--line-number",      # 显示行号
            "--no-heading",       # 不显示文件名分组头
            "--color", "never",   # 不要颜色转义码
            "--no-messages",      # 抑制"没有匹配"等消息
            "--max-count", str(self.max_results),  # 限制匹配数
            "-e", pattern,        # 搜索模式
            str(search_path),
        ]

        # rg 原生支持 glob 过滤
        if include:
            for glob_pattern in self._parse_glob_list(include):
                cmd.extend(["--glob", glob_pattern])

        try:
            proc = await self._run_subprocess(cmd)
            if proc.returncode == 0 and proc.stdout:
                return self._format_output(proc.stdout, "rg")
            elif proc.returncode == 1:
                # rg 返回 1 = 没有匹配
                return f"(未找到匹配 '{pattern}' 的内容)"
            else:
                return None  # 其他错误，降级
        except Exception:
            return None  # 任何异常都降级

    # ═══════════════════════════════════════════════════════════
    # 第 2 层：Windows findstr
    # ═══════════════════════════════════════════════════════════

    async def _try_findstr(self, pattern, search_path, include, exclude) -> str | None:
        """
        尝试用 Windows findstr 搜索。

        findstr 的局限性：
            - 不支持标准的正则表达式（只支持非常基础的子集）
            - 不支持 exclude 过滤
            - 不跳过 __pycache__ 等目录

        因此 findstr 只在以下条件同时满足时才使用：
            1. Windows 系统
            2. 纯字面搜索（无正则特殊字符）
            3. 没有 exclude 过滤（否则无法排除文件）
        """
        if os.name != "nt":
            return None

        # 只处理简单的字面搜索
        if not self._is_literal(pattern) or exclude:
            return None  # 降级到 Python 兜底

        if self._findstr_available is None:
            self._findstr_available = shutil.which("findstr") is not None

        if not self._findstr_available:
            return None

        # findstr 参数说明：
        #   /s  递归子目录
        #   /n  显示行号
        #   /r  使用正则表达式（有限支持）
        #   /c: 字面搜索（当 pattern 不含正则特殊字符时用这个更可靠）
        # 判断是字面搜索还是正则搜索
        if self._is_literal(pattern):
            cmd = ["findstr", "/s", "/n", "/c:" + pattern]
        else:
            cmd = ["findstr", "/s", "/n", "/r", "/c:" + pattern]

        # findstr 的文件过滤用通配符拼接
        file_filter = self._build_findstr_filter(search_path, include)
        if file_filter:
            cmd.append(file_filter)
        else:
            cmd.append(str(search_path / "*"))

        try:
            proc = await self._run_subprocess(cmd)
            if proc.returncode == 0 and proc.stdout:
                return self._format_output(proc.stdout, "findstr")
            elif proc.returncode == 1:
                return f"(未找到匹配 '{pattern}' 的内容)"
            else:
                return None
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════
    # 第 3 层：纯 Python 兜底 (pathlib + re)
    # ═══════════════════════════════════════════════════════════

    async def _python_search(self, pattern, search_path, include, exclude) -> str:
        """纯 Python 实现 — 永远可用，但速度较慢。"""
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"[错误] 正则表达式无效: {e}"

        results: list[str] = []

        # 遍历文件
        for file_path in search_path.rglob("*"):
            if not file_path.is_file():
                continue

            # 跳过隐藏目录和常见无需搜索的目录
            if self._should_skip_dir(file_path):
                continue

            # include / exclude 过滤
            if include and not self._match_glob(file_path, include):
                continue
            if exclude and self._match_glob(file_path, exclude):
                continue

            # 跳过二进制和过大文件
            if self._is_binary(file_path):
                continue
            if file_path.stat().st_size > 500 * 1024:  # 500KB 限制
                continue

            # 搜索文件内容
            file_results = self._search_in_file(file_path, regex)
            results.extend(file_results)

            if len(results) >= self.max_results:
                break

        if not results:
            return f"(未找到匹配 '{pattern}' 的内容)"

        return self._format_output("\n".join(results), "python")

    def _search_in_file(self, file_path: Path, regex: re.Pattern) -> list[str]:
        """在单个文件中搜索匹配行。"""
        matches = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, 1):
                    if regex.search(line):
                        rel = self._relative_path(file_path)
                        matches.append(f"{rel}:{line_no}:{line.rstrip()}")
                        if len(matches) >= self.max_results:
                            break
        except Exception:
            pass  # 无法读取的文件静默跳过
        return matches

    # ═══════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    async def _run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess:
        """
        在线程池中运行 subprocess（避免阻塞事件循环）。
        统一处理 UTF-8 编码，errors='replace' 防止 GBK 终端乱码崩溃。
        """
        import asyncio

        def _run():
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                encoding="utf-8",
                errors="replace",
            )

        return await asyncio.to_thread(_run)

    def _format_output(self, raw: str, source: str) -> str:
        """截断过长结果，添加来源和统计信息。"""
        lines = raw.strip().split("\n")
        total = len(lines)

        if total > self.max_results:
            lines = lines[: self.max_results]
            suffix = (
                f"\n... 结果已截断（显示前 {self.max_results} 行，共 {total} 行匹配）。"
                f"\n请提供更精确的搜索模式以减少结果。"
            )
        else:
            suffix = f"\n(共 {total} 行匹配)"

        header = f"[{source}] 搜索模式匹配结果:\n"
        return header + "\n".join(lines) + suffix

    @staticmethod
    def _is_literal(pattern: str) -> bool:
        """判断 pattern 是否为纯字面搜索（不含正则特殊字符）。"""
        special_chars = r".^$*+?{}[]\|()"
        return not any(c in pattern for c in special_chars)

    @staticmethod
    def _parse_glob_list(pattern_str: str) -> list[str]:
        """解析 '*/.py' 或 '*/.{js,ts}' 这样的 glob 字符串。"""
        patterns = [p.strip() for p in pattern_str.split(",") if p.strip()]
        return patterns

    @staticmethod
    def _match_glob(file_path: Path, glob_str: str) -> bool:
        """检查文件路径是否匹配 glob 模式。"""
        patterns = [p.strip() for p in glob_str.split(",") if p.strip()]
        for pat in patterns:
            if file_path.match(pat):
                return True
        return False

    @staticmethod
    def _should_skip_dir(file_path: Path) -> bool:
        """跳过常见的非代码目录。"""
        skip_dirs = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
            "dist", "build", ".egg-info", ".next", ".nuxt",
        }
        return any(part in skip_dirs for part in file_path.parts)

    @staticmethod
    def _is_binary(file_path: Path) -> bool:
        """简单检测是否为二进制文件。"""
        # 先按后缀快速判断
        text_extensions = {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
            ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
            ".scala", ".r", ".R", ".pl", ".sh", ".bash", ".zsh",
            ".txt", ".md", ".rst", ".csv", ".json", ".xml", ".yaml", ".yml",
            ".toml", ".ini", ".cfg", ".conf", ".html", ".css", ".scss",
            ".sql", ".graphql", ".proto",
        }
        if file_path.suffix.lower() in text_extensions:
            return False
        # 无后缀或未知后缀：尝试读取前几个字节
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
            return b"\x00" in chunk  # null byte = 二进制
        except Exception:
            return True

    @staticmethod
    def _relative_path(file_path: Path) -> str:
        """返回相对于当前工作目录的路径。"""
        try:
            return str(file_path.relative_to(Path.cwd()))
        except ValueError:
            return str(file_path)

    @staticmethod
    def _build_findstr_filter(search_path: Path, include: str) -> str | None:
        """为 findstr 构建文件过滤参数。"""
        if include:
            # findstr 不支持复杂的 glob，取第一个
            first = include.split(",")[0].strip()
            return str(search_path / first)
        return None
