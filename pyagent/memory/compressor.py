"""
ContextCompressor v2 — 分类型智能上下文压缩 + CCR (Compress, Cache, Retrieve)

核心理念：
    不同内容类型用不同的压缩策略，不搞一刀切。
    JSON 用结构压缩，代码用 AST 签名保留，日志用模式去重。
    压缩后的原文保留在本地缓存，LLM 需要时可调 retrieve_context 拉回原文。

压缩策略：
    ┌──────────────────────────────────────────────────────────┐
    │  ContentRouter: 自动识别 contentType                     │
    │    → JSON   → JSONCompressor  (保留 schema + 采样)       │
    │    → Code   → CodeCompressor  (保留签名 + 压缩体)        │
    │    → Log    → LogCompressor   (去重 + 保留唯一错误)      │
    │    → Text   → LLMCompressor   (调用 LLM 生成结构化摘要)   │
    └──────────────────────────────────────────────────────────┘

    CCR 机制:
    compress(message) → (compressed_text, cache_id)
    retrieve(cache_id) → original_text
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ..core.message import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    Message,
)

logger = logging.getLogger("pyagent.memory.compressor")

# ── 缓存存储 ──────────────────────────────────────────────────
# 内存 LRU dict，存储原文以备按需拉回
_CACHE: dict[str, str] = {}
_CACHE_MAX_SIZE = 100


def _cache_put(text: str) -> str:
    """存入缓存，返回 cache_id。LRU 驱逐。"""
    cid = hashlib.sha256(text.encode()).hexdigest()[:16]
    if len(_CACHE) >= _CACHE_MAX_SIZE:
        oldest = next(iter(_CACHE))
        del _CACHE[oldest]
    _CACHE[cid] = text
    return cid


def retrieve_cache(cache_id: str) -> Optional[str]:
    """按 cache_id 拉回原文。"""
    return _CACHE.get(cache_id)


def clear_cache():
    """清空缓存。"""
    _CACHE.clear()


# ── 压缩结果 ──────────────────────────────────────────────────


@dataclass
class Compressed:
    """压缩结果。"""

    text: str                # 压缩后文本
    cache_ids: list[str] = field(default_factory=list)  # 原文缓存 ID
    method: str = ""         # 使用的压缩方法
    original_chars: int = 0  # 原文字符数
    compressed_chars: int = 0  # 压缩后字符数

    @property
    def ratio(self) -> float:
        """压缩比 (0-1)，越小压缩越狠。"""
        if self.original_chars == 0:
            return 1.0
        return self.compressed_chars / self.original_chars

    @property
    def savings(self) -> str:
        """节省百分比。"""
        return f"{(1 - self.ratio) * 100:.0f}%"


# ── 内容类型检测 ──────────────────────────────────────────────


class ContentRouter:
    """自动识别内容类型，分配最优压缩器。"""

    @staticmethod
    def detect(text: str) -> str:
        """
        返回 content_type: json | code | log | text
        """
        if not text or not text.strip():
            return "text"

        s = text.strip()

        # JSON: 以 { [ 开头，或可解析为 JSON
        if s[0] in "{[":
            try:
                json.loads(s)
                return "json"
            except (json.JSONDecodeError, ValueError):
                pass  # 不是有效 JSON

        # 如果有 JSON 片段嵌入文本中（如 tool_calls JSON）
        if re.search(r'"\w+":\s*"[^"]*"', s) and ("function_name" in s or "tool_calls" in s):
            # 包含 JSON 键值对模式，分离处理
            return "json"

        # Code: 基于关键词密度 + 缩进 + 注释综合判断
        code_patterns = r'\b(def\s|class\s|import\s|from\s\w+\s+import)'
        kw_lines = sum(1 for line in s.split('\n') if re.search(code_patterns, line.strip()))
        has_indent = bool(re.search(r'^(    |\t)\S', s, re.MULTILINE))
        has_semicolon_code = bool(re.search(r'\b(print|return|yield|raise|pass|continue|break)\b', s))
        if kw_lines >= 1 or (has_indent and has_semicolon_code):
            return "code"

        # Log: 时间戳 + 日志级别模式
        log_patterns = sum([
            bool(re.search(r'\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}', s)),
            bool(re.search(r'\b(ERROR|WARN|INFO|DEBUG|TRACE|FATAL)\b', s)),
            bool(re.search(r'(\[ERROR\]|\[WARN\]|\[INFO\]|stack trace|traceback)', s, re.IGNORECASE)),
        ])
        if log_patterns >= 1:
            return "log"

        return "text"


# ── 各类型压缩器 ──────────────────────────────────────────────


class JSONCompressor:
    """
    JSON 压缩器 — 保留 schema/结构，采样数据。

    策略：
        - 提取 JSON 的键结构（保留顶层 keys）
        - 数组只保留前 3 项 + 总数
        - 字符串值截断至 80 字符
        - 嵌套对象保留 2 层
    """

    MAX_ITEMS = 3
    MAX_DEPTH = 2
    MAX_STR_LEN = 80

    def compress(self, text: str) -> Compressed:
        """压缩 JSON 文本。"""
        cid = _cache_put(text)
        try:
            data = json.loads(text)
            compact = self._compact(data, depth=0)
            result = json.dumps(compact, ensure_ascii=False, indent=1)
        except (json.JSONDecodeError, ValueError):
            # 包含 JSON 片段但非纯 JSON — 尝试提取键结构
            result = self._compress_json_fragments(text)

        return Compressed(
            text=result,
            cache_ids=[cid],
            method="json",
            original_chars=len(text),
            compressed_chars=len(result),
        )

    def _compact(self, obj, depth: int = 0):
        """递归压缩 JSON 对象。"""
        if depth > self.MAX_DEPTH:
            if isinstance(obj, dict):
                n_keys = len(obj)
                return {"...": f"{{{n_keys} keys}}"}
            elif isinstance(obj, list):
                return [f"... [{len(obj)} items]"]
            return f"...({type(obj).__name__})"

        if isinstance(obj, dict):
            return {
                k: self._compact(v, depth + 1)
                for k, v in list(obj.items())
            }

        elif isinstance(obj, list):
            compacted = [self._compact(item, depth + 1) for item in obj[: self.MAX_ITEMS]]
            if len(obj) > self.MAX_ITEMS:
                compacted.append(f"... 还有 {len(obj) - self.MAX_ITEMS} 项")
            return compacted

        elif isinstance(obj, str) and len(obj) > self.MAX_STR_LEN:
            return obj[: self.MAX_STR_LEN] + f"...({len(obj)} chars)"

        return obj

    def _compress_json_fragments(self, text: str) -> str:
        """从文本中提取 JSON 键结构。"""
        keys = set(re.findall(r'"(\w+)"\s*:', text))
        lines = [f"JSON fragments — keys detected: {', '.join(sorted(keys))}"]
        # 保留前 500 字符
        if len(text) > 500:
            lines.append(f"(原文 {len(text)} 字符)")
        return "\n".join(lines)


class CodeCompressor:
    """
    代码压缩器 — 保留签名 + 关键结构，压缩实现体。

    策略：
        - 保留 import/from 语句
        - 保留 def/class 签名行
        - 函数体用 "# ... (N lines)" 替代
        - docstring 保留
        - 注释保留（可能含关键信息）
    """

    def compress(self, text: str) -> Compressed:
        """压缩代码。"""
        cid = _cache_put(text)
        lines = text.split("\n")
        result = []
        in_function_body = False
        body_lines = 0

        for line in lines:
            stripped = line.strip()

            # Import 行全保留
            if stripped.startswith(("import ", "from ")):
                if in_function_body and body_lines > 0:
                    result.append(f"    # ... ({body_lines} lines)")
                in_function_body = False
                body_lines = 0
                result.append(line)
                continue

            # 函数/类定义 — 保留签名
            if re.match(r'^\s*(async\s+)?def\s+|^\s*class\s+', stripped):
                if in_function_body and body_lines > 0:
                    result.append(f"    # ... ({body_lines} lines)")
                in_function_body = True
                body_lines = 0
                result.append(line)
                continue

            # docstring — 保留（紧跟在 def/class 后面）
            if in_function_body and re.match(r'^\s*"""', stripped):
                if body_lines == 0:
                    result.append(line)
                else:
                    result.append(f"    # ... ({body_lines} lines)")
                    in_function_body = False
                    body_lines = 0
                    result.append(line)
                continue

            # 注释保留
            if stripped.startswith("#") and not in_function_body:
                result.append(line)
                continue

            # @装饰器保留
            if stripped.startswith("@"):
                result.append(line)
                continue

            # 空行
            if not stripped:
                result.append(line)
                continue

            # 函数体 — 计数
            if in_function_body:
                body_lines += 1
            else:
                result.append(line)

        if in_function_body and body_lines > 0:
            result.append(f"    # ... ({body_lines} lines)")

        compressed_text = "\n".join(result)
        return Compressed(
            text=compressed_text,
            cache_ids=[cid],
            method="code",
            original_chars=len(text),
            compressed_chars=len(compressed_text),
        )


class LogCompressor:
    """
    日志压缩器 — 去重 + 保留唯一错误。

    策略：
        - 相同模式的日志行合并为 "pattern (×N)"
        - ERROR/FATAL 全保留
        - WARN 取前 10 条
        - INFO/DEBUG/TRACE 取前 3 条 + 总数
        - 堆栈去重（相同堆栈只保留第一条）
    """

    def compress(self, text: str) -> Compressed:
        """压缩日志。"""
        cid = _cache_put(text)
        lines = text.split("\n")

        errors = []
        warns = []
        others = []
        stacks = {}
        seen_patterns = {}

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 分类
            if re.search(r'\b(ERROR|FATAL)\b', stripped, re.IGNORECASE):
                sig = self._log_signature(stripped)
                if sig not in stacks:
                    stacks[sig] = stripped
                errors.append(stripped)
            elif re.search(r'\bWARN', stripped, re.IGNORECASE):
                warns.append(stripped)
            else:
                # 合并重复模式
                pattern = re.sub(r'\d+', 'N', stripped)
                pattern = re.sub(r'0x[0-9a-f]+', '0xHEX', pattern, flags=re.IGNORECASE)
                if pattern not in seen_patterns:
                    seen_patterns[pattern] = {"count": 0, "sample": stripped}
                seen_patterns[pattern]["count"] += 1
                others.append(pattern)

        result = []

        # 唯一错误 + 堆栈
        if errors:
            result.append(f"--- 错误 ({len(errors)} 条) ---")
            # 去重 + 前 5 条
            unique_errors = list(dict.fromkeys(errors))[:5]
            for e in unique_errors:
                result.append(e)
                if len(e) > 200:
                    result.append("    ...(截断)")
            if len(errors) > 5:
                result.append(f"    ... 还有 {len(errors) - 5} 条错误")

        if warns:
            unique_warns = list(dict.fromkeys(warns))[:10]
            result.append(f"\n--- 警告 ({len(warns)} 条, 显示前 {len(unique_warns)}) ---")
            result.extend(unique_warns)
            if len(warns) > 10:
                result.append(f"    ... 还有 {len(warns) - 10} 条警告")

        # 模式去重后的其他行
        if seen_patterns:
            result.append(f"\n--- 常规日志 (去重后 {len(seen_patterns)} 种模式) ---")
            for pattern, info in sorted(
                seen_patterns.items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            )[:5]:
                suffix = f" (×{info['count']})" if info["count"] > 1 else ""
                result.append(f"  [{info['count']}次] {info['sample'][:120]}{suffix}")

        compressed_text = "\n".join(result)
        return Compressed(
            text=compressed_text,
            cache_ids=[cid],
            method="log",
            original_chars=len(text),
            compressed_chars=len(compressed_text),
        )

    @staticmethod
    def _log_signature(line: str) -> str:
        """提取错误签名（去时序/进程号/行号等变体）。"""
        sig = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}[.,\d]*', '[TIME]', line)
        sig = re.sub(r'pid[=:]\d+', 'pid=N', sig, flags=re.IGNORECASE)
        sig = re.sub(r'line\s*\d+', 'line N', sig, flags=re.IGNORECASE)
        sig = re.sub(r'0x[0-9a-f]+', '0xHEX', sig, flags=re.IGNORECASE)
        return sig


class LLMTextCompressor:
    """
    LLM 文本压缩器 — 保留现有的结构化摘要能力。

    用于处理对话、文档等非结构化文本。
    与原有 ContextCompressor 功能一致。
    """

    # ── 压缩 Prompt 模板 ──
    COMPRESSION_SYSTEM_PROMPT = """\
你是一个项目的技术负责人，正在对团队的开发对话进行结构化总结。
你的总结将被用作 Agent 的"长期记忆"，后续任务将依赖这份摘要来理解上下文。

总结规则：
1. 只提取工程事实（文件名、函数名、决策、Bug），忽略闲聊和情感表达。
2. 按以下固定格式输出，每个字段都必须填写，没有内容则写"无"。
3. 尽量压缩冗余信息，但不要丢失关键的技术细节。
4. 工具调用的结果只保留结论性信息，丢弃原始输出的过程日志。
"""

    COMPRESSION_USER_TEMPLATE = """\
请将以下对话历史压缩为结构化摘要。严格按格式输出：

===== 开始对话历史 =====
{conversation}
===== 结束对话历史 =====

请按以下格式输出（不要输出其他内容）：

【已达成共识】
- [列出已确认的需求、设计决策、代码变更结论]
- [无则写"无"]

【待解决问题】
- [列出未完成的 Bug、未决定的设计选项、TODO]
- [无则写"无"]

【关键文件路径】
- [列出所有被读取、修改、提及的代码文件路径]
- [无则写"无"]

【工具调用结论】
- [总结工具执行的关键结果和发现]
- [无则写"无"]

【关键实体】
- [列出重要的函数名、类名、变量名、技术术语]
- [无则写"无"]
"""

    SUMMARY_ROLE = "summary"

    def __init__(self, llm_provider=None, max_summary_tokens: int = 2000):
        self.llm = llm_provider
        self.max_summary_tokens = max_summary_tokens

    async def compress(self, messages: list[Message]) -> Compressed:
        """调用 LLM 将消息压缩为结构化摘要。"""
        if not messages:
            return Compressed(
                text="无",
                method="llm_text",
                original_chars=0,
                compressed_chars=0,
            )

        conversation_text = self._messages_to_text(messages)
        cid = _cache_put(conversation_text)

        # 如果 LLM 不可用，降级
        if self.llm is None:
            return self._fallback(messages, cid)

        compress_msgs = [
            SystemMessage(content=self.COMPRESSION_SYSTEM_PROMPT),
            UserMessage(
                content=self.COMPRESSION_USER_TEMPLATE.format(
                    conversation=conversation_text
                )
            ),
        ]

        try:
            response = await self.llm.generate(compress_msgs, tools=[])
            summary = (response.content or "").strip()
            if "【" not in summary:
                return self._fallback(messages, cid)
            return Compressed(
                text=summary,
                cache_ids=[cid],
                method="llm_text",
                original_chars=len(conversation_text),
                compressed_chars=len(summary),
            )
        except Exception:
            return self._fallback(messages, cid)

    def _messages_to_text(self, messages: list[Message]) -> str:
        """消息列表 → 可读文本。"""
        lines = []
        total = 0
        limit = self.max_summary_tokens * 4

        for msg in messages:
            role = msg.role
            if isinstance(msg, UserMessage):
                line = f"[用户]: {msg.content or ''}"
            elif isinstance(msg, AssistantMessage):
                parts = [msg.content or ""]
                if msg.tool_calls:
                    parts.append(f"[调用: {', '.join(tc.function_name for tc in msg.tool_calls)}]")
                line = f"[助手]: {' '.join(parts)}"
            elif isinstance(msg, ToolMessage):
                content = (msg.content or "")[:500]
                line = f"[工具 {msg.name}]: {content}"
            elif isinstance(msg, SystemMessage):
                continue
            else:
                line = f"[{role}]: {msg.content or ''}"

            if total + len(line) > limit:
                lines.append("...(早期对话已截断)")
                break
            lines.append(line)
            total += len(line)

        return "\n".join(lines)

    def _fallback(self, messages: list[Message], cid: str = "") -> Compressed:
        """降级：从消息中直接提取文件路径和工具名。"""
        files = set()
        tools = set()
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tools.add(msg.name)
            content = msg.content or ""
            for word in content.split():
                if any(word.endswith(ext) for ext in (".py", ".js", ".ts", ".go",
                        ".rs", ".java", ".md", ".txt", ".json", ".yaml", ".toml")):
                    files.add(word)

        text = (
            f"【已达成共识】\n- (自动降级摘要)\n\n"
            f"【待解决问题】\n无\n\n"
            f"【关键文件路径】\n" +
            "\n".join(f"- {f}" for f in sorted(files) if files) +
            ("" if files else "- 无") +
            f"\n\n【工具调用结论】\n- 使用了: {', '.join(sorted(tools)) if tools else '无'}\n\n"
            f"【关键实体】\n无"
        )
        return Compressed(
            text=text,
            cache_ids=[cid] if cid else [],
            method="llm_text_fallback",
            original_chars=0,
            compressed_chars=len(text),
        )


# ── 门面：ContextCompressor v2 ────────────────────────────────


@dataclass
class CompressionReport:
    """一次压缩操作的完整报告。"""
    total_original: int = 0
    total_compressed: int = 0
    details: list[Compressed] = field(default_factory=list)

    @property
    def overall_savings(self) -> str:
        if self.total_original == 0:
            return "0%"
        pct = (1 - self.total_compressed / self.total_original) * 100
        return f"{pct:.0f}%"

    @property
    def cache_ids(self) -> list[str]:
        """所有压缩结果的 cache IDs（用于 retrieve）。"""
        ids = []
        for d in self.details:
            ids.extend(d.cache_ids)
        return ids


class ContextCompressor:
    """
    上下文压缩器 v2 — 分类型智能压缩。

    使用方式:
        compressor = ContextCompressor(llm_provider)
        result = await compressor.compress(messages)
        print(f"节省: {result.report.overall_savings}")

        # 需要原文时
        original = retrieve_cache(cid)
    """

    def __init__(self, llm_provider=None, max_summary_tokens: int = 2000):
        self.router = ContentRouter()
        self.json_comp = JSONCompressor()
        self.code_comp = CodeCompressor()
        self.log_comp = LogCompressor()
        self.text_comp = LLMTextCompressor(llm_provider, max_summary_tokens)
        self.max_summary_tokens = max_summary_tokens

    async def compress(self, messages: list[Message]) -> str:
        """
        压缩消息列表。兼容旧接口（返回纯文本）。

        内部为每条消息匹配最优压缩器，最后整合输出。
        """
        if not messages:
            return ""

        report = CompressionReport()
        compressed_parts = []

        for msg in messages:
            content = msg.content or ""
            if not content.strip():
                # 只有 tool_calls 的 AssistantMessage
                if isinstance(msg, AssistantMessage) and msg.tool_calls:
                    content = json.dumps([
                        {"fn": tc.function_name, "args": tc.arguments}
                        for tc in msg.tool_calls
                    ], ensure_ascii=False)

            if not content.strip():
                continue

            content_type = self.router.detect(content)

            if content_type == "json":
                cr = self.json_comp.compress(content)
            elif content_type == "code":
                cr = self.code_comp.compress(content)
            elif content_type == "log":
                cr = self.log_comp.compress(content)
            else:
                # 文本类消息 — 批量交给 LLM 压缩
                continue  # 最后统一处理

            compressed_parts.append(
                f"[{content_type} | 节省 {cr.savings}]\n{cr.text}"
            )
            report.total_original += cr.original_chars
            report.total_compressed += cr.compressed_chars
            report.details.append(cr)

        # 文本类消息用 LLM 统一压缩
        text_messages = [
            m for m in messages
            if (m.content or "").strip()
            and self.router.detect(m.content or "") == "text"
        ]
        if text_messages:
            cr = await self.text_comp.compress(text_messages)
            compressed_parts.append(cr.text)
            report.total_original += cr.original_chars
            report.total_compressed += cr.compressed_chars
            report.details.append(cr)

        return "\n\n".join(compressed_parts)

    def compress_sync(self, text: str) -> Compressed:
        """同步压缩单段文本（不调 LLM）。"""
        ct = self.router.detect(text)
        if ct == "json":
            return self.json_comp.compress(text)
        elif ct == "code":
            return self.code_comp.compress(text)
        elif ct == "log":
            return self.log_comp.compress(text)
        else:
            cid = _cache_put(text)
            # 文本压缩需要 LLM，这里简单截断
            truncated = text[:self.max_summary_tokens * 4]
            suffix = f"...(截断, {len(text) - len(truncated)} chars)" if len(text) > len(truncated) else ""
            return Compressed(
                text=truncated + suffix,
                cache_ids=[cid],
                method="text_truncate",
                original_chars=len(text),
                compressed_chars=len(truncated) + len(suffix),
            )

    async def compress_report(self, messages: list[Message]) -> CompressionReport:
        """压缩并返回完整报告（含 cache_ids 供按需检索）。"""
        report = CompressionReport()

        for msg in messages:
            content = msg.content or ""
            if not content.strip():
                continue

            content_type = self.router.detect(content)
            if content_type in ("json", "code", "log"):
                if content_type == "json":
                    cr = self.json_comp.compress(content)
                elif content_type == "code":
                    cr = self.code_comp.compress(content)
                else:
                    cr = self.log_comp.compress(content)
                report.details.append(cr)
                report.total_original += cr.original_chars
                report.total_compressed += cr.compressed_chars

        # 文本类批量
        text_msgs = [
            m for m in messages
            if (m.content or "").strip()
            and self.router.detect(m.content or "") == "text"
        ]
        if text_msgs:
            cr = await self.text_comp.compress(text_msgs)
            report.details.append(cr)
            report.total_original += cr.original_chars
            report.total_compressed += cr.compressed_chars

        return report


# ── retrieve_context 工具描述 ──────────────────────────────

# 兼容旧接口
SUMMARY_ROLE = "summary"

RETRIEVE_CONTEXT_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "cache_id": {
            "type": "string",
            "description": "压缩摘要中附带的 cache_id，用于拉回原文",
        },
    },
    "required": ["cache_id"],
}
