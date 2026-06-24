"""
条件规则加载器 — 基于 YAML Frontmatter 的按需规则注入。

支持三种加载模式：
    1. 全局规则：无 paths 字段的 .md 文件 → 每次请求都注入
    2. 条件规则：有 paths 字段 → 仅当用户输入中提取的文件路径匹配 glob 时才注入
    3. 显式文件：context_files 中列出的文件 → 始终注入（优先级最高）

规则文件格式 (Markdown + YAML Frontmatter):
    ---
    paths:
      - "src/**/*.test.ts"
      - "tests/**/*.py"
    description: "测试规范"
    ---
    # 测试规范
    - 单元测试必须使用 AAA 模式

用法:
    loader = RuleLoader(
        rules_dir="E:/Harness/pyagent/rules",
        context_files=["rules/special.md"],
    )
    rules_text = loader.load(
        user_input="请修改 src/utils/format.test.ts",
        workspace="E:/thepython/super_study",
    )
"""

import re
from pathlib import Path
from typing import Optional


class RuleLoader:
    """条件规则加载器。

    从 rules_dir 目录加载 .md 规则文件，根据 Frontmatter 的 paths 字段
    判断是否需要注入当前请求的 system prompt。

    性能：
        - 每次调用 load() 都会重新读取文件（热更新支持）
        - Frontmatter 解析使用 yaml.safe_load（已预装）
        - Glob 匹配使用 pathlib.PureWindowsPath.match()（原生支持 **）
    """

    def __init__(
        self,
        rules_dir: str = "",
        context_files: Optional[list[str]] = None,
    ):
        """
        Args:
            rules_dir: 规则目录路径。为空时不加载目录规则。
            context_files: 显式指定的规则文件列表（始终加载）。
        """
        self._rules_dir = Path(rules_dir) if rules_dir else None
        self._context_files: list[str] = list(context_files or [])

    # ── 主入口 ──────────────────────────────────────

    def load(
        self,
        user_input: str = "",
        workspace: str = "",
    ) -> str:
        """加载适用的规则文件，拼接为 system prompt 注入块。

        每次调用都会重新读取磁盘文件，实现热更新。

        Args:
            user_input: 用户输入文本，用于提取文件路径做条件匹配。
            workspace: 工作区根目录，用于规范化相对路径。

        Returns:
            拼接后的规则文本块，无适用规则时返回空字符串。
            格式: "## 项目规范与规则\n[来源: xxx]\n..."
        """
        parts: list[str] = []

        # ── 1. 显式 context_files（始终加载）──
        for file_path in self._context_files:
            content = _read_text(Path(file_path))
            if content:
                parts.append(f"[来源: {file_path}]\n{content}")

        # ── 2. 目录规则（条件加载）──
        if self._rules_dir and self._rules_dir.is_dir():
            target_paths = _extract_paths(user_input, workspace)

            for md_file in sorted(self._rules_dir.glob("*.md")):
                # 避免重复加载（已在 context_files 中）
                if self._is_context_file(str(md_file)):
                    continue

                raw = _read_text(md_file)
                if raw is None:
                    continue

                fm = _parse_frontmatter(raw)
                rule_paths: list[str] = fm.get("paths") or []
                rule_keywords: list[str] = fm.get("keywords") or []

                has_paths = bool(rule_paths)
                has_keywords = bool(rule_keywords)

                # ── 全局规则：无 paths 且无 keywords → 始终加载 ──
                if not has_paths and not has_keywords:
                    parts.append(f"[来源: {md_file}]\n{raw}")
                    continue

                # ── 路径匹配：有 paths 且提取到目标路径 ──
                if has_paths and target_paths:
                    matched = _match_any(target_paths, rule_paths)
                    if matched:
                        parts.append(
                            f"[来源: {md_file} (路径匹配: {matched})]\n{raw}"
                        )
                        continue

                # ── 关键词匹配：有 keywords 且在用户输入中找到 ──
                if has_keywords and _match_keywords(user_input, rule_keywords):
                    # 确定匹配标签
                    if has_paths:
                        tag = "关键词兜底"
                    else:
                        tag = "关键词匹配"
                    kw = _match_keywords(user_input, rule_keywords)
                    parts.append(
                        f"[来源: {md_file} ({tag}: {kw})]\n{raw}"
                    )
                    continue

                # ── 都不匹配 → 跳过（节省 Token）──

        if not parts:
            return ""

        return "## 项目规范与规则\n\n" + "\n\n".join(parts)

    # ── 辅助 ────────────────────────────────────────

    def _is_context_file(self, path_str: str) -> bool:
        """检查路径是否已在 context_files 中。"""
        target = Path(path_str).resolve()
        for cf in self._context_files:
            if Path(cf).resolve() == target:
                return True
        return False


# ═══════════════════════════════════════════════════════════
# 内部工具函数
# ═══════════════════════════════════════════════════════════

def _read_text(path: Path) -> Optional[str]:
    """读取文件文本，不存在时返回 None。"""
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        pass
    return None


def _parse_frontmatter(content: str) -> dict:
    """解析 Markdown 文件的 YAML Frontmatter。

    仅处理文件开头的 --- ... --- 块。
    格式异常时返回空 dict，不抛异常。
    """
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    try:
        import yaml
        fm = yaml.safe_load(content[3:end].strip())
        return fm if isinstance(fm, dict) else {}
    except Exception:
        return {}


def _extract_paths(user_input: str, workspace: str = "") -> list[str]:
    r"""从用户输入文本中提取文件路径。

    识别模式（按优先级）:
        1. Windows 绝对路径: E:\dir\file.py, C:/dir/file.py
        2. Unix 绝对路径: /home/user/file.py
        3. 相对路径（含扩展名）: src/foo.ts, ./bar.py, ..\baz.js
        4. 裸文件名（含常见扩展名）: file.ts, component.vue

    workspace 用于解析相对路径（暂未实现路径拼接，仅做标记）。
    """
    paths: list[str] = []

    # 1. Windows 绝对路径 (E:\... 或 C:/...)
    paths.extend(
        re.findall(r'[A-Za-z]:[\\/][^\s"\'`\n*?]{3,}', user_input)
    )

    # 2. Unix 绝对路径
    paths.extend(
        re.findall(r'/(?:home|usr|opt|var|tmp|etc|mnt)/[^\s"\'`\n]{2,}', user_input)
    )

    # 3. 相对路径（含扩展名）
    paths.extend(
        re.findall(r'(?:\.\.?[\\/])?[\w\-]+[\\/][\w\-\\/]+\.\w{1,8}', user_input)
    )

    # 4. 裸文件名（常见代码扩展名）
    paths.extend(
        re.findall(
            r'\b[\w\-]+\.(?:py|ts|js|jsx|tsx|vue|go|rs|java|cpp|c|h|hpp|css|scss|html|md|yaml|yml|json|toml)\b',
            user_input,
        )
    )

    # 去重并保留顺序
    seen: set[str] = set()
    unique: list[str] = []
    for p in paths:
        if p.lower() not in seen:
            seen.add(p.lower())
            unique.append(p)

    return unique


def _match_any(target_paths: list[str], glob_patterns: list[str]) -> str:
    """检查是否有 target 匹配任一 glob pattern。

    匹配规则：
        - 使用 pathlib.PureWindowsPath.match()，原生支持 ** 递归匹配
        - 同时尝试完整路径匹配和仅文件名匹配
        - 同时尝试 Windows 和 POSIX 风格路径

    Returns:
        匹配的 pattern 字符串，无匹配时返回空字符串。
    """
    from pathlib import PureWindowsPath, PurePosixPath

    for target in target_paths:
        # 规范化路径分隔符
        posix_target = target.replace("\\", "/")
        fname = Path(target).name

        # 尝试列表：完整路径、加 ./ 前缀（修复 ** 不匹配零组件问题）、仅文件名
        candidates = [
            posix_target,
            "./" + posix_target,
            fname,
        ]

        for pattern in glob_patterns:
            for candidate in candidates:
                # POSIX 风格匹配
                try:
                    if PurePosixPath(candidate).match(pattern):
                        return pattern
                except (ValueError, TypeError):
                    pass

                # Windows 风格匹配
                try:
                    if PureWindowsPath(candidate).match(pattern):
                        return pattern
                except (ValueError, TypeError):
                    pass

    return ""


def _match_keywords(user_input: str, keywords: list[str]) -> str:
    """检查用户输入是否包含任一关键词（大小写不敏感）。

    匹配规则：
        - 每个关键词作为一个整体在用户输入中进行子串搜索
        - 中文关键词直接匹配（如 "测试" 匹配 "帮我写个测试"）
        - 英文关键词忽略大小写（如 "pytest" 匹配 "use Pytest"）
        - 支持多词关键词（如 "单元测试" 匹配 "我需要单元测试"）

    Returns:
        匹配到的关键词字符串，无匹配时返回空字符串。
    """
    if not user_input or not keywords:
        return ""

    lower_input = user_input.lower()
    for kw in keywords:
        if not kw.strip():
            continue
        if kw.lower() in lower_input:
            return kw

    return ""
