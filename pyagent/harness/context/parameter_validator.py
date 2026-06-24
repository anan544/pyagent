"""
参数白名单校验器 — Layer 1.3 参数级安全检查。

对每种高危工具定义参数合法性规则：
    - execute_command: 命令前缀白名单 + 危险模式黑名单
    - http_request: 域名白名单 + 内网 IP 黑名单
    - file 类操作（write_file/read_file/delete_file）: 路径穿越检测 + 工作区外访问拦截

所有规则均为同步 O(1) 字符串/正则匹配，确保低延迟。

使用方式：
    validator = ParameterWhitelistValidator()
    decision = validator.validate("execute_command", {"command": "rm -rf /"})
    if decision is not None:
        print(decision.blocked_message)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from .security_governance import SecurityDecision

logger = logging.getLogger("pyagent.security.param_validator")


# ═══════════════════════════════════════════════════════════════
# 默认规则集
# ═══════════════════════════════════════════════════════════════

# execute_command: 危险命令模式（正则）
DEFAULT_BLOCKED_COMMAND_PATTERNS: list[str] = [
    r'rm\s+-rf\s+/',              # 递归强制删除根目录
    r'>\s*/dev/',                  # 写入设备文件
    r'mkfs\.',                     # 格式化文件系统
    r'dd\s+if=',                   # 裸磁盘操作
    r':\(\)\s*\{',                 # Fork 炸弹特征
    r'chmod\s+777\s+/',            # 根目录全局可写
    r'wget\s+.*\|\s*sh',           # 下载管道到 shell
    r'curl\s+.*\|\s*sh',           # 下载管道到 shell
    r'>\s*/etc/',                   # 写入 /etc 系统配置
    r'systemctl\s+(stop|disable)',  # 停止/禁用系统服务
]

# execute_command: 允许的命令前缀
DEFAULT_ALLOWED_COMMAND_PREFIXES: list[str] = [
    # ── 脚本语言 ──
    "python", "python3", "pytest", "pip", "pip3",
    "node", "npm", "npx", "yarn", "pnpm",
    "ruby", "gem", "bundle", "rake",
    "perl", "php", "lua",
    # ── 编译语言 ──
    "go", "cargo", "rustc", "rustup",
    "java", "javac", "mvn", "gradle",
    "dotnet", "nuget",
    "make", "cmake", "ninja", "meson",
    "gcc", "g++", "clang", "clang++",
    # ── 版本控制 ──
    "git", "svn", "hg",
    # ── 容器与虚拟化 ──
    "docker", "docker-compose", "podman", "kubectl", "helm",
    # ── 系统管理 ──
    "ls", "dir", "cat", "type", "head", "tail", "find", "where",
    "grep", "rg", "echo", "mkdir", "cp", "mv", "rm", "rmdir",
    "cd", "pwd", "touch", "ln", "chmod", "chown", "chgrp",
    "ps", "kill", "taskkill", "tasklist", "top", "htop",
    "df", "du", "free", "uname", "whoami", "id", "env",
    "export", "source", "set", "printenv",
    "netstat", "ss", "lsof", "ping", "traceroute", "nslookup",
    # ── 文件处理 ──
    "tar", "gzip", "gunzip", "zip", "unzip", "xz", "bzip2",
    "curl", "wget",
    "sed", "awk", "tr", "cut", "sort", "uniq", "wc",
    "diff", "patch", "xargs", "tee", "basename", "dirname", "readlink",
    # ── 网络服务 ──
    "uvicorn", "gunicorn", "daphne", "http-server", "live-server",
    "ssh", "scp", "rsync",
    # ── 数据库 CLI ──
    "mysql", "mysqldump", "psql", "sqlite3", "redis-cli", "mongosh",
    # ── Python 生态 ──
    "poetry", "pipenv", "conda", "mamba", "virtualenv", "venv",
    "flake8", "black", "isort", "mypy", "ruff", "bandit",
    "pre-commit",
    # ── JS/TS 生态 ──
    "npx", "tsc", "eslint", "prettier", "jest", "vitest", "webpack", "vite",
    # ── Git 扩展 ──
    "gh", "glab",
]

# http_request: 默认阻止的内网 IP 范围
DEFAULT_BLOCKED_IP_PATTERNS: list[str] = [
    r'^10\.',                                    # A 类私有
    r'^172\.(1[6-9]|2\d|3[01])\.',               # B 类私有
    r'^192\.168\.',                               # C 类私有
    r'^127\.',                                    # 本地回环
    r'^169\.254\.',                               # 链路本地
    r'^0\.0\.0\.0$',                              # 零地址
    r'^localhost$',                               # localhost
    r'^\[::1\]$',                                 # IPv6 回环
    r'^fc00:',                                     # IPv6 唯一本地
]


class ParameterWhitelistValidator:
    """参数白名单校验器。

    支持的校验类型：
        - execute_command: 命令前缀 + 危险模式
        - http_request: 域名/IP 检查
        - write_file / read_file / delete_file: 路径安全检查
    """

    def __init__(
        self,
        allowed_command_prefixes: Optional[list[str]] = None,
        blocked_command_patterns: Optional[list[str]] = None,
        domain_allowlist: Optional[list[str]] = None,
        blocked_ip_patterns: Optional[list[str]] = None,
    ):
        """
        Args:
            allowed_command_prefixes: execute_command 允许的命令前缀。
                                      空列表或 None 表示使用默认值。
            blocked_command_patterns: execute_command 额外阻止的正则模式
                                      （会合并到默认列表中）。
            domain_allowlist: http_request 允许的域名列表。
                              空列表表示允许所有域名。
            blocked_ip_patterns: http_request 额外阻止的 IP 正则
                                 （会合并到默认列表中）。
        """
        self._allowed_prefixes = (
            allowed_command_prefixes
            if allowed_command_prefixes
            else DEFAULT_ALLOWED_COMMAND_PREFIXES
        )
        self._blocked_patterns = list(DEFAULT_BLOCKED_COMMAND_PATTERNS)
        if blocked_command_patterns:
            self._blocked_patterns.extend(blocked_command_patterns)
        self._domain_allowlist = domain_allowlist or []
        self._blocked_ips = list(DEFAULT_BLOCKED_IP_PATTERNS)
        if blocked_ip_patterns:
            self._blocked_ips.extend(blocked_ip_patterns)

    # ── 主入口 ────────────────────────────────────

    def validate(
        self, tool_name: str, params: dict,
    ) -> Optional[SecurityDecision]:
        """校验工具参数。

        Args:
            tool_name: 工具名称。
            params: 工具参数字典。

        Returns:
            SecurityDecision 若阻止，None 若通过。
        """
        if tool_name == "execute_command":
            return self._validate_command(params)
        elif tool_name == "execute_python":
            return self._validate_python_code(params)
        elif tool_name in ("http_request", "fetch_url", "web_fetch"):
            return self._validate_http(params)
        elif tool_name in ("write_file", "read_file", "delete_file"):
            return self._validate_path(params, tool_name)
        return None  # 无特定规则

    # ── 命令校验 ──────────────────────────────────

    def _validate_command(self, params: dict) -> Optional[SecurityDecision]:
        """校验 execute_command 参数。"""
        command = params.get("command", "")
        if not command:
            return SecurityDecision.block(
                "param_whitelist:missing_command",
                risk_score=70,
                reason="execute_command 缺少 'command' 参数。",
                tool_name="execute_command",
            )

        # 1. 危险模式检查（优先级高，先执行）
        for pattern in self._blocked_patterns:
            if re.search(pattern, command):
                return SecurityDecision.block(
                    "param_whitelist:blocked_pattern",
                    risk_score=90,
                    reason=f"命令包含被禁止的模式: {pattern}",
                    tool_name="execute_command",
                )

        # 2. 命令前缀白名单
        cmd_trimmed = command.strip()
        if not cmd_trimmed:
            return SecurityDecision.block(
                "param_whitelist:empty_command",
                risk_score=60,
                reason="命令为空。",
                tool_name="execute_command",
            )

        # 提取命令的第一个词（处理引号包裹的情况）
        first_word = cmd_trimmed.split()[0] if cmd_trimmed else ""

        if first_word and self._allowed_prefixes:
            # 前缀匹配（如 "python" 匹配 "python3", "python -m"）
            if not any(
                first_word == prefix or first_word.startswith(prefix)
                for prefix in self._allowed_prefixes
            ):
                return SecurityDecision.block(
                    "param_whitelist:unknown_prefix",
                    risk_score=60,
                    reason=(
                        f"命令 '{first_word}' 不在允许的前缀列表中。"
                        f"允许的前缀: {', '.join(self._allowed_prefixes[:10])}..."
                    ),
                    tool_name="execute_command",
                )

        return None

    # ── HTTP 校验 ─────────────────────────────────

    def _validate_http(self, params: dict) -> Optional[SecurityDecision]:
        """校验 HTTP 请求参数。"""
        url = params.get("url", "")
        if not url:
            return SecurityDecision.block(
                "param_whitelist:missing_url",
                risk_score=70,
                reason="HTTP 请求缺少 'url' 参数。",
                tool_name="http_request",
            )

        # 1. 协议检查 — 阻止 file:// 等危险协议
        if re.match(r'^file://', url, re.IGNORECASE):
            return SecurityDecision.block(
                "param_whitelist:blocked_protocol",
                risk_score=95,
                reason="禁止使用 file:// 协议。",
                tool_name="http_request",
            )

        # 2. 提取主机名
        host_match = re.search(r'://([^/:]+)', url)
        if not host_match:
            return None  # 无法解析 → 放行（由工具层处理错误）

        host = host_match.group(1)

        # 3. IP 黑名单检查
        for pattern in self._blocked_ips:
            if re.match(pattern, host):
                return SecurityDecision.block(
                    "param_whitelist:blocked_ip",
                    risk_score=85,
                    reason=f"禁止访问内网/保留 IP 地址: {host}",
                    tool_name="http_request",
                )

        # 4. 域名白名单检查（若配置了白名单）
        if self._domain_allowlist:
            if not any(
                host == allowed or host.endswith("." + allowed)
                for allowed in self._domain_allowlist
            ):
                return SecurityDecision.block(
                    "param_whitelist:domain_not_allowed",
                    risk_score=70,
                    reason=f"域名 '{host}' 不在允许列表中。",
                    tool_name="http_request",
                )

        return None

    # ── 文件路径校验 ──────────────────────────────

    def _validate_path(
        self, params: dict, tool_name: str,
    ) -> Optional[SecurityDecision]:
        """校验文件操作路径（路径穿越 + 工作区外访问）。"""
        # 提取路径（不同工具的键名可能不同）
        path = params.get("path") or params.get("file_path") or ""
        if not path:
            return None  # 无路径可校验

        # 1. 路径穿越检测
        normalized = os.path.normpath(path)
        segments = normalized.replace("\\", "/").split("/")
        if ".." in segments:
            return SecurityDecision.block(
                "param_whitelist:path_traversal",
                risk_score=85,
                reason=f"文件路径包含目录穿越 '..': {path}",
                tool_name=tool_name,
            )

        # 2. 绝对路径 → 检查是否在工作区内
        if os.path.isabs(normalized):
            cwd = os.getcwd()
            try:
                common = os.path.commonpath([cwd, normalized])
                if common != cwd:
                    return SecurityDecision.block(
                        "param_whitelist:outside_workspace",
                        risk_score=70,
                        reason=f"不允许访问工作区外的绝对路径: {path}",
                        tool_name=tool_name,
                    )
            except ValueError:
                # Windows 不同盘符无法比较
                return SecurityDecision.block(
                    "param_whitelist:cross_drive",
                    risk_score=70,
                    reason=f"不允许跨盘符访问: {path}",
                    tool_name=tool_name,
                )

        return None

    # ── Python 代码校验 ────────────────────────────

    # execute_python 危险模式（正则）
    DEFAULT_BLOCKED_PYTHON_PATTERNS: list[str] = [
        r'os\.system\s*\(',           # 调用系统命令
        r'subprocess\.(call|run|Popen)\s*\(',  # 子进程调用
        r'__import__\s*\(\s*[\'\"]os[\'\"]',   # 动态导入 os 模块
        r'exec\s*\(',                  # 动态执行代码
        r'compile\s*\(.*eval\s*\)',    # 编译并执行
        r'ctypes\.',                   # C 类型调用
        r'shutil\.rmtree\s*\(',        # 递归删除目录
        r'shutil\.(copy|move)\s*\(.*[\'\"]/etc',  # 复制/移动到系统目录
        r'open\s*\(.*[\'\"]/etc/',     # 读取系统配置文件
        r'socket\.',                   # 网络 socket
        r'requests\.(get|post|put|delete)\s*\(',  # 网络请求
    ]

    MAX_PYTHON_CODE_LENGTH = 100_000   # 100KB

    def _validate_python_code(
        self, params: dict,
    ) -> Optional[SecurityDecision]:
        """校验 execute_python 代码内容。

        检查项：
            1. 代码长度限制（默认 100KB）
            2. 危险模式检测（系统调用/子进程/动态执行/网络访问）
            3. 缺失 code 参数
        """
        code = params.get("code", "")
        if not code:
            return SecurityDecision.block(
                "param_whitelist:missing_code",
                risk_score=60,
                reason="execute_python 缺少 'code' 参数。",
                tool_name="execute_python",
            )

        # 1. 代码长度限制
        if len(code) > self.MAX_PYTHON_CODE_LENGTH:
            return SecurityDecision.block(
                "param_whitelist:code_too_large",
                risk_score=70,
                reason=(
                    f"Python 代码过大 ({len(code)} 字符，"
                    f"上限 {self.MAX_PYTHON_CODE_LENGTH})。请拆分任务。"
                ),
                tool_name="execute_python",
            )

        # 2. 危险模式检测
        for pattern in self.DEFAULT_BLOCKED_PYTHON_PATTERNS:
            if re.search(pattern, code):
                return SecurityDecision.block(
                    "param_whitelist:blocked_python_pattern",
                    risk_score=85,
                    reason=f"Python 代码包含危险操作: {pattern}",
                    tool_name="execute_python",
                )

        return None
