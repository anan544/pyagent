"""
文件操作工具 — read_file 和 write_file。

提供 Agent 读写文件系统的能力。
"""

import os
from .base import Tool


class ReadFileTool(Tool):
    """读取指定路径的文件内容，返回带行号的全文。"""

    name = "read_file"
    description = (
        "读取指定文件的内容。输出包含行号（1-indexed），方便定位和引用。\n"
        "\n"
        "关键规则：\n"
        "1. **部分视图**：默认返回完整文件内容，大文件会自动截断（500KB限制）。\n"
        "2. **主动重读**：如果当前视图缺少关键的 import、依赖或功能，请再次调用此工具。\n"
        "3. **完整文件**：对于大型文件（>200行），请考虑分段读取以减少 Token 消耗。\n"
        "4. **明确意图**：使用 explanation 参数说明为什么需要读取此文件。"
    )
    risk_level = "low"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要读取的文件路径（绝对路径或相对于工作区的相对路径）。",
            },
            "explanation": {
                "type": "string",
                "description": "一句话说明为什么读取此文件以及它对目标的贡献。",
            },
        },
        "required": ["path"],
    }

    async def execute(self, path: str, cwd: str | None = None, **kwargs) -> str:
        """
        读取文件内容。

        Args:
            path: 文件路径（绝对路径，或相对于 cwd 的相对路径）
            cwd: 工作目录。相对路径将基于此目录解析。

        Returns:
            文件内容，或错误信息
        """
        # ★ 解析相对路径：优先基于 cwd
        if cwd and not os.path.isabs(path):
            path = os.path.join(cwd, path)

        # 安全检查：确保路径存在
        if not os.path.exists(path):
            return f"[错误] 文件不存在: {path}"

        if not os.path.isfile(path):
            return f"[错误] 路径不是文件: {path}"

        # 大小限制：拒绝读取过大的文件（防止 token 爆炸）
        file_size = os.path.getsize(path)
        max_size = 500 * 1024  # 500KB
        if file_size > max_size:
            return (
                f"[错误] 文件过大 ({file_size} bytes)，"
                f"超过限制 ({max_size} bytes)。"
                f"请指定行号范围分段读取。"
            )

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            # 添加行号，方便 LLM 理解
            lines = content.split("\n")
            numbered = "\n".join(f"{i+1:>4}|{line}" for i, line in enumerate(lines))
            return numbered
        except Exception as e:
            return f"[错误] 读取文件失败: {e}"


class WriteFileTool(Tool):
    """创建新文件或覆盖已有文件。"""

    name = "write_file"
    risk_level = "high"
    description = (
        "创建新文件或覆盖已有文件，写入完整内容。\n"
        "\n"
        "使用此工具编写代码、配置文件或文档。如果文件已存在将被覆盖。\n"
        "请始终提供文件的完整内容，而非增量补丁。\n"
        "⚠️ 如果目标是修改已有文件的部分内容，请先用 read_file 读取，\n"
        "修改后再用本工具完整写入。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要创建或覆盖的文件路径（绝对路径或相对于工作区的相对路径）。",
            },
            "content": {
                "type": "string",
                "description": "要写入文件的完整内容。",
            },
            "explanation": {
                "type": "string",
                "description": "一句话说明创建或覆盖此文件的目的。",
            },
        },
        "required": ["path", "content"],
    }

    async def execute(self, path: str, content: str, cwd: str | None = None, **kwargs) -> str:
        """
        写入文件内容。

        Args:
            path: 文件路径（绝对路径，或相对于 cwd 的相对路径）
            content: 要写入的内容
            cwd: 工作目录。相对路径将基于此目录解析。

        Returns:
            操作结果描述
        """
        # ★ 解析相对路径：优先基于 cwd
        if cwd and not os.path.isabs(path):
            path = os.path.join(cwd, path)

        try:
            # 确保父目录存在
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            file_size = os.path.getsize(path)
            line_count = content.count("\n") + 1
            return (
                f"文件写入成功: {path}\n"
                f"  大小: {file_size} bytes\n"
                f"  行数: {line_count}"
            )
        except Exception as e:
            return f"[错误] 写入文件失败: {e}"
