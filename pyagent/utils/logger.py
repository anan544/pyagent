"""
统一日志工具 — 支持输出到控制台或文件。
"""

import sys
from datetime import datetime


class Logger:
    """简单的日志记录器，支持 info / warning / error 三个级别。"""

    def __init__(self, name: str = "PyAgent", stream=None):
        self.name = name
        if stream:
            self.stream = stream
        else:
            # 确保 stdout 使用 UTF-8，处理 Windows GBK 终端兼容问题
            self.stream = sys.stdout
            if hasattr(self.stream, "reconfigure"):
                try:
                    self.stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

    def info(self, msg: str):
        self._write("INFO", msg)

    def warning(self, msg: str):
        self._write("WARN", msg)

    def error(self, msg: str):
        self._write("ERROR", msg)

    def __call__(self, msg: str):
        """允许直接调用 logger(msg) 作为快捷方式。"""
        self.info(msg)

    def _write(self, level: str, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{level}] {msg}\n"
        # 安全写入：处理控制台编码不支持的字符
        try:
            self.stream.write(line)
        except UnicodeEncodeError:
            self.stream.write(line.encode(self.stream.encoding or "utf-8", errors="replace").decode(
                self.stream.encoding or "utf-8", errors="replace"
            ))
        self.stream.flush()
