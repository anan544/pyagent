"""
交互式调试 REPL 工具 — 支持分步执行、变量检查、异常分析。

特性：
    - 持久化命名空间：同一会话内的多次调用共享变量
    - 完整异常捕获：返回 traceback + 局部变量
    - 表达式求值：直接计算并返回结果
    - 代码检查：不执行，仅做静态分析
"""

import traceback
import sys
import io
import ast
from .base import Tool


class DebugReplTool(Tool):
    """交互式 Python 调试 REPL — 分步执行代码并检查变量。"""

    name = "debug_repl"
    risk_level = "medium"
    description = (
        "交互式 Python 调试 REPL（Read-Eval-Print Loop）环境。\n"
        "\n"
        "当你需要实时检查变量、测试小段代码或调试逻辑时使用此工具。"
        "你可以设置变量状态并逐步执行代码。\n"
        "支持的操作：\n"
        "  - exec: 执行 Python 代码，捕获 stdout（变量在同一会话内持久化）\n"
        "  - inspect: 检查命名空间中的变量（列出所有或查看指定变量）\n"
        "  - eval: 求值单个表达式并返回结果\n"
        "  - trace: 执行代码，异常时返回完整 traceback + 局部变量\n"
        "  - reset: 清空当前会话的命名空间\n"
        "适合：调试复杂算法、逐步验证假设、分析异常原因，无需运行完整应用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "操作类型: exec（执行代码）, inspect（检查变量）, eval（求值表达式）, trace（异常追踪）, reset（重置命名空间）",
                "enum": ["exec", "inspect", "eval", "trace", "reset"],
            },
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码块或表达式（exec / eval / trace 操作需要）",
            },
            "variable": {
                "type": "string",
                "description": "要检查的特定变量名（inspect 操作可选，不填则列出所有变量）",
            },
            "session_id": {
                "type": "string",
                "description": "调试会话标识符，用于隔离不同上下文的命名空间",
            },
        },
        "required": ["action"],
    }

    # 类级别命名空间存储：{ session_key: { var_name: value } }
    _namespaces: dict[str, dict] = {}

    def _get_namespace(self, session_id: str = "") -> dict:
        key = session_id or "__default__"
        if key not in self._namespaces:
            self._namespaces[key] = {
                "__builtins__": __builtins__,
            }
        return self._namespaces[key]

    async def execute(
        self,
        action: str,
        code: str = "",
        variable: str = "",
        session_id: str = "",
        **kwargs,
    ) -> str:
        ns = self._get_namespace(session_id)

        if action == "reset":
            key = session_id or "__default__"
            self._namespaces.pop(key, None)
            return "命名空间已重置。"

        if action == "inspect":
            if variable:
                if variable in ns:
                    val = ns[variable]
                    return (
                        f"变量: {variable}\n"
                        f"类型: {type(val).__name__}\n"
                        f"值: {self._format_value(val)}"
                    )
                return f"变量 '{variable}' 不存在。可用变量: {', '.join(self._list_vars(ns))}"
            else:
                user_vars = self._list_vars(ns)
                if not user_vars:
                    return "命名空间为空（无用户定义变量）。"
                lines = ["当前命名空间变量:"]
                for v in user_vars:
                    val = ns[v]
                    lines.append(f"  {v}: {type(val).__name__} = {self._format_value(val, 200)}")
                return "\n".join(lines)

        if action == "eval":
            if not code:
                return "[错误] eval 操作需要 'code' 参数"
            try:
                result = eval(code, ns)
                return (
                    f"表达式: {code}\n"
                    f"结果类型: {type(result).__name__}\n"
                    f"结果: {self._format_value(result)}"
                )
            except Exception as e:
                return f"[异常] {type(e).__name__}: {e}"

        if action == "exec":
            if not code:
                return "[错误] exec 操作需要 'code' 参数"
            stdout = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout
            try:
                exec(code, ns)
                output = stdout.getvalue()
                result_parts = []
                if output.strip():
                    result_parts.append(f"[stdout]\n{output.rstrip()}")
                # 自动展示新变量
                new_vars = [v for v in ns if not v.startswith("__") and v != "__builtins__"]
                if new_vars:
                    var_lines = ["[当前变量]"]
                    for v in sorted(new_vars):
                        var_lines.append(f"  {v}: {type(ns[v]).__name__} = {self._format_value(ns[v], 150)}")
                    result_parts.append("\n".join(var_lines))
                return "\n\n".join(result_parts) if result_parts else "(执行完成，无输出，无用户定义变量)"
            except Exception as e:
                return f"[异常] {type(e).__name__}: {e}"
            finally:
                sys.stdout = old_stdout

        if action == "trace":
            if not code:
                return "[错误] trace 操作需要 'code' 参数"
            stdout = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout
            try:
                exec(code, ns)
                output = stdout.getvalue()
                return f"[执行成功]\n{output.rstrip()}" if output.strip() else "(执行成功，无输出)"
            except Exception:
                exc_type, exc_value, exc_tb = sys.exc_info()
                tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
                # 捕获异常时的局部变量
                local_vars = {}
                tb_frame = exc_tb
                while tb_frame is not None:
                    local_vars.update(tb_frame.tb_frame.f_locals)
                    tb_frame = tb_frame.tb_next

                result = ["[异常 + 完整 Traceback]"]
                result.append("".join(tb_lines))
                if local_vars:
                    result.append("\n[异常时局部变量]")
                    for k, v in sorted(local_vars.items()):
                        if not k.startswith("__"):
                            result.append(f"  {k}: {type(v).__name__} = {self._format_value(v, 150)}")
                return "\n".join(result)
            finally:
                sys.stdout = old_stdout

        return f"[错误] 未知操作: {action}"

    @staticmethod
    def _list_vars(ns: dict) -> list[str]:
        return sorted(
            k for k in ns
            if not k.startswith("__") and k != "__builtins__"
        )

    @staticmethod
    def _format_value(val, max_len: int = 500) -> str:
        s = repr(val)
        if len(s) > max_len:
            # 根据类型截断
            if isinstance(val, str):
                s = f'"{s[1:max_len-2]}..."'
            elif isinstance(val, (list, tuple)):
                s = s[:max_len-3] + "...]"
            elif isinstance(val, dict):
                s = s[:max_len-3] + "...}"
            else:
                s = s[:max_len-3] + "..."
        return s
