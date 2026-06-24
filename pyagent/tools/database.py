"""
数据库交互工具 — 执行 SQL 查询、查看表结构、管理数据库。

支持 MySQL/MariaDB（通过 pymysql）。
连接信息通过参数传入，也可从环境变量读取。
"""

import os
import asyncio
from .base import Tool


class DatabaseTool(Tool):
    """数据库交互 — 执行 SQL、查看表结构、列出数据库。"""

    name = "database_query"
    risk_level = "high"
    description = (
        "对数据库执行查询或修改操作。\n"
        "\n"
        "支持 MySQL/MariaDB（通过 pymysql）。可执行 SELECT 查询、INSERT/UPDATE/DELETE 修改、\n"
        "CREATE/ALTER/DROP 表结构变更等完整 SQL 操作。\n"
        "连接信息通过参数指定，或从环境变量 DB_HOST/DB_USER/DB_PASS/DB_NAME 读取。\n"
        "⚠️ 写操作直接生效，无法撤销。执行 DROP/TRUNCATE 前请确认。\n"
        "查询结果最多返回 200 行，超出会被截断。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要执行的 SQL 语句。支持 SELECT、INSERT、UPDATE、DELETE、CREATE、ALTER、DROP、SHOW、DESCRIBE 等。",
            },
            "host": {
                "type": "string",
                "description": "数据库主机地址，默认 localhost",
                "default": "localhost",
            },
            "port": {
                "type": "integer",
                "description": "数据库端口，默认 3306",
                "default": 3306,
            },
            "user": {
                "type": "string",
                "description": "数据库用户名，默认 root",
                "default": "root",
            },
            "password": {
                "type": "string",
                "description": "数据库密码",
                "default": "",
            },
            "database": {
                "type": "string",
                "description": "要连接的数据库名称",
                "default": "",
            },
            "explanation": {
                "type": "string",
                "description": "一句话说明此查询的目的。",
            },
        },
        "required": ["query"],
    }

    def __init__(self):
        self._conn_pool = None

    async def execute(
        self,
        query: str,
        host: str = "localhost",
        port: int = 3306,
        user: str = "",
        password: str = "",
        database: str = "",
        **kwargs,
    ) -> str:
        # 参数优先级：参数 > 环境变量 > 默认值
        host = host or os.getenv("DB_HOST", "localhost")
        port = port or int(os.getenv("DB_PORT", "3306"))
        user = user or os.getenv("DB_USER", "root")
        password = password or os.getenv("DB_PASS", "123456")
        database = database or os.getenv("DB_NAME", "")

        query_stripped = query.strip().rstrip(";")

        try:
            return await asyncio.to_thread(
                self._execute_sync, query_stripped, host, port, user, password, database
            )
        except Exception as e:
            return f"[数据库错误] {e}"

    def _execute_sync(self, query: str, host: str, port: int,
                      user: str, password: str, database: str) -> str:
        """在线程池中同步执行 SQL（避免阻塞事件循环）。"""
        import pymysql

        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database if database else None,
            charset="utf8mb4",
            connect_timeout=10,
            read_timeout=30,
            autocommit=True,  # ★ 写操作直接生效
        )

        try:
            with conn.cursor() as cursor:
                cursor.execute(query)

                if query.upper().startswith(("SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN")):
                    rows = cursor.fetchall()
                    if not rows:
                        return "(查询成功，但无返回结果)"

                    headers = [desc[0] for desc in cursor.description]
                    # 限制返回行数
                    max_rows = 200
                    truncated = len(rows) > max_rows
                    display_rows = rows[:max_rows]

                    # 格式化输出
                    col_widths = [len(h) for h in headers]
                    for row in display_rows:
                        for i, val in enumerate(row):
                            col_widths[i] = max(col_widths[i], len(str(val)))

                    lines = []
                    # 表头
                    header_line = " | ".join(
                        h.ljust(col_widths[i]) for i, h in enumerate(headers)
                    )
                    lines.append(header_line)
                    lines.append("-" * len(header_line))
                    # 数据行
                    for row in display_rows:
                        line = " | ".join(
                            str(v).ljust(col_widths[i]) for i, v in enumerate(row)
                        )
                        lines.append(line)

                    result = "\n".join(lines)
                    if truncated:
                        result += f"\n\n... 结果已截断（显示前 {max_rows} 行，共 {len(rows)} 行）"
                    result += f"\n({len(rows)} 行返回)"
                    return result

                elif query.upper().startswith("USE"):
                    return f"已切换到数据库: {query.split()[1]}"
                else:
                    return f"执行成功 ({cursor.rowcount} 行受影响)"
        finally:
            conn.close()
