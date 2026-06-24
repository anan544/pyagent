"""
现实世界交互工具 — 时间、日期、时区信息。

提供当前时间戳、时区转换、日期计算等能力。
"""

import time as _time
from datetime import datetime, timezone, timedelta
from .base import Tool


class TimeLocationTool(Tool):
    """获取当前时间和地理位置相关信息。"""

    name = "time_location"
    risk_level = "low"
    description = (
        "获取 Agent 当前的时间和空间上下文。\n"
        "\n"
        "此工具提供当前时间、时区信息和用户地理位置。"
        "使用此信息判断任务是否具有时间敏感性、正确格式化日期或提供位置相关的协助。\n"
        "支持的操作：now（当前时间）、timestamp（Unix 时间戳）、"
        "timezone（时区信息）、datetime_info（完整时间信息）、time_diff（时差计算）。\n"
        "注意：时间以 ISO 8601 格式返回，包含时区偏移。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "操作类型: now, timestamp, timezone, datetime_info, time_diff",
                "enum": ["now", "timestamp", "timezone", "datetime_info", "time_diff"],
            },
            "timezone_offset": {
                "type": "string",
                "description": "目标时区偏移(如 '+8' 表示 UTC+8)，用于 time_diff 操作",
            },
        },
        "required": ["action"],
    }

    async def execute(self, action: str, timezone_offset: str = "", **kwargs) -> str:
        now_utc = datetime.now(timezone.utc)
        now_local = datetime.now()
        ts = int(_time.time())

        if action == "now":
            return (
                f"当前本地时间: {now_local.strftime('%Y-%m-%d %H:%M:%S.%f')}\n"
                f"星期: {self._weekday_cn(now_local.weekday())}\n"
                f"UTC 时间: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        elif action == "timestamp":
            return (
                f"Unix 时间戳: {ts}\n"
                f"可读格式: {datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"ISO 8601: {now_utc.isoformat()}"
            )

        elif action == "timezone":
            local_offset = now_local.utcoffset()
            hours = local_offset.total_seconds() / 3600 if local_offset else 0
            return (
                f"本地时区: UTC{'+' if hours >= 0 else ''}{hours:.0f}\n"
                f"时区名称: {_time.tzname[0]}\n"
                f"夏令时: {'是' if _time.localtime().tm_isdst else '否'}\n"
                f"UTC 偏移秒数: {local_offset.total_seconds() if local_offset else 0}"
            )

        elif action == "datetime_info":
            return (
                f"=== 完整时间信息 ===\n"
                f"本地时间: {now_local.strftime('%Y-%m-%d %H:%M:%S.%f')}\n"
                f"UTC 时间: {now_utc.strftime('%Y-%m-%d %H:%M:%S.%f')}\n"
                f"Unix 时间戳: {ts}\n"
                f"ISO 8601: {now_utc.isoformat()}\n"
                f"星期: {self._weekday_cn(now_local.weekday())}\n"
                f"年份: {now_local.year} | 月份: {now_local.month} | 日: {now_local.day}\n"
                f"时: {now_local.hour} | 分: {now_local.minute} | 秒: {now_local.second}\n"
                f"时区: {_time.tzname[0]}"
            )

        elif action == "time_diff":
            target = now_utc
            if timezone_offset:
                try:
                    offset_hours = float(timezone_offset)
                    target = now_utc + timedelta(hours=offset_hours)
                except ValueError:
                    return f"[错误] 无效的时区偏移: {timezone_offset}"
            diff = target - now_utc
            return (
                f"UTC 时间: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"目标时区(UTC{timezone_offset or '+0'}): {target.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"时差: {diff.total_seconds() / 3600:.1f} 小时"
            )

        return f"[错误] 未知操作: {action}"

    @staticmethod
    def _weekday_cn(day: int) -> str:
        return ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][day]
