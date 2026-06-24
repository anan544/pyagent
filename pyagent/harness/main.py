"""
PyAgent Harness API 入口模块 — 供 uvicorn 直接导入。

启动方式：
    uvicorn pyagent.harness.main:app --host 127.0.0.1 --port 8080
"""

from pyagent.harness.api import create_app

app = create_app()
