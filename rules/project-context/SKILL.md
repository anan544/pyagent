---
name: project-context
description: "super_study 项目结构、端口、启动方式。当用户提到项目路径或服务时使用。"
---

# 项目上下文

## 项目结构
本项目 `super_study` 是一个微服务架构的学习平台：

| 服务 | 框架 | 默认端口 | 路径 |
|------|------|----------|------|
| super_register | Django | 8000 | E:\thepython\super_study\super_register |
| super_apply | FastAPI | 8001 | E:\thepython\super_study\super_apply |
| super_front | Vite/Vue | 5173 | E:\thepython\super_study\super_front |

## 启动命令
- Django: `python manage.py runserver 0.0.0.0:8000`
- FastAPI: `uvicorn main:app --host 0.0.0.0 --port 8001 --reload`
- Vite: `npm run dev`

## 数据库
- Django 使用 SQLite（`db.sqlite3`）
- FastAPI 可能使用独立数据库或调用 Django API

## 服务检查方式
检查服务是否运行应使用 `netstat -ano` 或 `tasklist` 查看端口占用，而非 curl。
Windows 上可用：
```
netstat -ano | findstr ":8000"
```
