---
name: coding-standards
paths:
  - "**/*.py"
  - "**/*.ts"
  - "**/*.tsx"
  - "**/*.js"
  - "**/*.jsx"
  - "**/*.vue"
  - "**/*.css"
  - "**/*.scss"
keywords:
  - "代码"
  - "重构"
  - "编码"
  - "规范"
  - "代码风格"
  - "refactor"
  - "format"
  - "格式化"
  - "类型注解"
  - "PEP"
description: "通用编码规范 — 编辑代码文件或提到编码相关关键词时加载"
---

# 编码规范

## 通用规则
- 使用 UTF-8 编码
- 缩进使用 4 空格（Python）、2 空格（JS/TS）
- 行尾不留空白字符
- 文件末尾保留一个空行

## Python 规范
- 遵循 PEP 8
- 类型注解：所有函数参数和返回值必须有类型注解
- 使用 `pathlib.Path` 而非 `os.path`
- 异步代码使用 `asyncio`，避免同步阻塞

## TypeScript/Vue 规范
- 使用 `const`/`let`，禁用 `var`
- 优先使用 `interface` 而非 `type`
- 组件命名使用 PascalCase
- 事件处理函数以 `on` 开头（如 `onClick`、`onSubmit`）
