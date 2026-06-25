---
name: testing
paths:
  - "**/*.test.ts"
  - "**/*.test.tsx"
  - "**/*.spec.ts"
  - "**/test_*.py"
  - "**/*_test.py"
  - "**/tests/**"
keywords:
  - "测试"
  - "单元测试"
  - "集成测试"
  - "pytest"
  - "vitest"
  - "jest"
  - "mock"
  - "fixture"
  - "test"
  - "spec"
  - "断言"
  - "覆盖率"
  - "coverage"
description: "测试规范 — 处理测试文件或提到测试相关关键词时加载"
---

# 测试规范

## 通用原则
- 单元测试必须使用 Arrange-Act-Assert 模式
- 测试函数名必须描述测试场景（`test_<what>_<expected>`）
- 每个测试只验证一个行为
- 使用 mock/stub 隔离外部依赖

## Python 测试
- 使用 `pytest` 框架
- fixture 放在 `conftest.py` 中
- 避免在测试中直接操作数据库，使用 `unittest.mock`

## TypeScript 测试
- 使用 `vitest` 或 `jest`
- 组件测试使用 `@vue/test-utils`
- 异步测试使用 `async/await`，避免 `done` 回调
