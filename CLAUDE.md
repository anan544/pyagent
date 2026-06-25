# PyAgent — LLM Agent Framework

独立 Python LLM Agent 框架。核心：ReAct Loop（文本双模） + PEVR Loop（规划-执行-验证-修复）。

## 架构速览

```
用户输入
  → ContextAssembler（系统提示词 + 规则注入 + 动态上下文）
  → Agent._react_loop()
      ├─ 文本 ReAct 模式（system prompt 含 <action> → tools=[]，解析 <thought>/<action>/<final_answer>）
      └─ 函数调用模式（传统 OpenAI function calling，向后兼容）
  → _execute_tool() → SecurityGovernance 门控 → 工具执行 → 输出压缩
  → 循环直到 final_answer 或 max_iterations
```

## 目录地图

```
pyagent/
├── core/agent.py            # ★ ReAct 主循环（985行，最核心）
├── core/config.py           # AgentConfig 数据类
├── core/message.py          # System/User/Assistant/ToolMessage 类型
├── llm/openai_compat.py     # OpenAI 兼容 LLM Provider
├── memory/
│   ├── database.py          # SQLite 持久化（aiosqlite）
│   ├── manager.py           # MemoryManager — 消息存取 + 滑动窗口
│   ├── budget.py            # TokenBudget — 五槽位分配
│   └── compressor.py        # ★ ContextCompressor v2 — 分类型压缩（Code/JSON/Log/Text）
├── tools/
│   ├── file_ops.py          # ReadFileTool + WriteFileTool
│   ├── code_executor.py     # Python 执行（subprocess）
│   ├── command_executor.py  # Shell 命令执行
│   ├── search.py            # 内容搜索（rg→findstr→Python 三层降级）
│   ├── orchestration.py     # ★ AgentTool (spawn_subagent) — 多 Agent 协作
│   └── registry.py          # ToolRegistry
├── harness/
│   ├── api/                 # FastAPI HTTP 服务
│   │   ├── server.py        # 应用工厂 + 生命周期
│   │   ├── dependencies.py  # AgentManager 单例 + 初始化
│   │   └── routes/run.py    # ★ POST /run + SSE 流式
│   ├── config/
│   │   ├── schema.py        # Pydantic 配置校验
│   │   └── loader.py        # YAML → 配置对象
│   └── context/
│       ├── rules_loader.py  # ★ 外部规则条件加载（YAML Frontmatter + paths + keywords）
│       ├── security_governance.py  # 安全治理
│       └── runner.py        # PEVRRunner 状态机
├── rules/                   # ★ 外部业务规则（.md 文件，热更新）
│   ├── project-context.md
│   ├── coding-standards.md
│   └── testing.md
├── config.dev.yaml          # 开发配置（含完整 ReAct system_prompt）
└── config.prod.yaml         # 生产配置
```

## 关键概念

### ReAct 文本模式（v1.2）
- 自检激活：system prompt 含 `<action>` → 自动启用
- LLM 纯文本输出 `<thought>` + `<action>`（JSON）+ `<final_answer>`
- Agent 正则解析标签 → 执行工具 → 容错 JSON 修复（Windows 路径反斜杠）
- 不走 OpenAI function calling（tools=[]）

### 外部规则分离（v1.2）
- `rules_dir` 下 `.md` 文件，YAML Frontmatter 声明 `paths` + `keywords`
- 三级加载：全局规则（无条件）→ 路径匹配（glob）→ 关键词兜底
- 每次请求重新读取磁盘（热更新，免重启）

### 工具输出压缩（v2.4）
- `_execute_tool()` 执行完后立即过 `compress_sync()`
- 阈值 2000 字符，按类型（Code/JSON/Log）自动选压缩器
- 纯规则，不调 LLM，不额外消耗 Token

### 多 Agent 协作
- `spawn_subagent` 是普通 Tool，LLM 自己判断什么时候调用
- 子 Agent 独立实例，复用 LLM Provider + ToolRegistry，任务结束销毁

### PEVR 循环
- 四阶段状态机：PLANNING → EXECUTING → VERIFYING → REPAIRING
- 五槽位 Token 预算分配（System/Plan/History/WorkingMemory/ObservabilityHints）

## 配置关键项

```yaml
llm: {model, base_url, api_key(环境变量)}
agent: {system_prompt, max_iterations, tools, rules_dir, context_files}
memory: {db_path, token_budget: {model_max_tokens, safety_factor}}
security: {enabled, circuit_breaker, combo_detection}
```

## 运行

```bash
# 服务端
PYAGENT_ENV=dev uvicorn pyagent.harness.main:app --host 0.0.0.0 --port 8080

# VS Code 扩展
cd pyagent-vscode && npm run compile
# F5 启动调试，或 pack 成 .vsix
```

## 代码约定

- **全 async**：Agent.run()、LLM.generate()、工具执行都是异步
- **消息格式**：OpenAI messages（System/User/Assistant/ToolMessage）
- **工具容错**：异常 → ToolMessage(error)，不中断 Agent 循环
- **依赖隔离**：core/ 和 tools/ 不依赖 Web 框架
- **零侵入**：governance=None 跳过安全检查，向后兼容
- **vsix 打包**：pyagent-vscode/ 是独立的 Node 子项目

## 当前状态

| 已完成 | 待做 |
|--------|------|
| ReAct 文本模式 | 并行工具调用 |
| 外部规则条件加载 | Anthropic provider |
| 工具输出压缩 | Headroom ML 模型集成 |
| 安全治理（三态熔断 + 参数白名单） | 输出后置过滤 Layer 2 |
| SSE 流式 + VS Code 扩展 | CCR 按需拉回（LLM 可调） |
| MCP 协议集成 | |
| PEVR 状态机 | |
| 多 Agent 协作（spawn_subagent） | |
