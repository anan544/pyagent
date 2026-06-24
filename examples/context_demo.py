"""
上下文管理 Demo — 滑动窗口 + 结构化压缩。

演示流程：
    1. 设置 Token 预算（模拟小模型，便于观察裁剪效果）
    2. 多轮对话累积大量消息
    3. 纯滑动窗口裁剪（不调用 LLM）
    4. LLM 上下文压缩（生成结构化摘要）
    5. 展示"三个区域"的 Token 分配

运行：
    python examples/context_demo.py

注意：使用 Mock LLM 演示机制，不需要 API key。
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyagent.core import Agent, AgentConfig
from pyagent.core.message import (
    SystemMessage, UserMessage, AssistantMessage, ToolMessage, ToolCall,
)
from pyagent.tools import ToolRegistry, ReadFileTool
from pyagent.memory import MemoryManager, TokenBudget, ContextCompressor, LoadResult
from pyagent.utils.logger import Logger


# ═══════════════════════════════════════════════════════════════
# 模拟 LLM — 演示上下文感知
# ═══════════════════════════════════════════════════════════════

class DemoAgentLLM:
    """模拟 Agent 使用的 LLM（回应问题）。"""
    async def generate(self, messages, tools):
        # 统计上下文
        user_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == 'user')
        summary_count = sum(1 for m in messages
                           if hasattr(m, 'content') and m.content
                           and '上下文摘要' in str(m.content))
        return AssistantMessage(
            content=f"收到 {user_count} 条用户消息 "
                    f"(含 {summary_count} 条压缩摘要)。分析完成。"
        )


class DemoCompressionLLM:
    """模拟压缩 LLM — 生成结构化摘要。"""
    async def generate(self, messages, tools):
        # 提取被压缩的消息数量
        user_msgs = [m for m in messages if hasattr(m, 'role') and m.role == 'user']
        # 提取文件名
        files_found = set()
        for m in messages:
            content = getattr(m, 'content', '') or ''
            for word in content.split():
                if word.endswith('.py') or word.endswith('.md'):
                    files_found.add(word)

        files_str = "\n".join(f"- {f}" for f in sorted(files_found)) if files_found else "- 无"

        return AssistantMessage(
            content=(
                f"【已达成共识】\n"
                f"- 本次压缩了 {len(user_msgs)} 条用户消息\n"
                f"- 对话涉及代码审查和文件分析\n\n"
                f"【待解决问题】\n"
                f"- 无（所有问题已处理）\n\n"
                f"【关键文件路径】\n"
                f"{files_str}\n\n"
                f"【工具调用结论】\n"
                f"- 已读取并分析相关文件\n\n"
                f"【关键实体】\n"
                f"- 无"
            )
        )


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

async def main():
    log = Logger(name="ContextDemo")

    db_path = os.path.join(tempfile.gettempdir(), "pyagent_context_demo.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    memory = MemoryManager(db_path)
    SESSION_ID = "context-demo"

    # ── 步骤 1：模拟多轮对话，累积大量消息 ──
    log.info("=" * 70)
    log.info("步骤 1：多轮对话累积消息")
    log.info("=" * 70)

    sid = await memory.create_session(SESSION_ID)

    # 模拟 3 轮对话，每轮产生多条消息
    demo_files = [
        "src/main.py", "src/utils.py", "tests/test_main.py",
        "config/settings.yaml", "docs/README.md",
    ]

    for round_num in range(1, 4):
        # User
        await memory.save_message(
            sid,
            UserMessage(content=f"[第{round_num}轮] 请审查 {demo_files[round_num-1]}")
        )
        # Assistant + tool_call
        tc = ToolCall(id=f"c{round_num}", function_name="read_file",
                      arguments={"path": demo_files[round_num-1]})
        await memory.save_message(
            sid,
            AssistantMessage(content=f"读取 {demo_files[round_num-1]}...",
                             tool_calls=[tc])
        )
        # Tool result
        await memory.save_message(
            sid,
            ToolMessage(
                content=f"# {demo_files[round_num-1]}\n"
                        f"def process():\n    return 'ok'\n\n"
                        f"class Service:\n    def run(self): pass\n",
                tool_call_id=f"c{round_num}",
                name="read_file",
            )
        )
        # Assistant final answer for this round
        await memory.save_message(
            sid,
            AssistantMessage(content=f"[第{round_num}轮] 审查完成: 代码结构良好，无问题。")
        )

    # 再加一些额外的 User→Assistant 来回（模拟后续对话）
    for i in range(4, 11):
        await memory.save_message(
            sid,
            UserMessage(content=f"[第{i}轮] 之前审查的文件中，最大函数是哪个？")
        )
        await memory.save_message(
            sid,
            AssistantMessage(content=f"[第{i}轮] 最大函数是 process()，只有 2 行。")
        )

    total_count = await memory.message_count(sid)
    log.info(f"累积消息总数: {total_count}")

    # ── 步骤 2：无预算 — 全部加载 ──
    log.info("")
    log.info("=" * 70)
    log.info("步骤 2：无 Token 预算 — 全部消息加载")
    log.info("=" * 70)

    all_msgs = await memory.load_messages(sid)
    log.info(f"加载消息数: {len(all_msgs)} (返回 list，向后兼容)")

    # ── 步骤 3：带预算 — 纯滑动窗口裁剪 ──
    log.info("")
    log.info("=" * 70)
    log.info("步骤 3：纯滑动窗口（模型上下文=200 tokens，裁剪早期消息）")
    log.info("=" * 70)

    # 使用极小预算来演示裁剪效果（26 条消息 ≈ 221 tokens，预算 120 tokens）
    small_budget = TokenBudget(model_max_tokens=200, safety_factor=0.8)
    total_budget = small_budget.total_budget
    precision_budget = small_budget.precision_budget
    log.info(f"总预算:     {total_budget} tokens")
    log.info(f"精确区:     {precision_budget} tokens")
    log.info(f"压缩区:     {small_budget.compression_budget} tokens")
    log.info("")

    result_trim = await memory.load_messages(sid, budget=small_budget)
    log.info(f"原始消息:   {result_trim.original_count} 条")
    log.info(f"原始Token:  {result_trim.original_tokens}")
    log.info(f"返回消息:   {len(result_trim.messages)} 条")
    log.info(f"返回Token:  {result_trim.total_tokens}")
    log.info(f"纯裁剪:     {result_trim.trimmed_count} 条早期消息被丢弃")
    log.info(f"压缩:       {'否' if not result_trim.was_compressed else '是'}")

    # 展示三个区域
    log.info("")
    log.info("Token 分配图：")
    system_est = len("你是一个代码审查助手...") // 4
    log.info(f"  [System Prompt]  {system_est:>5} tokens  ← 锚点，固定不变")
    log.info(f"  [压缩历史]      {'0':>5} tokens  ← 无 compressor，压缩区为空")
    log.info(f"  [近期消息]      {result_trim.total_tokens:>5} tokens  ← {len(result_trim.messages)} 条原始消息")
    log.info(f"  {'─' * 35}")
    log.info(f"  [总计]          {result_trim.total_tokens + system_est:>5} tokens  /  预算 {total_budget}")

    # ── 步骤 4：带压缩器 — LLM 压缩早期消息 ──
    log.info("")
    log.info("=" * 70)
    log.info("步骤 4：LLM 上下文压缩（模型上下文=100 tokens + Compressor）")
    log.info("=" * 70)

    # 极小预算强制触发压缩（precision_budget ≈ 48 tokens）
    tiny_budget = TokenBudget(model_max_tokens=100, safety_factor=0.8)
    log.info(f"总预算:     {tiny_budget.total_budget} tokens")
    log.info(f"精确区:     {tiny_budget.precision_budget} tokens")
    log.info("")

    compressor = ContextCompressor(DemoCompressionLLM())
    result_compress = await memory.load_messages(
        sid, budget=tiny_budget, compressor=compressor
    )
    log.info(f"原始消息:   {result_compress.original_count} 条")
    log.info(f"原始Token:  {result_compress.original_tokens}")
    log.info(f"返回消息:   {len(result_compress.messages)} 条")
    log.info(f"压缩了:     {result_compress.compressed_count} 条早期消息 → 1 条结构化摘要")
    log.info(f"裁剪了:     {result_compress.trimmed_count} 条")

    log.info("")
    log.info("压缩摘要内容：")
    if result_compress.summary:
        for line in result_compress.summary.split("\n")[:10]:
            log.info(f"  {line}")
        if len(result_compress.summary.split("\n")) > 10:
            log.info(f"  ... (共 {len(result_compress.summary.split(chr(10)))} 行)")

    # 展示三个区域（有压缩）
    log.info("")
    log.info("Token 分配图（三区域模型）：")
    summary_tokens = len(result_compress.summary) // 4 if result_compress.summary else 0
    recent_tokens = sum(len(m.content or '') // 4
                        for m in result_compress.messages
                        if hasattr(m, 'content') and m.content
                        and '上下文摘要' not in str(m.content))
    recent_tokens = max(1, recent_tokens)
    log.info(f"  [System Prompt]  {system_est:>5} tokens  ← 深蓝色锚点，固定不变")
    log.info(f"  [压缩历史]      {summary_tokens:>5} tokens  ← 浅橙色，结构化摘要")
    log.info(f"  [近期消息]      {recent_tokens:>5} tokens  ← 浅绿色，原始未压缩")
    log.info(f"  {'─' * 35}")
    log.info(f"  [总计]          {summary_tokens + recent_tokens + system_est:>5} tokens  /  预算 {tiny_budget.total_budget}")

    # ── 步骤 5：后续对话 — 压缩摘要被复用 ──
    log.info("")
    log.info("=" * 70)
    log.info("步骤 5：新对话 — 压缩摘要在上下文中")
    log.info("=" * 70)

    # 添加一条新消息
    await memory.save_message(
        sid,
        UserMessage(content="综合前面的审查结果，生成总体报告")
    )

    new_result = await memory.load_messages(
        sid, budget=tiny_budget, compressor=compressor
    )

    # 检查返回消息中是否包含压缩摘要
    has_summary = any(
        '上下文摘要' in (getattr(m, 'content', '') or '')
        for m in new_result.messages
    )
    log.info(f"新对话返回 {len(new_result.messages)} 条消息")
    log.info(f"包含压缩摘要: {'是' if has_summary else '否 （已在预算内，不需要压缩）'}")
    if new_result.was_compressed:
        log.info(f"再次压缩了: {new_result.compressed_count} 条")
    log.info(f"裁剪: {new_result.trimmed_count} 条")

    # ── 步骤 6：在真正的 Agent 中使用 ──
    log.info("")
    log.info("=" * 70)
    log.info("步骤 6：Agent 集成示例")
    log.info("=" * 70)

    registry = ToolRegistry()
    registry.register(ReadFileTool())

    config = AgentConfig(
        system_prompt="你是一个代码审查助手。请分析代码文件并给出反馈。",
        max_iterations=3,
        verbose=False,
        token_budget=TokenBudget.for_model("deepseek-chat"),  # ← 通过 config 注入
    )

    agent = Agent(
        config=config,
        tool_registry=registry,
        llm_provider=DemoAgentLLM(),
        memory=memory,
        context_compressor=compressor,  # ← 可选，不传则纯滑动窗口
        logger=log,
    )

    # 使用同一个 session_id → Agent 自动加载历史 + 应用滑动窗口/压缩
    result = await agent.run("综合所有审查结果，给出总体评分", session_id=sid)
    log.info(f"Agent 回复: {result}")

    await memory.close()

    # 清理
    if os.path.exists(db_path):
        os.remove(db_path)
        log.info(f"\n清理演示数据库: {db_path}")

    log.info("Demo 完成!")


if __name__ == "__main__":
    asyncio.run(main())
