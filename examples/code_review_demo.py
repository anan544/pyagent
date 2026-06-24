"""
第三步：代码审查 Demo — 用真实 LLM + 真实工具完成代码审查。

运行：
    # 使用 OpenAI
    set OPENAI_API_KEY=sk-xxx
    python examples/code_review_demo.py path/to/file.py

    # 使用 DeepSeek
    set OPENAI_API_KEY=sk-xxx
    python examples/code_review_demo.py path/to/file.py --provider deepseek

如果没有 API key，会自动退化为 mock 模式演示流程。
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyagent.core import Agent, AgentConfig
from pyagent.tools import (
    ToolRegistry,
    ReadFileTool,
    WriteFileTool,
    CodeExecutorTool,
    SearchTool,
)
from pyagent.llm import LLMProvider
from pyagent.utils.logger import Logger


CODE_REVIEW_SYSTEM_PROMPT = """\
你是一个专业的代码审查助手。你的任务是审查用户提供的代码文件。

你可以使用以下工具：
- read_file: 读取文件内容
- search_content: 在代码库中搜索关键字或正则表达式（查找函数定义、引用等）
- write_file: 写入审查报告
- execute_python: 执行代码来验证逻辑和运行测试

审查流程：
1. 先使用 read_file 读取目标文件
2. 如有需要，使用 search_content 在项目中搜索相关引用
3. 分析代码的质量、风格、潜在 bug、安全问题
4. 如果代码包含可执行逻辑，使用 execute_python 运行测试
5. 使用 write_file 将审查报告写入 {output_path}

请用中文输出审查报告，包括：
- 代码整体评价
- 发现的问题（按严重程度排列）
- 改进建议
- 测试执行结果（如果有）
"""


async def run_with_mock(target_file: str):
    """
    无 API key 时的 mock 演示 — 展示 Agent 的推理流程。
    """
    logger = Logger(name="CodeReviewMock")

    # 使用第一步中的 mock 模式
    from examples.step1_mock_demo import MockLLMProvider, MockToolRegistry

    config = AgentConfig(
        system_prompt=CODE_REVIEW_SYSTEM_PROMPT.format(output_path="review_report.md"),
        max_iterations=10,
        verbose=True,
    )

    agent = Agent(
        config=config,
        tool_registry=MockToolRegistry(),
        llm_provider=MockLLMProvider(),
        logger=logger,
    )

    logger.info(f"[MOCK MODE] 模拟审查文件: {target_file}")
    logger.info("=" * 60)

    result = await agent.run(
        f"请审查这个文件: {target_file}\n"
        f"先读取文件，分析代码质量，然后写入审查报告。"
    )

    print("\n" + "=" * 60)
    print("[审查结果]")
    print(result)
    print("=" * 60)
    print("\n(这是 mock 模式 — 设置 OPENAI_API_KEY 后可体验真实审查)")


async def run_with_real_llm(
    target_file: str,
    provider_name: str = "openai",
):
    """
    使用真实 LLM 进行代码审查。
    """
    from pyagent.llm.openai_compat import OpenAICompatProvider

    logger = Logger(name="CodeReview")
    report_path = os.path.join(
        os.path.dirname(target_file) or ".",
        f"review_report_{os.path.basename(target_file)}.md",
    )

    # 选择 provider
    if provider_name == "deepseek":
        provider = OpenAICompatProvider(
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )
    else:
        provider = OpenAICompatProvider(
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )

    # 组装工具
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(CodeExecutorTool())
    registry.register(SearchTool())

    # 配置 Agent
    config = AgentConfig(
        system_prompt=CODE_REVIEW_SYSTEM_PROMPT.format(output_path=report_path),
        max_iterations=15,
        verbose=True,
    )

    agent = Agent(
        config=config,
        tool_registry=registry,
        llm_provider=provider,
        logger=logger,
    )

    logger.info(f"审查目标: {target_file}")
    logger.info(f"Provider: {provider_name}")
    logger.info(f"报告输出: {report_path}")
    logger.info("=" * 60)

    result = await agent.run(
        f"请审查文件 {target_file}，先用 read_file 读取内容，"
        f"分析代码质量，如有可执行逻辑则用 execute_python 运行测试，"
        f"最后用 write_file 将审查报告写入 {report_path}。"
    )

    print("\n" + "=" * 60)
    print("[最终回复]")
    print(result)
    print("=" * 60)

    # 检查报告是否已写入
    if os.path.exists(report_path):
        print(f"\n审查报告已保存到: {report_path}")
        with open(report_path, "r", encoding="utf-8") as f:
            preview = f.read()[:500]
        print(f"报告预览:\n{preview}...")
    else:
        print("\n(报告未生成 — Agent 可能没有调用 write_file)")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="PyAgent 代码审查 Demo")
    parser.add_argument(
        "file",
        nargs="?",
        help="要审查的文件路径",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "deepseek"],
        default="deepseek",
        help="LLM provider（默认 deepseek）",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="强制使用 mock 模式（不需要 API key）",
    )
    args = parser.parse_args()

    # 确定审查目标
    if args.file:
        target = args.file
    else:
        # 默认审查项目自身的代码
        target = os.path.join(
            os.path.dirname(__file__), "..", "pyagent", "core", "agent.py"
        )
        target = os.path.abspath(target)
        print(f"未指定文件，使用默认目标: {target}")

    if not os.path.exists(target):
        print(f"[错误] 文件不存在: {target}")
        sys.exit(1)

    # 判断使用真实 LLM 还是 mock
    has_api_key = (
        os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    ) and not args.mock

    if has_api_key:
        await run_with_real_llm(target, args.provider)
    else:
        print("(未检测到 OPENAI_API_KEY，使用 mock 模式)")
        await run_with_mock(target)


if __name__ == "__main__":
    asyncio.run(main())
