"""
Tool 抽象基类 — 所有工具必须继承此类。

定义工具的统一接口：
    - name:        工具名称（LLM 通过此名称调用）
    - description: 工具描述（帮助 LLM 决定何时使用）
    - parameters:  参数 JSON Schema（定义工具接受的参数）
    - risk_level:  工具风险等级（v0.10.0 新增，供安全治理层使用）
    - execute():   执行工具逻辑的抽象方法
"""

from abc import ABC, abstractmethod
from typing import Any, Literal


class Tool(ABC):
    """
    工具抽象基类。

    子类需要定义：
        name: str        — 工具名称（唯一标识）
        description: str — 工具的功能描述
        parameters: dict — JSON Schema 格式的参数定义
        risk_level: str  — 风险等级: "low" / "medium" / "high"（默认 "medium"）

    子类需要实现：
        async execute(**kwargs) -> str — 执行工具逻辑，返回结果字符串
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}
    risk_level: Literal["low", "medium", "high"] = "medium"

    def get_schema(self) -> dict:
        """
        返回 OpenAI function calling 格式的工具定义。
        用于发送给 LLM 的 tools 参数。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """
        执行工具逻辑。

        Args:
            **kwargs: 根据 parameters 定义的参数，由框架自动解包传入

        Returns:
            执行结果的字符串表示

        Raises:
            各种可能的异常 — 由 Agent 循环捕获并包装为 ToolMessage
        """
        ...
