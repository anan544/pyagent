"""
TokenBudget — Token 预算管理器。

实现文献中的"Token 分配图"模型：
    ┌────────────┬──────────────────┬─────────────────┐
    │ 系统提示    │ 压缩历史          │ 近期消息         │
    │ (锚点)     │ (精华摘要)        │ (原始未压缩)     │
    │ 固定不变    │ 最少 Token        │ 精确上下文       │
    └────────────┴──────────────────┴─────────────────┘

使用方式：
    budget = TokenBudget.for_model("deepseek-chat")
    budget.system_prompt_tokens = estimate_tokens(system_prompt)
    print(budget.precision_budget)  # 近期消息的 Token 上限
"""

from dataclasses import dataclass, field

# 常见模型的上下文窗口大小
MODEL_CONTEXT_LIMITS = {
    "deepseek-chat": 128_000,
    "deepseek-reasoner": 64_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_384,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-8": 200_000,
    "claude-haiku-4-5": 200_000,
}


@dataclass
class TokenBudget:
    """
    Token 预算分配器。

    三个区域：
        system_prompt_tokens — 系统提示，锚点，不参与压缩
        precision_budget     — 近期消息，原始保留
        compression_budget   — 早期消息，可压缩为摘要
    """

    # 模型原始上下文窗口大小
    model_max_tokens: int = 128_000

    # 安全系数（0.7 = 只使用 70% 的上下文窗口，留 30% 给 LLM 回复）
    safety_factor: float = 0.7

    # 系统提示的 Token 消耗（由外部估算后填入）
    system_prompt_tokens: int = 0

    # 精确区占比（在可用预算中的比例，剩余给压缩区）
    precision_ratio: float = 0.6

    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> "TokenBudget":
        """
        根据模型名称创建 TokenBudget。

        Args:
            model_name: 模型名称，如 'deepseek-chat', 'gpt-4o'。
            **kwargs: 覆盖默认参数（safety_factor, precision_ratio 等）。
        """
        limit = MODEL_CONTEXT_LIMITS.get(model_name, 128_000)
        return cls(model_max_tokens=limit, **kwargs)

    # ── 计算属性 ─────────────────────────────────────────────────

    @property
    def total_budget(self) -> int:
        """经过安全系数打折后的总 Token 预算。"""
        return int(self.model_max_tokens * self.safety_factor)

    @property
    def available_budget(self) -> int:
        """扣除系统提示后的可用 Token 预算。"""
        return max(0, self.total_budget - self.system_prompt_tokens)

    @property
    def precision_budget(self) -> int:
        """
        精确区 Token 预算 — 近期消息的最大 Token 数。
        这些消息保持原始未压缩状态。
        """
        return int(self.available_budget * self.precision_ratio)

    @property
    def compression_budget(self) -> int:
        """
        压缩区 Token 预算 — 早期消息压缩后的 Token 上限。
        压缩摘要不能超过这个值。
        """
        return self.available_budget - self.precision_budget

    # ── 分区分割 ─────────────────────────────────────────────────

    def split_messages(
        self, messages: list, token_estimator=None
    ) -> tuple[list, list]:
        """
        按 Token 预算将消息分为"精确区"和"压缩区"。

        从最新消息往前累加 token_count，累加值不超 precision_budget 的
        消息放入精确区，其余划入压缩区。

        Args:
            messages: Message 对象列表（按时间升序，最旧在前）。
            token_estimator: 可选的外部 Token 估算函数，签名为 (msg) -> int。
                             不传则使用内置的字符/4 估算。

        Returns:
            (compression_zone, precision_zone) — 两个都是列表，
            compression_zone 是早期消息（需要压缩），
            precision_zone 是近期消息（保持原样）。
            两个列表都按时间升序。
        """
        if not messages:
            return [], []

        # 内置 Token 估算器
        if token_estimator is None:
            import json
            def _default_estimator(msg) -> int:
                text = ""
                if hasattr(msg, 'content') and msg.content:
                    text += str(msg.content)
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        text += getattr(tc, 'function_name', '')
                        text += json.dumps(
                            getattr(tc, 'arguments', {}), ensure_ascii=False
                        )
                if hasattr(msg, 'name'):
                    text += str(msg.name)
                return max(1, len(text) // 4)
            token_estimator = _default_estimator

        # 从最新到最旧遍历，累加 Token
        # cutoff_idx = 第一个属于精确区的消息索引
        # 默认：所有消息都在精确区（cutoff_idx = 0 → 压缩区为空）
        accumulated = 0
        cutoff_idx = 0

        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            tokens = token_estimator(msg)
            accumulated += tokens
            if accumulated > self.precision_budget:
                # 第 i 条消息是第一个超出预算的
                # 精确区从 i+1 开始
                cutoff_idx = i + 1
                break

        # 分割
        compression_zone = messages[:cutoff_idx]
        precision_zone = messages[cutoff_idx:]

        return compression_zone, precision_zone

    # ── 统计信息 ─────────────────────────────────────────────────

    def info(self) -> dict:
        """返回预算配置的可读摘要。"""
        return {
            "model_max_tokens": self.model_max_tokens,
            "safety_factor": self.safety_factor,
            "total_budget": self.total_budget,
            "system_prompt_tokens": self.system_prompt_tokens,
            "available_budget": self.available_budget,
            "precision_budget": self.precision_budget,
            "compression_budget": self.compression_budget,
        }

    def __repr__(self) -> str:
        return (
            f"TokenBudget(model={self.model_max_tokens}, "
            f"safety={self.safety_factor}, "
            f"total={self.total_budget}, "
            f"precision={self.precision_budget}, "
            f"compression={self.compression_budget})"
        )
