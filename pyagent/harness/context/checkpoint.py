"""
PEVR 状态持久化 — 最小可恢复单元设计。

设计原则：
    - 不序列化整个 WorkingMemory 或 LLM 对话历史
    - History 和 Artifacts 通过 Memory 后端按需懒加载
    - 恢复时先重建状态机到断点，再触发 ContextAssembler 重新组装 Prompt
    - 永远信任 Assembler 的实时组装能力，而非缓存的历史消息
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
import logging

logger = logging.getLogger("pyagent.harness.checkpoint")


# ── PEVRCheckpoint ──────────────────────────────────────

class PEVRCheckpoint(BaseModel):
    """
    PEVR 状态检查点 — 最小可恢复单元。

    仅保存恢复所必需的最小信息：
        - 当前状态（PEVRState 值）
        - Plan 快照引用 ID
        - 当前步骤索引
        - 修补次数
        - 必要的环境变量

    明确不保存：
        - WorkingMemory 全部内容（通过 Memory 后端懒加载）
        - LLM 对话历史（由 ContextAssembler 实时组装）
        - Artifacts 全部内容（按需从 Memory 加载）
    """

    state: str = Field(
        default="planning",
        description="当前 PEVRState 值",
    )
    plan_ref_id: str = Field(
        default="",
        description="Plan 快照在 Memory 中的引用 ID（session_id）。",
    )
    current_step_index: int = Field(
        default=0,
        description="当前执行步骤索引（1-based, 0 表示未开始执行）。",
    )
    repair_count: int = Field(
        default=0,
        description="已执行的修补次数。",
    )
    total_steps: int = Field(
        default=0,
        description="执行计划总步骤数。",
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict,
        description="必要的环境变量快照（如 WORKDIR 等）。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据（如 task, acceptance_criteria 摘要等）。",
    )
    # ── 1.5.4 新增：追踪 ID ──
    trace_id: str = Field(
        default="",
        description="全生命周期追踪 ID（从 ExecutionPlan.trace_id 继承）。"
                    "断点恢复后延续原追踪链路。",
    )


# ── 持久化函数 ─────────────────────────────────────────

async def save_checkpoint(
    checkpoint: PEVRCheckpoint,
    memory_manager,
    session_id: str,
) -> None:
    """
    将状态检查点写入 Memory 后端。

    使用 memory_manager 的 artifact 存储机制（key: __pevr_checkpoint__）。

    Args:
        checkpoint: PEVRCheckpoint 实例。
        memory_manager: MemoryManager 实例。
        session_id: 会话 ID。
    """
    if memory_manager is None:
        logger.debug("MemoryManager 未配置，跳过 checkpoint 保存")
        return

    if not session_id:
        logger.debug("session_id 为空，跳过 checkpoint 保存")
        return

    try:
        checkpoint_json = checkpoint.model_dump_json()
        # 通过 update_session 存储 custom metadata
        if hasattr(memory_manager, 'update_session'):
            await memory_manager.update_session(
                session_id,
                __pevr_checkpoint__=checkpoint_json,
            )
        elif hasattr(memory_manager, 'save_artifact'):
            # 回退：存储为 artifact
            await memory_manager.save_artifact(
                session_id, "__pevr_checkpoint__", checkpoint_json,
            )
        else:
            logger.warning("MemoryManager 不支持 checkpoint 持久化，跳过")
            return

        logger.debug(
            "[Checkpoint] 已保存: state=%s step=%d/%d repair=%d",
            checkpoint.state, checkpoint.current_step_index,
            checkpoint.total_steps, checkpoint.repair_count,
        )
    except Exception as e:
        logger.warning("[Checkpoint] 保存失败: %s", e)


async def load_checkpoint(
    memory_manager,
    session_id: str,
) -> Optional[PEVRCheckpoint]:
    """
    从 Memory 后端读取状态检查点。

    Args:
        memory_manager: MemoryManager 实例。
        session_id: 会话 ID。

    Returns:
        PEVRCheckpoint 或 None（未找到检查点）。
    """
    if memory_manager is None or not session_id:
        return None

    try:
        checkpoint_json = None
        # 尝试从 session metadata 读取
        if hasattr(memory_manager, 'load_messages'):
            # MemoryManager 的 metadata 可能通过 get_session 获取
            pass

        # 尝试从 artifact 读取
        if hasattr(memory_manager, 'load_artifact'):
            checkpoint_json = await memory_manager.load_artifact(
                session_id, "__pevr_checkpoint__"
            )

        if checkpoint_json:
            checkpoint = PEVRCheckpoint.model_validate_json(checkpoint_json)
            logger.debug(
                "[Checkpoint] 已加载: state=%s step=%d/%d",
                checkpoint.state, checkpoint.current_step_index,
                checkpoint.total_steps,
            )
            return checkpoint

        logger.debug("[Checkpoint] 未找到检查点（session=%s）", session_id)
        return None
    except Exception as e:
        logger.warning("[Checkpoint] 加载失败: %s", e)
        return None


async def delete_checkpoint(
    memory_manager,
    session_id: str,
) -> None:
    """
    删除检查点（任务成功完成后清理）。

    Args:
        memory_manager: MemoryManager 实例。
        session_id: 会话 ID。
    """
    if memory_manager is None or not session_id:
        return

    try:
        if hasattr(memory_manager, 'delete_artifact'):
            await memory_manager.delete_artifact(
                session_id, "__pevr_checkpoint__"
            )
        logger.debug("[Checkpoint] 已删除（session=%s）", session_id)
    except Exception as e:
        logger.debug("[Checkpoint] 删除失败: %s", e)
