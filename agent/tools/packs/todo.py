"""任务清单工具包（plan）：`todo_write`。
文件名用 `todo.py` 而非 `plan.py`，以免和存储实现 `core/plan.py` 撞名。
"""
from __future__ import annotations

import core.plan as plan_store
from tools.registry import ToolOutcome, tool


def _conversation_id() -> int:
    """当前会话 id。取不到就用 0（测试/无会话场景，也能正常读写）。"""
    try:
        from core.loop import get_session
        return get_session().conversation_id or 0
    except Exception:
        return 0


@tool(
    name="todo_write",
    description=("维护当前任务清单。**每次给出完整清单**（全量替换，不是增量）。\n"
                 "任务超过 3 步时，先列出来再动手；每完成一步就更新它。\n"
                 "硬性约束：**同时最多只能有一个 in_progress** —— "
                 "多个任务同时'进行中'等于没有计划。"),
    parameters={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "完整任务清单，按执行顺序",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "这一步要做什么"},
                        "status": {"type": "string",
                                   "enum": ["pending", "in_progress", "completed"]},
                    },
                    "required": ["content", "status"],
                },
            },
            "explanation": {"type": "string",
                            "description": "可选：为什么这样调整计划（改计划时值得写）"},
        },
        "required": ["items"],
    },
    pack="plan",
)
def todo_write(items: list, explanation: str | None = None) -> ToolOutcome:
    try:
        normalized = plan_store.normalize(items)
    except plan_store.PlanError as exc:
        return ToolOutcome(
            content=f"错误：{exc}\n"
                    f"正确写法：{{\"items\": ["
                    f"{{\"content\": \"读代码\", \"status\": \"completed\"}}, "
                    f"{{\"content\": \"改函数\", \"status\": \"in_progress\"}}]}}",
            ok=False)

    cid = _conversation_id()
    try:
        plan_store.save(cid, normalized, explanation or "")
    except OSError as exc:
        return ToolOutcome(content=f"错误：计划存不下来：{exc}", ok=False)

    c = plan_store.counts(normalized)
    head = (f"计划已更新（{c['total']} 项：{c['completed']} 完成，"
            f"{c['in_progress']} 进行中，{c['pending']} 待办）。"
            f"它每一轮都会出现在你的上下文里，不用自己记着。")
    if explanation:
        head += f"\n（调整原因：{explanation}）"
    return ToolOutcome(content=f"{head}\n\n{plan_store.render(normalized)}")
