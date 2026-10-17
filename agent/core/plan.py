"""任务清单（todo）—— 计划状态的存放与渲染。

计划**存在别处、每轮重新渲染**：压缩会把早期的 `todo_write` 调用移出上下文，
从对话历史里捞计划会在压缩后失忆。渲染进尾部是因为前缀要逐字节稳定才能命中
缓存，而计划每几步就变一次；代价只是每轮多 100–200 token。

写入是**全量替换**：模型每次给出完整清单、服务端覆盖，因此不存在"增量操作
序列错了导致状态错乱"，而且天然幂等。
"""
from __future__ import annotations

import json
from pathlib import Path

from config import DATA_DIR

PLAN_DIR = DATA_DIR / "plan"

#: 模型可能用中文写状态（我们的提示词是中文的，这很自然）→ 归一化
_STATUS_ALIASES = {
    "pending": "pending", "todo": "pending", "待办": "pending", "未开始": "pending",
    "in_progress": "in_progress", "doing": "in_progress", "进行中": "in_progress",
    "completed": "completed", "done": "completed", "完成": "completed", "已完成": "completed",
}

_MARK = {"completed": "✅", "in_progress": "🔄", "pending": "⬜"}
_LABEL = {"completed": "完成", "in_progress": "进行中", "pending": "待办"}


class PlanError(Exception):
    """清单本身不合法（给模型看的理由在消息里）。"""


def normalize(items) -> list[dict]:
    """校验并归一化清单。**不合法就抛，不静默丢弃。**

    最关键的一条是"最多一个 in_progress"：三个任务同时"进行中"等于没有计划。
    """
    if not isinstance(items, list):
        raise PlanError(f"items 必须是数组，收到 {type(items).__name__}。")
    if not items:
        raise PlanError("items 是空的。想清空计划请直接说，不用调工具。")

    out: list[dict] = []
    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise PlanError(f"第 {i + 1} 项不是对象，是 {type(raw).__name__}。"
                            f"每项要形如 {{\"content\": \"做什么\", \"status\": \"pending\"}}。")
        content = str(raw.get("content") or raw.get("step") or "").strip()
        if not content:
            raise PlanError(f"第 {i + 1} 项没有 content（要写清做什么）。")
        status = str(raw.get("status") or "pending").strip().lower()
        status = _STATUS_ALIASES.get(status, "")
        if not status:
            raise PlanError(f"第 {i + 1} 项的 status 是 {raw.get('status')!r}，"
                            f"只能是 pending / in_progress / completed。")
        out.append({"content": content, "status": status})

    running = [x["content"] for x in out if x["status"] == "in_progress"]
    if len(running) > 1:
        raise PlanError(
            f"同时**最多只能有一个** in_progress，你给了 {len(running)} 个：{running}。\n"
            "多个任务同时'进行中'等于没有计划。把其余几个改成 pending，"
            "只留当前正在做的那一个。")
    return out


def _path(conversation_id: int) -> Path:
    return PLAN_DIR / f"{conversation_id}.json"


def load(conversation_id: int) -> list[dict]:
    p = _path(conversation_id)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = data.get("items") if isinstance(data, dict) else data
    try:
        return normalize(items) if items else []
    except PlanError:
        return []           # 存坏了就当没有，不要因为一个坏文件把整轮打断


def save(conversation_id: int, items: list[dict], explanation: str = "") -> None:
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    _path(conversation_id).write_text(
        json.dumps({"explanation": explanation, "items": items}, ensure_ascii=False,
                   indent=2),
        encoding="utf-8")


def clear(conversation_id: int) -> None:
    _path(conversation_id).unlink(missing_ok=True)


def render(items: list[dict]) -> str:
    """渲染进尾部通知的那一段。空清单返回空串（不占 token）。"""
    if not items:
        return ""
    done = sum(1 for x in items if x["status"] == "completed")
    doing = sum(1 for x in items if x["status"] == "in_progress")
    todo = len(items) - done - doing
    lines = ["【当前任务】"]
    for x in items:
        lines.append(f"  {_MARK[x['status']]} {x['content']}")
    lines.append(f"（{len(items)} 项：{done} 完成，{doing} 进行中，{todo} 待办）")
    if done == len(items):
        lines.append("全部完成了 —— 该收尾并汇报结果。")
    return "\n".join(lines)


def counts(items: list[dict]) -> dict:
    return {
        "total": len(items),
        "completed": sum(1 for x in items if x["status"] == "completed"),
        "in_progress": sum(1 for x in items if x["status"] == "in_progress"),
        "pending": sum(1 for x in items if x["status"] == "pending"),
    }
