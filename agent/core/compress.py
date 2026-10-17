"""上下文压缩：把较早的对话固化进长期记忆，再折叠成摘要。

顺序缺一不可：① consolidate 结构化记忆入库 → ② 摘要锚定合并（旧摘要 + 新内容
→ 新摘要）→ ③ 从 history 移除、水位线前移。少了 ①，被丢掉的消息里的具体事实
会永久消失，只剩模型自己写的叙述摘要。

④ 读后写保护跟着失效（`readset.forget_all`）：谁移除了消息，谁就负责让依赖
消息流的东西失效，否则守卫会替"盲改"背书。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from config import get_settings
from core import readset
from core.context import counter, split_history_by_tokens
from store.db import get_context_state, update_context_state
from llm import chat_json

settings = get_settings()

SUMMARY_SYSTEM = f"""你是对话上下文整理器，负责维护「{settings.NAME}」与{settings.USER_TERM}对话的滚动摘要。

输入包含两部分：① 此前已积累的摘要（可能为空）② 一段较早的对话原文。
请把新内容**合并**进摘要，输出更新后的完整摘要。

要求：
- 保留：事实、决定、约定、{settings.USER_TERM}的偏好与状态变化、未了结的事
- 时间一律用**绝对日期**（如 9-14），不要写"昨天/上周/前几天"
- 按主题分节，每节一两句；总长控制在 500 字以内
- 如果新对话推翻了旧信息，改写该条或标注「已过时」，不要简单叠加矛盾内容
- 不要编造对话里没有的内容
- 只输出 JSON：{{"summary": "更新后的摘要正文"}}"""


@dataclass
class CompressResult:
    ok: bool
    reason: str = ""
    summary: str = ""
    summary_upto_id: int = 0
    keep: list[dict] = field(default_factory=list)     # 保留的近期历史
    removed_count: int = 0
    removed_tokens: int = 0
    keep_tokens: int = 0
    consolidated: dict = field(default_factory=dict)


def check_guards(
    history: list[dict],
    *,
    turn_count: int,
    last_compress_turn: int,
    keep_recent_tokens: int | None = None,
    force: bool = False,
    ignore_cooldown: bool = False,
) -> tuple[bool, str, list[dict], list[dict]]:
    """护栏判定。返回（是否可压, 拒绝原因, older, recent）。

    两条护栏针对**会话内**压缩（怕模型每轮压一次把前缀缓存打烂）：冷却（距上次
    不足 N 轮）和最小收益（可压缩量太小）。退出时可传 `ignore_cooldown=True`，
    最小收益仍保留 —— 那是"值不值得花一次 LLM 调用"的问题。
    """
    keep = keep_recent_tokens or settings.COMPRESS_KEEP_TOKENS
    older, recent = split_history_by_tokens(history, keep)
    if not older:
        return False, "没有可压缩的内容（历史本来就短于保留窗口）。", older, recent

    removed = sum(counter.count(m["content"]) for m in older)
    if not force:
        cooldown = settings.COMPRESS_COOLDOWN_TURNS
        if (not ignore_cooldown) and last_compress_turn \
                and turn_count - last_compress_turn < cooldown:
            gap = turn_count - last_compress_turn
            return (False,
                    f"距上次压缩只过了 {gap} 轮（冷却 {cooldown} 轮），现在压缩得不偿失。",
                    older, recent)
        if removed < settings.COMPRESS_MIN_GAIN_TOKENS:
            return (False,
                    f"可压缩量只有约 {removed} token（低于 {settings.COMPRESS_MIN_GAIN_TOKENS}），"
                    "省不下多少，先别压。", older, recent)
    return True, "", older, recent


def compress(
    conversation_id: int,
    history: list[dict],
    *,
    turn_count: int = 0,
    keep_recent_tokens: int | None = None,
    focus: str | None = None,
    force: bool = False,
    ignore_cooldown: bool = False,
    verbose: bool = True,
) -> CompressResult:
    """执行一次压缩：固化记忆 → 合并摘要 → 移出历史 → 落库。"""
    state = get_context_state(conversation_id)
    old_summary = state["summary"]

    ok, reason, older, recent = check_guards(
        history,
        turn_count=turn_count,
        last_compress_turn=state["last_compress_turn"],
        keep_recent_tokens=keep_recent_tokens,
        force=force,
        ignore_cooldown=ignore_cooldown,
    )
    if not ok:
        return CompressResult(ok=False, reason=reason, summary=old_summary,
                              summary_upto_id=state["summary_upto_id"], keep=history)

    # ① 固化：压缩前先把事实存进长期记忆（否则细节永久丢失）；② 合并摘要；③ 落库
    consolidated = _consolidate(older, verbose=verbose)

    new_summary = _merge_summary(old_summary, older, focus)

    # 水位线前移 + 落库
    upto_id = max(m["id"] for m in older)
    update_context_state(
        conversation_id,
        summary=new_summary,
        summary_upto_id=upto_id,
        last_compress_turn=turn_count,
        compress_count=state["compress_count"] + 1,
    )

    removed_tokens = sum(counter.count(m["content"]) + 4 for m in older)
    keep_tokens = sum(counter.count(m["content"]) + 4 for m in recent)
    if verbose:
        print(f"[压缩] {len(older)} 条 / 约 {removed_tokens} token → 摘要"
              f"（保留 {len(recent)} 条 / 约 {keep_tokens} token）")

    # ④ 读后写保护跟着失效：`readset` 的判据是"这个会话里读过"，想保证的却是
    #    "模型此刻看得见"，`older` 移出上下文之后两者分开了。放在这里而不是
    #    调用方，是因为这是压缩这个操作自身的不变量。
    if older:
        readset.forget_all("上次压缩把它的内容移出了上下文")

    return CompressResult(
        ok=True, reason="已压缩", summary=new_summary, summary_upto_id=upto_id,
        keep=recent, removed_count=len(older), removed_tokens=removed_tokens,
        keep_tokens=keep_tokens, consolidated=consolidated,
    )


def _consolidate(messages: list[dict], *, verbose: bool = True) -> dict:
    """调用离线整合器。失败不阻断压缩（消息索引行会保留，仍可被检索到）。"""
    try:
        from offline.consolidate import process_messages
        return process_messages(messages, verbose=verbose)
    except Exception as exc:
        if verbose:
            print(f"[压缩] 记忆固化失败（不阻断压缩）：{type(exc).__name__}: {exc}")
        return {"error": f"{type(exc).__name__}: {exc}"}


def _merge_summary(old_summary: str, older: list[dict], focus: str | None) -> str:
    """把被压缩的对话合并进旧摘要。LLM 失败时退化为机械拼接，保证不丢内容。"""
    lines = []
    for m in older:
        ts = m.get("ts")
        stamp = f"[{ts:%m-%d %H:%M}] " if isinstance(ts, datetime) else ""
        who = settings.USER_TERM if m["role"] == "user" else settings.NAME
        lines.append(f"{stamp}{who}：{m['content']}")
    body = "\n".join(lines)

    user = f"【已有摘要】\n{old_summary or '（暂无）'}\n\n【新增对话原文】\n{body}"
    if focus:
        user += f"\n\n【希望重点保留】{focus}"

    try:
        data = chat_json(
            [{"role": "system", "content": SUMMARY_SYSTEM},
             {"role": "user", "content": user}],
            temperature=0.2, max_tokens=1200,
        )
        summary = (data.get("summary") or "").strip()
        if summary:
            return summary
    except Exception:
        pass

    # 兜底：宁可摘要粗糙，也不能丢内容
    return (f"{old_summary}\n\n【{older[0].get('ts', '')} 起的对话】\n{body}"
            if old_summary else f"【{older[0].get('ts', '')} 起的对话】\n{body}")
