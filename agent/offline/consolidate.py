"""离线记忆整合：重读未处理消息 → 结构化提取 → 写入记忆库。

- 触发：手动（python consolidate.py）/ CLI 退出 / 压缩前置步骤 / Phase 1 定时
- 中立事实提取，严格区分 stated_fact / stated_preference / event / inference /
  discussion / summary；直接陈述必须带 quote（原文摘录）锚定保真
- inference 单次只入库为推断（support_count=1），靠 corroborate() 累积独立佐证
- 置信度低于 MEMORY_MIN_CONFIDENCE → 只进 SQLite（待审核池），不进向量库
- 标记：处理过的消息 review_count+1，并删除其消息索引行
"""
from datetime import datetime

from config import get_settings
from store.db import MemoryType, get_unreviewed_messages, mark_messages_reviewed
from llm import chat_json
from store.memory import corroborate, drop_message_index_rows

settings = get_settings()

CONSOLIDATE_SYSTEM = f"""你是对话记忆提取器，负责为「{settings.NAME}」整理与{settings.USER_TERM}的对话记忆。

把下面这段对话按话题切分，逐条提取结构化记忆。
注意：{settings.USER_TERM}是与「{settings.NAME}」平等对话的人，不是服务对象——提取时保持中立、不加评判。

type 必须严格区分（这是最重要的规则）：
- stated_fact       {settings.USER_TERM}直接陈述的事实（"我养了一只叫小橘的猫"）
- stated_preference {settings.USER_TERM}直接陈述的偏好（"我不吃香菜"）
- event             发生的事件 / 计划 / 约定（"下周一和朋友去浙江旅游"）
- discussion        双方讨论过的话题内容（"讨论了项目 X 的做法"）
- inference         你的推断（"{settings.USER_TERM}可能习惯晚睡"）——必须能由对话内容推出，不许凭空猜
- summary           整段对话的概括

每条记忆输出字段：
- fact:       简洁陈述，一句话，人名/日期/数字必须与对话完全一致
- abstract:   稍详细描述（可含情绪与脉络）
- quote:      若为{settings.USER_TERM}直接陈述，摘录其原话（一字不改）；推断类留空字符串
- confidence: 0-1，你对准确性的把握（直接陈述 ≥0.9；推断 0.4-0.7；不确定就低）
- topic:      话题名（如"宠物""工作""出行计划"）

不许编造对话中没有的信息。只输出 JSON：
{{"memories": [{{"type": "stated_fact", "fact": "...", "abstract": "...", "quote": "...", "confidence": 0.95, "topic": "..."}}]}}"""

_VALID_TYPES = {t.value: t for t in MemoryType}

_EMPTY_STATS = {"days": 0, "created": 0, "corroborated": 0, "messages": 0}


def _day_of(m: dict) -> str:
    """消息 → `YYYY-MM-DD`。**取不到就返回空串，绝不返回 "None"。**

    消息字典有两种形状：`db._to_dict()` 用 `ts`（compress 走这条），
    `db.get_unreviewed_messages()` 用 `created_at`（process_all 走这条），两种都认。
    返回空串表示"不知道日期"，让调用方看得见。
    """
    for key in ("created_at", "ts"):
        ts = m.get(key)
        if isinstance(ts, datetime):
            return ts.date().isoformat()
    return ""


def _group_by_date(messages: list) -> dict:
    groups: dict = {}
    for m in messages:
        groups.setdefault(_day_of(m), []).append(m)
    return groups


def _parse_time(day: str) -> datetime | None:
    try:
        return datetime.fromisoformat(day)
    except (TypeError, ValueError):
        return None


def _label(speaker_type: str) -> str:
    return settings.USER_TERM if speaker_type in ("owner", "person") else settings.NAME


def process_messages(messages: list[dict], *, verbose: bool = True,
                     mark: bool = True) -> dict:
    """整合指定的一批消息（按天分组）。

    mark=False 时只提取不标记——预留给「只提取、稍后再标记」的场景。
    """
    if not messages:
        return dict(_EMPTY_STATS)

    groups = _group_by_date(messages)
    stats = {"days": len(groups), "created": 0, "corroborated": 0,
             "messages": len(messages)}

    for day, day_msgs in sorted(groups.items()):
        if not day:
            # 没有日期的分组照常提取（不丢数据），但要提示出来。
            if verbose:
                print(f"  ⚠️ {len(day_msgs)} 条消息没有时间戳，"
                      f"抽出的记忆将没有日期（date_range/time 留空）")
        text = "\n".join(f"[{_label(m['speaker_type'])}] {m['content']}" for m in day_msgs)
        data = chat_json(
            [{"role": "system", "content": CONSOLIDATE_SYSTEM},
             {"role": "user", "content": text}],
            max_tokens=1536,
        )
        memories = data.get("memories", [])
        if not memories:
            if verbose:
                print(f"  {day}: 提取失败（跳过，下次重试）")
            continue

        ids = [m["id"] for m in day_msgs]
        source_ref = f"{min(ids)}-{max(ids)}" if len(ids) > 1 else str(ids[0])
        evidence = [{"message_ids": ids, "date": day} if day
                    else {"message_ids": ids}]

        for item in memories:
            if not isinstance(item, dict):
                continue
            fact = str(item.get("fact", "")).strip()
            if not fact:
                continue
            type_ = _VALID_TYPES.get(str(item.get("type", "")).strip(), MemoryType.stated_fact)
            confidence = item.get("confidence")
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None
            result = corroborate(
                fact,
                type_=type_,
                abstract=(str(item.get("abstract", "")).strip() or None),
                quote=(str(item.get("quote", "")).strip() or None),
                evidence=evidence,
                confidence=confidence,
                topic=(str(item.get("topic", "")).strip()[:100] or None),
                date_range=(day or None),
                source_ref=source_ref,
                time=_parse_time(day),
            )
            stats[result] += 1

        if mark:
            mark_messages_reviewed(ids)
            drop_message_index_rows(ids)
        if verbose:
            print(f"  {day}: 整合 {len(day_msgs)} 条消息 → {len(memories)} 条记忆 ✓")

    return stats


def process_all(verbose: bool = True, limit: int | None = None) -> dict:
    """整合未处理消息（全库扫描，启动/退出时用）。limit 限制单次处理条数。"""
    msgs = get_unreviewed_messages(limit=limit) if limit else get_unreviewed_messages()
    if not msgs:
        if verbose:
            print("没有未处理的消息")
        return dict(_EMPTY_STATS)

    stats = process_messages(msgs, verbose=verbose)
    if verbose:
        print(f"\n整合完成：{stats['messages']} 条消息 → "
              f"新建 {stats['created']} 条 / 佐证 {stats['corroborated']} 条")
    return stats


if __name__ == "__main__":
    process_all()
