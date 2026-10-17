"""每日总结：幂等补课，无守护进程。

无 cron / 常驻守护，三个入口（启动补课 / 退出 / 常驻时定时）调同一个幂等函数，
幂等靠 `Rollup.watermark_message_id`（这一天没有新消息就跳过）。

产物只落 `rollup` 表，不写长期记忆（长期事实由 consolidate 负责）。
"""
from __future__ import annotations

from datetime import date as date_cls

from store.db import get_messages_on_day, get_rollup, list_active_days, upsert_rollup
from config import get_settings

settings = get_settings()


def build_daily(day: date_cls | str, *, verbose: bool = True) -> dict | None:
    """生成/刷新某一天的回顾。没有新消息则跳过（幂等）。"""
    if isinstance(day, str):
        day = date_cls.fromisoformat(day)

    msgs = get_messages_on_day(day)
    if not msgs:
        return None

    watermark = max(m["id"] for m in msgs)   # 这一天最后一条消息的 id
    existing = get_rollup(day)
    if existing and existing["watermark_message_id"] >= watermark:
        # 幂等：水位线已覆盖当天全部消息 → 跳过。
        # 必须带回 skipped 标记，调用方靠它区分"跳过"与"生成了"。
        if verbose:
            print(f"  {day.isoformat()}: 无新消息，跳过")
        return {"skipped": True, **existing}

    from tools.core import daily_digest_for_rollup

    transcript = "\n".join(
        f"[{m['ts']:%H:%M}] {settings.USER_TERM if m['speaker_type'] == 'owner' else settings.NAME}：{m['content']}"
        for m in msgs
    ) # 从当日信息中组装出对话
    summary = daily_digest_for_rollup(day, transcript)
    if not summary:
        if verbose:
            print(f"  {day.isoformat()}: 总结生成失败（下次重试）")
        return None

    upsert_rollup(day, summary, watermark, len(msgs))
    if verbose:
        print(f"  {day.isoformat()}: {len(msgs)} 条消息 → 回顾 ✓")
    return {"date": day.isoformat(), "summary": summary,
            "watermark_message_id": watermark, "message_count": len(msgs)}


def catch_up(*, days_back: int = 7, verbose: bool = True) -> dict:
    """补课：扫描最近有对话的日子，为缺失/过期的日子生成回顾。"""
    days = list_active_days(limit=days_back)
    done = 0
    skipped = 0
    failed = 0
    for day in days:
        try:
            result = build_daily(day, verbose=verbose)
        except Exception as exc:
            failed += 1
            if verbose:
                print(f"  {day.isoformat()}: 失败 {type(exc).__name__}: {exc}")
            continue
        if result is None:
            failed += 1                      # 生成失败（下次重试）
        elif result.get("skipped"):
            skipped += 1
        else:
            done += 1
    if verbose:
        print(f"每日总结补课完成：检查 {len(days)} 天，"
              f"生成/更新 {done} 天，跳过 {skipped} 天，失败 {failed} 天")
    return {"checked": len(days), "done": done, "skipped": skipped, "failed": failed}


if __name__ == "__main__":
    catch_up()
