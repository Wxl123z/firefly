"""退出时的离线工作：压缩上下文 → 固化记忆 → 每日总结 →（可选）LLM Wiki。

放在退出而不是启动：压缩会改写前缀、炸掉缓存（退出时没有缓存可炸）。
廉价闸门：只有历史真的超预算才压缩，所以退出通常是零 LLM 调用的。

非正常退出（强杀进程）不会跑这里；`compress()` 在会话内阈值触发时也会固化记忆。
"""
from __future__ import annotations

from config import get_settings
from core.context import counter
from store.db import get_context_state, get_or_create_conversation, load_history

settings = get_settings()


def _conversation_id() -> int:
    """优先用会话里的 id；没启动过就按**当前 Profile** 取默认会话。"""
    try:
        from core.loop import get_session
        cid = get_session().conversation_id
        if cid:
            return cid
    except Exception:
        pass
    try:
        from core.profile import active
        prof = active()
        source = prof.conversation_source
        workspace = prof.conversation_workspace()
    except Exception:
        source, workspace = None, None
    return get_or_create_conversation(source=source, workspace=workspace)


def _current_history(cid: int) -> list[dict]:
    """优先用内存里的会话历史（含本会话内已发生的压缩）；否则从库装载。"""
    try:
        from core.loop import get_session
        session = get_session()
        if session.ready and session.history:
            return list(session.history)
    except Exception:
        pass
    ctx = get_context_state(cid)
    return load_history(cid, after_id=ctx["summary_upto_id"])


def compress_if_needed(cid: int, *, verbose: bool = True) -> dict:
    """历史超预算才压缩（廉价闸门）。返回统计。"""
    history = _current_history(cid)
    tokens = sum(counter.count(m["content"]) + 4 for m in history)
    budget = settings.OFFLINE_KEEP_TOKENS

    if tokens <= budget + settings.COMPRESS_MIN_GAIN_TOKENS:
        if verbose:
            print(f"  上下文约 {tokens:,} token（预算 {budget:,}）→ 无需压缩")
        return {"skipped": True, "tokens": tokens}

    if verbose:
        print(f"  上下文约 {tokens:,} token → 压缩到约 {budget:,} token…")
    from core.compress import compress
    result = compress(
        cid, history,
        turn_count=0,
        keep_recent_tokens=budget,
        ignore_cooldown=True,     # 退出时缓存已经无所谓了，冷却没有意义
        verbose=verbose,
    )
    if not result.ok:
        if verbose:
            print(f"  未压缩：{result.reason}")
        return {"skipped": True, "reason": result.reason, "tokens": tokens}
    return {"skipped": False, "removed": result.removed_count,
            "saved": result.removed_tokens, "tokens": tokens}


def run(*, verbose: bool = True) -> dict:
    """退出时的完整离线流程。任一步失败不影响后续步骤。"""
    stats: dict = {}
    cid = _conversation_id()

    # ① 压缩（含对它丢掉的那批消息做记忆固化）
    try:
        stats["compress"] = compress_if_needed(cid, verbose=verbose)
    except Exception as exc:
        stats["compress"] = {"error": f"{type(exc).__name__}: {exc}"}
        if verbose:
            print(f"  压缩失败（继续）：{stats['compress']['error']}")

    # ② 记忆固化：把剩下的未处理消息（含保留窗口内的）补齐
    try:
        from offline.consolidate import process_all
        stats["consolidate"] = process_all(verbose=verbose)
    except Exception as exc:
        stats["consolidate"] = {"error": f"{type(exc).__name__}: {exc}"}
        if verbose:
            print(f"  记忆固化失败（继续）：{stats['consolidate']['error']}")

    # ③ 每日总结
    try:
        from store.rollup import catch_up
        stats["rollup"] = catch_up(days_back=settings.OFFLINE_ROLLUP_DAYS, verbose=verbose)
    except Exception as exc:
        stats["rollup"] = {"error": f"{type(exc).__name__}: {exc}"}
        if verbose:
            print(f"  每日总结失败（继续）：{stats['rollup']['error']}")

    # ④ LLM Wiki：**必须排在 rollup 之后**（它的 sources/ 页拿 Rollup 当骨架）。
    # 纯派生物，失败不影响前序步骤，下次再建即可。
    if settings.WIKI_BUILD_ON_EXIT:
        try:
            from wiki.build import build as wiki_build
            result = wiki_build()
            stats["wiki"] = {"new_memories": result.new_memories,
                             "pages_written": result.pages_written,
                             "page_count": result.page_count}
            if verbose:
                print(f"  Wiki：{result.summary()}")
        except Exception as exc:
            stats["wiki"] = {"error": f"{type(exc).__name__}: {exc}"}
            if verbose:
                print(f"  Wiki 构建失败（继续）：{stats['wiki']['error']}")

    return stats


if __name__ == "__main__":
    run()
