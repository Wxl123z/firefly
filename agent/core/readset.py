"""会话内的「已读文件」记录 —— `edit_file` 的读后写保护依据。

规则：**没在本会话读过，就不许改。** 模型最容易犯的昂贵错误是"凭想象编辑"：
它记得文件里大概是那样的，写一个 `old_string`，运气好匹配失败，运气不好
**匹配到了别的地方**。

记 `(mtime_ns, size)` 而不是内容哈希：这里要防的是"读过之后文件被外部改了"，
一次 `os.stat` 就够。**不持久化** —— 进程重启后模型的历史也没了，它本来就必须重读。

`forget_all()` 补的是一个语义错位：判据是"这个会话里读过"，想保证的却是"模型
此刻看得见"。压缩把消息移出上下文之后两者分开了，守卫会对一个模型已经看不见的
文件说"读过，放行"。所以 `compress()` 成功之后必须全部作废；精确作废做不到 ——
`mark_read` 在工具层调用，不知道自己属于哪条消息。
"""
from __future__ import annotations

import os
from pathlib import Path

# 解析后的绝对路径字符串 → (st_mtime_ns, st_size)
_seen: dict[str, tuple[int, int]] = {}

#: 被压缩作废掉的路径 → 作废原因（原样给模型看）。
#: 和 `_seen` 分开存：`_seen` 回答"现在能不能改"，这个回答"为什么不能"。
#: 模型重读之后会重新进 `_seen`，所以不用从这里删。
_forgotten: dict[str, str] = {}


def _stat_key(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def mark_read(path: Path) -> None:
    """记下"这个文件在本会话被完整看过"。读失败时不要调它。"""
    key = _stat_key(path)
    if key is not None:
        _seen[str(path)] = key


def has_read(path: Path) -> bool:
    return str(path) in _seen


def stale_reason(path: Path) -> str | None:
    """能不能安全地改这个文件？返回 None 表示可以，否则返回**给模型看**的拒绝理由。"""
    key = str(path)
    if key not in _seen:
        try:
            rel = path.name
        except Exception:
            rel = str(path)
        # ⚠️ 分两种"没读过"：压缩移走的那种要说清楚是**被谁**移走的。
        if key in _forgotten:
            return (f"{rel} 的内容已经不在上下文里了（{_forgotten[key]}）。"
                    f"重新 read_file 一遍，拿到当前内容再改。")
        return (f"{rel} 在本次会话里还没被读过。"
                f"先用 read_file 看一遍内容，确认了再改 —— "
                f"不读就改很容易改在别的地方。")

    old = _seen[key]
    new = _stat_key(path)
    if new is None:
        return f"{path.name} 已经不存在了（读完之后被删掉或改名了）。重新确认一下。"
    if new != old:
        return (f"{path.name} 在你读过之后被改动过了"
                f"（大小 {old[1]} → {new[1]} 字节）。"
                f"重新 read_file 一次，拿最新的内容再改。")
    return None


def forget_all(reason: str) -> int:
    """把"读过"的记录**全部作废**（压缩把消息移出上下文时调）。返回作废了几个。

    作废不等于"从没读过"：原因记在 `_forgotten` 里，拒绝时原样告诉模型。
    """
    paths = list(_seen)
    _seen.clear()
    for p in paths:
        _forgotten[p] = reason
    return len(paths)


def reset() -> None:
    """会话重置时清空（`loop.reset_session()` 会调它）。

    ⚠️ **两个都要清**，否则上一个会话的 `_forgotten` 会让新会话里的文件被说成
    "上次压缩移走了"。
    """
    _seen.clear()
    _forgotten.clear()


def snapshot() -> dict[str, tuple[int, int]]:
    """只读快照，给测试和排查用。"""
    return dict(_seen)


def forgotten_snapshot() -> dict[str, str]:
    """被作废的路径 → 原因。同样给测试和排查用。"""
    return dict(_forgotten)
