"""重建向量索引。

    python maintenance/reindex.py            # 重建消息索引（默认）
    python maintenance/reindex.py --check    # 只看索引全不全，不动它

什么时候需要它：`message_index` 和 DB 里的消息数对不上时。常见原因：

- 嵌入后端挂过。后端没起的那段时间里索引全部失败，而调用方把异常吞了，
  唯一症状是 Milvus 的行数不对，检索照样"能用"（只在已索引的那部分里搜）。
- 换了嵌入模型。不同模型的向量空间不通用，混着用会让相似度变成噪音。

重建是幂等的：先按 `message_id` 删掉已有的行再重插，跑几遍结果一样。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from store.db import Message, SessionLocal                        # noqa: E402
from sqlalchemy import func, select                          # noqa: E402
import store.memory as M                                           # noqa: E402


def check(verbose: bool = True) -> dict:
    """索引全不全？

    分母是"还没整合的消息"，不是"全部消息"：消息一旦整合成记忆，它的索引行
    就会被删掉（`consolidate` 里的 `drop_message_index_rows`）。拿全部消息当分母
    的话，跑过一次离线整理之后覆盖率永远是 0.2%，检查器只会误报。
    """
    from sqlalchemy import or_
    from store.db import _memory_conversation_ids

    base = [Message.kind == "chat",
            Message.conversation_id.in_(_memory_conversation_ids())]
    with SessionLocal() as s:
        total = s.execute(select(func.count()).select_from(Message)
                          .where(*base)).scalar() or 0
        pending = s.execute(
            select(func.count()).select_from(Message)
            .where(*base,
                   or_(Message.review_count.is_(None), Message.review_count == 0))
        ).scalar() or 0

    n_idx = M.count_indexed_messages()
    # 索引行会残留一些已整合的（按 id 范围删，边界会漏），所以封顶 100%
    covered = min(n_idx / pending, 1.0) if pending and n_idx >= 0 else 1.0
    if verbose:
        print(f"  chat 消息总数     : {total:,}")
        print(f"  其中已整合        : {total - pending:,}（设计上不进索引）")
        print(f"  待索引（未整合）  : {pending:,}")
        print(f"  message_index     : {n_idx:,}" if n_idx >= 0
              else "  message_index     : 打不开")
        if n_idx >= 0:
            print(f"  覆盖率            : {covered:.1%}")
            if covered < 0.9:
                print("  ⚠️ 索引不全 —— 检索只能在那部分里搜。"
                      "跑 `python maintenance/reindex.py` 重建。")
            else:
                print("  ✓ 索引是全的")
    fails = M.index_failures()
    if fails and verbose:
        print(f"  最近失败 {len(fails)} 次，最后一次：{fails[-1][:120]}")
    return {"messages": total, "pending": pending, "indexed": n_idx,
            "coverage": covered}


def main() -> int:
    ap = argparse.ArgumentParser(description="重建向量索引")
    ap.add_argument("--check", action="store_true", help="只检查，不重建")
    ap.add_argument("--limit", type=int, default=None, help="只重建最后 N 条")
    ap.add_argument("--batch", type=int, default=32, help="每批嵌多少条")
    args = ap.parse_args()

    print("=== 索引现状 ===")
    before = check()

    if args.check:
        return 0

    if before["indexed"] >= 0 and before["coverage"] >= 0.99:
        print("\n索引已经是全的，不用重建。")
        return 0

    print("\n=== 重建 ===")
    try:
        M.reindex_messages(batch=args.batch, limit=args.limit)
    except Exception as exc:
        print(f"重建失败：{type(exc).__name__}: {exc}")
        return 1

    print("\n=== 重建后 ===")
    after = check()
    return 0 if after["coverage"] >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
