"""命令行入口：

    python -m wiki.build            增量构建（--rebuild 清空 wiki/ 从 0 全量重跑）
    python -m wiki.build --no-llm   不调 LLM（综述留占位）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wiki.build import build            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 LLM Wiki")
    ap.add_argument("--rebuild", action="store_true",
                    help="清空 wiki/ 后从 0 全量重跑（LLM 缓存保留）")
    ap.add_argument("--no-llm", action="store_true", help="不调用 LLM")
    ap.add_argument("--root", default=None, help="Wiki 根目录，默认 config.WIKI_DIR")
    args = ap.parse_args()

    result = build(root=Path(args.root) if args.root else None,
                   rebuild=args.rebuild,
                   use_llm=False if args.no_llm else None)
    print(result.summary())
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
