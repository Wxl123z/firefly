"""LLM Wiki（v20 阶段 8/9）。

把 `memory` 表里散装的条目，离线整理成一套**人可读、机器可检索、可重建**的
markdown 知识库。**它是记忆的第二视图，不是第二份真相**：单向派生，只读
`Memory` / `Rollup` / `Person`，只写 `data/wiki/`；**对话中是只读的**（§7.8）。
"""
from wiki.build import BuildResult, build, resolve_topics      # noqa: F401
from wiki.contract import WIKI_CONTRACT, render_contract       # noqa: F401
from wiki.pages import Evidence, Page, all_pages, parse, slug  # noqa: F401

__all__ = ["build", "resolve_topics", "BuildResult", "Page", "Evidence",
           "parse", "slug", "all_pages", "WIKI_CONTRACT", "render_contract"]
