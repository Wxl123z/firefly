"""Wiki 工具包：`wiki_search`（只读）。

对话中的模型能查 Wiki，不能写 —— 写会形成第二个记忆出口，与 `Memory` 双写，
两个真相源必然不一致。`read_file` / `grep_search` 也能读 Wiki 目录，故不重复造轮子。
"""
from __future__ import annotations

from tools.registry import ToolOutcome, tool
from wiki.search import search, stats


@tool(
    name="wiki_search",
    description=("查知识库（Wiki）。里面是按主题整理过、**每条论断都挂着来源锚点**"
                 "（`[memory:<id>]`）的笔记。\n"
                 "想问「我之前说过什么」用 retrieve_memory；"
                 "想知道「关于某件事整理出来的结论是什么」用这个。\n"
                 "拿到页名之后可以用 read_file 看全文。"),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "要找什么（自然语言）"},
            "top_k": {"type": "integer", "description": "最多返回几页，默认 5",
                      "default": 5},
        },
        "required": ["query"],
    },
    pack="wiki",
)
def wiki_search(query: str, top_k: int = 5) -> ToolOutcome:
    query = (query or "").strip()
    if not query:
        return ToolOutcome(content="错误：query 不能为空。", ok=False)

    try:
        hits, method = search(query, top_k=max(1, int(top_k)))
    except Exception as exc:                  # noqa: BLE001 —— 检索失败不该炸回合
        return ToolOutcome(content=f"错误：Wiki 检索失败：{type(exc).__name__}: {exc}",
                           ok=False)

    info = stats()
    if not hits:
        return ToolOutcome(
            content=(f"知识库里没有匹配 `{query}` 的页面。\n"
                     f"（现有 {info['total']} 页："
                     + " · ".join(f"{k} {v}" for k, v in sorted(info["by_type"].items()))
                     + "）\n"
                     "可以换几个词再试，或者用 retrieve_memory 回忆原话。"))

    lines = []
    for h in hits:
        lines.append(f"- **{h.title}**  `wiki/{h.path}`")
        if h.snippet:
            lines.append(f"    {h.snippet}")
    body = f"知识库里匹配 `{query}` 的页面（{len(hits)} 个）：\n" + "\n".join(lines)

    # 退化必须写在结果里：关键词匹配和语义匹配召回的东西不一样，静默换掉会被当成"没搜到"。
    if method == "keyword":
        body += ("\n\n⚠️ 本次用的是**关键词匹配**（语义检索的后端没起来），"
                 "换个说法可能会有不同结果。")
    body += f"\n\n（看全文用 `read_file`，路径是 `wiki/<上面的路径>`）"
    return ToolOutcome(content=body)
