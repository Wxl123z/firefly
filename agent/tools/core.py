"""常用工具：陪伴 Profile 的常驻工具集，走 `tools=` 字段，恒定不变。

**"哪些工具常驻"由 `profile.Profile.core_tools` 决定**，不由这里的装饰器声明。
**这些工具的定义会渲染进 system 前缀，任何字段的改动都会让缓存纪元失效。**
"""
from __future__ import annotations

import json
from datetime import date as date_cls

import requests

from config import get_settings
from tools.registry import ToolOutcome, tool

settings = get_settings()


# ---------------------------------------------------------------- 记忆检索

@tool(
    name="retrieve_memory",
    description=("检索长期记忆与近期对话原文。当对方提到过去的事，或你需要回忆对方的偏好、"
                 "约定、近况、人物关系时调用。"),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "要回忆什么，自然语言，例如「对方对咖啡的偏好」"},
            "top_k": {"type": "integer", "description": "返回条数，默认 5", "default": 5},
        },
        "required": ["query"],
    },
    pack="memory",
)
def retrieve_memory(query: str, top_k: int = 5) -> str:
    from store.memory import recall_memory_dual   # 延迟导入：避免启动即连 Milvus

    items = recall_memory_dual(query, top_k=top_k)
    if not items:
        return "没有检索到相关记忆。（这说明长期记忆里确实没有，可以如实说你不记得。）"
    lines = []
    for it in items:
        sim = it.get("similarity")
        tag = it.get("source", "记忆")
        score = f"（相关度 {sim:.2f}）" if isinstance(sim, (int, float)) else ""
        lines.append(f"- [{tag}]{score} {it.get('content', '')}")
    return "检索到以下记忆（可能相关也可能过时，准确就自然引用）：\n" + "\n".join(lines)


# ---------------------------------------------------------------- 工具检索

@tool(
    name="search_tools",
    description=("检索扩展工具。当任务需要的能力不在常用工具中时调用；匹配到的工具会被"
                 "注册进你的可用工具列表，之后可直接调用它们。"),
    parameters={
        "type": "object",
        "properties": {
            "task": {"type": "string",
                     "description": "你要完成的任务，自然语言，越具体越好"},
            "pack": {"type": "string",
                     "description": "可选：限定工具包，如 fs / code / web / schedule / memory"},
        },
        "required": ["task"],
    },
    pack="meta",
)
def search_tools(task: str, pack: str | None = None) -> ToolOutcome:
    from core import toolsearch
    from tools.registry import REGISTRY, expand_closure, pack_index

    names = expand_closure(toolsearch.search_tools(task, pack=pack))
    if not names:
        packs = "、".join(p for p, _ in pack_index()) or "（暂无）"
        return ToolOutcome(
            content=(f"没有找到匹配「{task}」的工具。当前可用工具包：{packs}。\n"
                     "可以换个说法再检索一次；如果确实没有，请直接说你缺少什么能力。"),
            control="load_tools", payload={"loaded": []}, ok=False)

    descs = "；".join(f"{n}（{REGISTRY[n].description.split('。')[0]}）" for n in names)
    return ToolOutcome(
        content=(f"已为本次任务加载 {len(names)} 个工具：{descs}。\n"
                 "它们的完整定义已经加入你的可用工具列表，**现在就可以直接调用**"
                 "（和常用工具完全一样）。请立刻调用其中合适的那个。"),
        control="load_tools", payload={"loaded": names})


# ---------------------------------------------------------------- 上下文压缩

@tool(
    name="compress_context",
    description=("压缩较早的对话：先把其中的重要事实存入长期记忆，再把它们整理成摘要并从上下文中"
                 "移除，最近的对话保持不变。上下文占用超过 60%，或对方明确要求"
                 "「忘掉前面」「清一下上下文」时使用。"),
    parameters={
        "type": "object",
        "properties": {
            "focus": {"type": "string",
                      "description": "可选：希望摘要重点保留什么，例如「关于换工作的决定」"},
            "keep_recent_tokens": {"type": "integer",
                                   "description": "可选：最近多少 token 逐字保留，默认 4000"},
        },
        "required": [],
    },
    pack="meta",
)
def compress_context(focus: str | None = None, keep_recent_tokens: int | None = None) -> ToolOutcome:
    """只提交压缩请求；是否执行由 graph 按护栏判定（冷却 / 最小收益）。"""
    return ToolOutcome(
        content="（压缩请求已提交，系统正在处理……）",
        control="compress",
        payload={"focus": focus, "keep_recent_tokens": keep_recent_tokens})


# ---------------------------------------------------------------- 当日回顾

_TODAY_DIGEST_SYSTEM = """你是对话整理助手。把下面这一天里{term}与助手（{name}）的对话整理成结构化回顾。

要求：
- 按话题分节，每节列出：发生了什么、{term}的态度/情绪、达成的约定或决定
- 人名、日期、数字必须与原文一致，不许编造
- 疑问或未完成的事单独列一节「未了结」
- 只输出 JSON：{{"digest": "整理后的正文（markdown 小标题 + 列表）"}}"""


@tool(
    name="summarize_today",
    description=("取出今天的对话内容（结构化回顾形式）。当对方问「今天我们聊了什么」"
                 "或你想回顾今天时调用。"),
    parameters={
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "可选：日期 YYYY-MM-DD，默认今天"},
        },
        "required": [],
    },
    pack="memory",
)
def summarize_today(date: str | None = None) -> str:
    from core.context import counter
    from store.db import get_messages_on_day, get_rollup
    from llm import chat_json

    try:
        day = date_cls.fromisoformat(date) if date else date_cls.today()
    except ValueError:
        return f"日期格式不对：{date}。请用 YYYY-MM-DD。"

    msgs = get_messages_on_day(day)
    if not msgs:
        return f"{day.isoformat()} 没有对话记录。"

    # 已经有覆盖当天全部消息的每日回顾 → 直接复用，不重复花一次 LLM 调用
    existing = get_rollup(day)
    if existing and existing.get("summary"):
        latest_id = max(m["id"] for m in msgs)
        if existing["watermark_message_id"] >= latest_id:
            return f"{day.isoformat()} 的每日回顾（已归档）：\n{existing['summary']}"

    transcript = "\n".join(
        f"[{m['ts']:%H:%M}] {settings.USER_TERM if m['speaker_type'] == 'owner' else settings.NAME}：{m['content']}"
        for m in msgs
    )

    # 短对话直接给原文（模型自己就能讲）；长对话先做一次结构化压缩
    if counter.count(transcript) <= 1500:
        return f"{day.isoformat()} 的对话原文（{len(msgs)} 条）：\n{transcript}"

    data = chat_json(
        [{"role": "system", "content": _TODAY_DIGEST_SYSTEM.format(
            name=settings.NAME, term=settings.USER_TERM)},
         {"role": "user", "content": transcript}],
        temperature=0.2, max_tokens=1200,
    )
    digest = (data.get("digest") or "").strip()
    if not digest:
        return f"{day.isoformat()} 的对话原文（{len(msgs)} 条，整理失败）：\n{transcript[:4000]}"
    return f"{day.isoformat()} 的结构化回顾（共 {len(msgs)} 条消息）：\n{digest}"


# ---------------------------------------------------------------- 联网搜索

@tool(
    name="web_search",
    description="联网搜索最新信息。需要实时信息（新闻、天气、价格、版本、文档）时调用。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "description": "返回条数，默认 5", "default": 5},
        },
        "required": ["query"],
    },
    pack="web",
)
def web_search(query: str, max_results: int = 5) -> str:
    provider = (settings.SEARCH_PROVIDER or "duckduckgo").lower()
    try:
        if provider == "tavily":
            items = _search_tavily(query, max_results)
        elif provider == "searxng":
            items = _search_searxng(query, max_results)
        else:
            items = _search_duckduckgo(query, max_results)
    except Exception as exc:
        return (f"联网搜索失败：{type(exc).__name__}: {exc}\n"
                "可以说明搜索暂时不可用，或改用你已有的知识回答（并注明不确定性）。")
    if not items:
        return f"没有搜到「{query}」的结果。"
    lines = [f"{i + 1}. {it['title']}\n   {it['url']}\n   {it['snippet']}"
             for i, it in enumerate(items)]
    return f"「{query}」的搜索结果：\n" + "\n".join(lines)


def _search_tavily(query: str, n: int) -> list[dict]:
    if not settings.SEARCH_API_KEY:
        raise RuntimeError("未配置 SEARCH_API_KEY（provider=tavily）")
    r = requests.post("https://api.tavily.com/search", timeout=20, json={
        "api_key": settings.SEARCH_API_KEY, "query": query,
        "max_results": n, "search_depth": "basic",
    })
    r.raise_for_status()
    return [{"title": x.get("title", ""), "url": x.get("url", ""),
             "snippet": (x.get("content") or "")[:400]}
            for x in r.json().get("results", [])][:n]


def _search_searxng(query: str, n: int) -> list[dict]:
    base = (settings.SEARCH_BASE_URL or "").rstrip("/")
    if not base:
        raise RuntimeError("未配置 SEARCH_BASE_URL（provider=searxng）")
    r = requests.get(f"{base}/search", timeout=20,
                     params={"q": query, "format": "json", "language": "zh-CN"})
    r.raise_for_status()
    return [{"title": x.get("title", ""), "url": x.get("url", ""),
             "snippet": (x.get("content") or "")[:400]}
            for x in r.json().get("results", [])][:n]


def _search_duckduckgo(query: str, n: int) -> list[dict]:
    """DuckDuckGo HTML 端点（无需 key）。国内网络可能不可达，失败由上层兜底。"""
    import re
    r = requests.post("https://html.duckduckgo.com/html/", timeout=20,
                      data={"q": query},
                      headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    r.raise_for_status()
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
        r'.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.S)
    items = []
    for m in pattern.finditer(r.text):
        strip = lambda s: re.sub(r"<[^>]+>", "", s).strip()
        items.append({"title": strip(m.group("title")),
                      "url": m.group("url"),
                      "snippet": strip(m.group("snippet"))[:400]})
        if len(items) >= n:
            break
    return items


# ---------------------------------------------------------------- 内部工具（不暴露给模型）

def daily_digest_for_rollup(day: date_cls, transcript: str) -> str:
    """供离线 daily_rollup 复用（不是模型可调工具）。"""
    from llm import chat_json
    data = chat_json(
        [{"role": "system", "content": _TODAY_DIGEST_SYSTEM.format(
            name=settings.NAME, term=settings.USER_TERM)},
         {"role": "user", "content": transcript}],
        temperature=0.2, max_tokens=1200,
    )
    return (data.get("digest") or "").strip()
