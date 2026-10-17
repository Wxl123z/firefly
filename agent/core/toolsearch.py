"""工具检索：按任务描述召回需要的扩展工具。

向量复用本地 bge-m3（与记忆检索同一套），索引单元是**工具**不是包。**召回
导向**：漏掉工具的代价远大于多给几个，所以嵌入不可用时退化为关键词匹配，
保证 search_tools 永远有结果。
"""
from __future__ import annotations

import math

from config import get_settings
from llm import embed
from tools.registry import ToolSpec, searchable_specs

settings = get_settings()

_cache: dict[str, list[float]] = {}
_cache_key: tuple[str, ...] = ()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _ensure_index(specs: list[ToolSpec]) -> bool:
    """构建/刷新工具向量索引。返回是否可用。"""
    global _cache_key
    key = tuple(sorted(s.name for s in specs))
    if key == _cache_key and all(s.name in _cache for s in specs):
        return True
    try:
        for s in specs:
            if s.name not in _cache:
                _cache[s.name] = embed(s.search_text())
        _cache_key = key
        return True
    except Exception:
        return False


def _keyword_search(task: str, specs: list[ToolSpec], top_k: int) -> list[str]:
    """退化路径：按词面重合度排序。"""
    tokens = [t for t in task.lower().replace("_", " ").split() if t]
    scored: list[tuple[float, str]] = []
    for s in specs:
        blob = s.search_text().lower()
        score = sum(1.0 for t in tokens if t in blob)
        score += sum(0.3 for ch in set(task) if ch in blob and "\u4e00" <= ch <= "\u9fff")
        scored.append((score, s.name))
    scored.sort(reverse=True)
    return [name for _, name in scored[:top_k]]


def search_tools(task: str, pack: str | None = None, top_k: int | None = None,
                 min_similarity: float | None = None) -> list[str]:
    """按任务描述召回工具名（降序相关度）。min_similarity 用于过滤低分项，
    防止工具集每回合抖动（tools= 字段一变，前缀缓存整段失效）。
    """
    top_k = top_k or settings.TOOL_SEARCH_TOP_K
    # ⚠️ 必须关键字传参：searchable_specs 的第一个位置参数是 profile，不是 pack。
    specs = searchable_specs(pack=pack)
    if not specs:
        return []
    if len(specs) <= top_k and min_similarity is None:
        return [s.name for s in specs]

    if not _ensure_index(specs):
        return _keyword_search(task, specs, top_k)

    try:
        qvec = embed(task)
    except Exception:
        return _keyword_search(task, specs, top_k)

    scored = [(_cosine(qvec, _cache[s.name]), s.name) for s in specs if s.name in _cache]
    scored.sort(reverse=True)
    if min_similarity is not None:
        scored = [x for x in scored if x[0] >= min_similarity]
    return [name for _, name in scored[:top_k]]


def invalidate() -> None:
    """工具集合变化时调用（运行期加插件用）。"""
    global _cache_key
    _cache_key = ()


def index_available() -> bool:
    """向量索引是否可用（即嵌入后端是否活着）。嵌入挂掉时 `search_tools()`
    会退化成关键词路径 —— 能用，但精度不同。
    """
    specs = searchable_specs()
    if not specs:
        return True
    return _ensure_index(specs)
