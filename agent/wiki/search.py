"""Wiki 检索：给对话中的模型一个"查知识库"的入口。

**和 `retrieve_memory` 分开**（§7.9）：那个返回散装记忆，Wiki 页面是有锚点的结构化
文本（论断挂着 `[memory:<id>]`），混在一起返回模型就分不清哪句可追溯。
嵌入优先，挂了退关键词（同 `toolsearch`）；**但退化必须写在结果里** ——
静默换掉会让模型以为"Wiki 里没有"，而其实只是没搜到。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from wiki import pages as P

#: 检索时忽略的字符（中英文标点、空白）
_SPLIT = re.compile(r"[\s,，。、；;：:！!？?（）()\[\]【】\"'“”‘’/\\|]+")


@dataclass
class Hit:
    path: str
    title: str
    snippet: str
    score: float
    method: str          # embedding | keyword


def _cache_file(root: Path) -> Path:
    return Path(root) / ".embed_cache.json"


def _load_vec_cache(root: Path) -> dict[str, list[float]]:
    f = _cache_file(root)
    if not f.is_file():
        return {}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_vec_cache(root: Path, cache: dict[str, list[float]]) -> None:
    P.write_text(_cache_file(root), json.dumps(cache) + "\n")


def _page_text(page: P.Page) -> str:
    """参与检索的正文：标题 + 别名 + 综述 + 证据。
    **不含 front-matter 的其余字段**（`sources` 里全是 `memory:123`，只会稀释语义）。"""
    parts = [page.title, " ".join(page.aliases), page.summary]
    parts += [e.text for e in page.evidence]
    return "\n".join(x for x in parts if x)


def _load_pages(wiki: Path) -> dict[str, P.Page]:
    out: dict[str, P.Page] = {}
    for f in sorted(wiki.rglob("*.md")):
        rel = f.relative_to(wiki).as_posix()
        if rel.startswith("templates/") or rel in (
                "index.md", "log.md", "overview.md", "QUESTIONS.md"):
            continue
        try:
            out[rel] = P.parse(f.read_text(encoding="utf-8"), rel)
        except (ValueError, OSError):
            continue
    return out


def _keyword(query: str, pages: dict[str, P.Page], top_k: int) -> list[Hit]:
    """退化的关键词路径：按词面重合度打分。返回的 `method` 是 `keyword`，
    **不假装它和语义检索等价**，调用方要把它写进结果里。"""
    terms = [t for t in _SPLIT.split(query.lower()) if t]
    if not terms:
        return []
    hits: list[Hit] = []
    for rel, page in pages.items():
        text = _page_text(page).lower()
        title = (page.title + " " + " ".join(page.aliases)).lower()
        score = 0.0
        for t in terms:
            score += text.count(t)
            score += 3.0 * title.count(t)      # 标题命中权重更高
        if score > 0:
            hits.append(Hit(rel, page.title, _snippet(page, terms), score, "keyword"))
    hits.sort(key=lambda h: (-h.score, h.path))
    return hits[:top_k]


def _snippet(page: P.Page, terms: list[str], width: int = 160) -> str:
    """挑一句最相关的正文当摘要。"""
    body = page.summary.strip()
    if body:
        for line in body.splitlines():
            if any(t in line.lower() for t in terms):
                return line.strip()[:width]
        return body.splitlines()[0].strip()[:width]
    for ev in page.evidence:
        if any(t in ev.text.lower() for t in terms):
            return ev.text[:width]
    return (page.evidence[0].text[:width] if page.evidence else "")


def search(query: str, wiki: Path | None = None, *, top_k: int = 5,
           embed=None) -> tuple[list[Hit], str]:
    """检索 Wiki。返回 `(命中, 用的方法)`；`method` 是 `embedding` / `keyword` /
    `empty`，**调用方要把它告诉模型**。"""
    from config import get_settings
    if wiki is None:
        wiki = Path(get_settings().WIKI_DIR) / "wiki"
    pages = _load_pages(Path(wiki))
    if not pages or not (query or "").strip():
        return [], "empty"

    if embed is None:
        try:
            from llm import embed as _e
            embed = _e
        except Exception:
            embed = None

    if embed is not None:
        try:
            hits = _embedding_search(query, pages, top_k, embed, Path(wiki))
            if hits:
                return hits, "embedding"
        except Exception:                     # noqa: BLE001 —— 后端挂了不该炸检索
            pass

    return _keyword(query, pages, top_k), "keyword"


def _embedding_search(query: str, pages: dict[str, P.Page], top_k: int,
                      embed, root: Path) -> list[Hit]:
    """嵌入检索。页面向量按**内容哈希**缓存在磁盘上 —— 页面没改就不重嵌；
    未命中的页**批量嵌入**（不逐条调用）。"""
    cache = _load_vec_cache(root)
    dirty = False

    texts = {rel: _page_text(page) for rel, page in pages.items()}
    keys = {rel: hashlib.sha1(t.encode("utf-8")).hexdigest()[:20]
            for rel, t in texts.items()}
    missing = [rel for rel in sorted(texts) if keys[rel] not in cache]

    if missing:
        try:
            from llm import embed_many
            vecs = embed_many([texts[rel] for rel in missing])
        except Exception:
            vecs = [embed(texts[rel]) for rel in missing]   # 注入的 embed 走这条
        for rel, vec in zip(missing, vecs):
            cache[keys[rel]] = vec
            dirty = True

    import math

    def cos(a, b):
        d = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return d / (na * nb) if na and nb else 0.0

    qvec = embed(query)
    scored = [(cos(qvec, cache[keys[rel]]), rel, page)
              for rel, page in pages.items()]
    if dirty:
        _save_vec_cache(root, cache)
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [Hit(rel, page.title, _snippet(page, _SPLIT.split(query.lower())), score,
                "embedding")
            for score, rel, page in scored[:top_k]]


def stats(wiki: Path | None = None) -> dict:
    """页面计数，给 `wiki_search` 的"没命中"回复用。"""
    from config import get_settings
    wiki = Path(wiki or Path(get_settings().WIKI_DIR) / "wiki")
    pages = _load_pages(wiki)
    by_type: dict[str, int] = {}
    for page in pages.values():
        by_type[page.type] = by_type.get(page.type, 0) + 1
    return {"total": len(pages), "by_type": by_type}
