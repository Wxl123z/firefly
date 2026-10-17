"""LLM Wiki 构建管线：离线、增量、幂等、可全量重建。

Wiki 是 `f(memory 表, rollup 表)` 的纯函数：`--rebuild` 全量重跑，或只处理
`memory.id > 水位线` 的增量；**重跑同一水位线必须产出逐字节相同的 Wiki**
（证据按锚点去重、`log.md` 按 (操作,目标,水位线) 去重、集合排序后写、LLM 输出按输入哈希缓存）。

⚠️ 挂在 `offline.run()` 里时**必须各自 try/except 隔离**，构建失败不能影响对话或前序步骤。
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from sqlalchemy import func, select

from config import get_settings
from store.db import (Memory, MemoryStatus, Person, Rollup, SessionLocal, get_wiki_state,
                save_wiki_state)
from wiki import pages as P
from wiki.contract import PAGE_TYPES, render_contract
from wiki.lint import dashboard_line

settings = get_settings()
log = logging.getLogger("wiki")

DISPLAY_TYPES = ("discussion", "stated_preference", "inference", "stated_fact",
                 "event")


# ---------------------------------------------------------------- LLM 缓存

def _cache_path(root: Path, name: str) -> Path:
    """LLM 缓存目录。**必须按 `root` 分**，和 `_state_key` 同一个道理：
    否则测试的假 LLM 输出会落进真 Wiki 的缓存并被真构建命中。
    """
    cache = Path(root) / ".llm_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{name}.txt"


def _cache_key(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _cached(root: Path, name: str, payload: str, produce: Callable[[], str]) -> str:
    """按输入哈希缓存 LLM 输出。**幂等的第三道闸门，也是 `--rebuild` 能逐字节重现的关键**：
    重跑时 LLM 措辞一变，产出就不一样了。
    """
    f = _cache_path(root, f"{name}-{_cache_key(payload)}")
    if f.is_file():
        return f.read_text(encoding="utf-8")
    out = produce()
    f.write_text(out, encoding="utf-8", newline="\n")
    return out


# ---------------------------------------------------------------- 主题归并

def _norm(s: str) -> str:
    return "".join(str(s).split()).lower()


@dataclass
class MergeResult:
    mapping: dict[str, str]          # topic → 归并后的规范标题
    method: str                      # exact | embedding | llm | none
    note: str = ""


def resolve_topics(topics: list[str], index: dict, *, root: Path | None = None,
                   embed=None, ask_llm=None) -> MergeResult:
    """把一批主题名归并到页面标题。
    匹配链（§7.4 第 3a 步）三档降级：**精确匹配 → 嵌入相似度 → 一次 LLM 归并**；
    都不可用时兜底为一主题一页，`method` / `note` 写明实际用了哪一档。
    ⚠️ 嵌入档挂掉时不能静默退化，否则会静默产出大量近义重复页而调用方看不出来。
    """
    existing: dict[str, str] = {}          # 归一化标题/别名 → 规范标题
    for rel, meta in sorted((index.get("pages") or {}).items()):
        title = meta.get("title") or ""
        if not title:
            continue
        for name in [title, *(meta.get("aliases") or [])]:
            existing.setdefault(_norm(name), title)

    mapping: dict[str, str] = {}
    pending: list[str] = []
    for t in sorted(set(topics)):          # **排序**：处理顺序影响归并结果，必须确定
        if not t:
            continue
        hit = existing.get(_norm(t))
        if hit:
            mapping[t] = hit
        else:
            pending.append(t)

    if not pending:
        return MergeResult(mapping, "exact")

    # ---- 第二档：嵌入相似度 ----
    if embed is not None:
        try:
            groups = _cluster_embedding(pending, list(existing.values()), embed)
            for canon, members in groups.items():
                for m in members:
                    mapping[m] = canon
            return MergeResult(mapping, "embedding")
        except Exception as exc:            # noqa: BLE001 —— 后端不可用不该炸构建
            log.warning("嵌入归并不可用，退回下一档：%s", exc)

    # ---- 第三档：一次 LLM 归并 ----
    if ask_llm is not None:
        try:
            groups = _cluster_llm(pending, ask_llm, root or Path(settings.WIKI_DIR))
            for canon, members in groups.items():
                for m in members:
                    mapping[m] = canon
            return MergeResult(mapping, "llm")
        except Exception as exc:            # noqa: BLE001
            log.warning("LLM 归并也不可用：%s", exc)

    # ---- 兜底：一主题一页，但**说出来** ----
    for t in pending:
        mapping[t] = t
    return MergeResult(mapping, "none",
                       note=f"⚠️ 归并有没跑成（嵌入和 LLM 都不可用），"
                            f"{len(pending)} 个主题各自成页，可能有近义重复。")


def _cluster_embedding(items: list[str], seeds: list[str], embed) -> dict[str, list[str]]:
    """贪心聚类。**按输入顺序**（调用方已排序）→ 结果确定；嵌入一次批量发过去。"""
    import math

    def cos(a, b):
        d = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return d / (na * nb) if na and nb else 0.0

    uniq_seeds = sorted(set(seeds))
    todo = uniq_seeds + list(items)
    try:
        from llm import embed_many
        vecs = dict(zip(todo, embed_many(todo)))
    except Exception:
        vecs = {t: embed(t) for t in todo}      # 退回逐条（注入的 embed 走这条）

    seed_vecs = [(s, vecs[s]) for s in uniq_seeds]
    groups: dict[str, list[str]] = {}
    reps: list[tuple[str, list]] = []
    for it in items:
        vec = vecs[it]
        hit = next((name for name, sv in seed_vecs
                    if cos(vec, sv) >= settings.WIKI_MERGE_SIMILARITY), None)
        if hit is None:
            hit = next((name for name, rv in reps
                        if cos(vec, rv) >= settings.WIKI_MERGE_SIMILARITY), None)
        if hit is None:
            reps.append((it, vec))
            groups.setdefault(it, []).append(it)
        else:
            groups.setdefault(hit, []).append(it)
    return groups


#: 改 `_LLM_MERGE_PROMPT` 时**必须同时改这个版本号** —— 缓存键含它，
#: 否则改完提示词重跑会命中的还是旧结果。
_MERGE_PROMPT_VERSION = 2

_LLM_MERGE_PROMPT = """下面是从**同一个人**多天的对话记忆里抽出的主题名。

关键背景：**同一个话题在不同天会被起不同的名字** —— 这是抽取时的措辞差异，
不是不同的话题。比如今天写成「工作」，明天写成「工作变动」。

下面这些是**同一件事**，必须归到一组（这是真实数据里的例子）：
- 「工作」「工作变动」
- 「对话概览」「对话概况」「对话状态」「对话记忆」
- 「心理状态」「情绪状态」「身心健康」
- 「阅读」「阅读计划」「阅读记录」
- 「待办事项」「生活待办」

列表里**没给出例子的**其它近义对，你也要自己判断并归并。

**但不要把只是相关的东西合并**：「宠物」和「明日行程与宠物照料」不是一回事
（后者是一次具体行程），「生活琐事」和「待办事项」也不是。

主题列表：
{items}

严格按下面的 JSON 输出，不要有别的内容。每组的第一个元素作为该组的规范名，
**规范名要从该组成员里选，不要自己造新词**：
{{"groups": [["规范名", "同义名1", "同义名2"], ...]}}"""


def _cluster_llm(items: list[str], ask_llm, root: Path) -> dict[str, list[str]]:
    payload = json.dumps({"v": _MERGE_PROMPT_VERSION, "items": items},
                         ensure_ascii=False)

    def produce() -> str:
        return ask_llm(_LLM_MERGE_PROMPT.format(
            items="\n".join(f"- {x}" for x in items)))

    raw = _cached(root, "merge", payload, produce)
    data = json.loads(_strip_fence(raw))
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    for grp in data.get("groups") or []:
        members = [str(x) for x in grp if str(x) in items and str(x) not in seen]
        if not members:
            continue
        seen.update(members)
        groups[members[0]] = members
    missing = [x for x in items if x not in seen]      # LLM 漏掉的不能丢
    for x in missing:
        groups[x] = [x]
    return groups


def _strip_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        t = t.rsplit("```", 1)[0]
    return t.strip()


def _state_key(root: Path) -> str:
    """水位线的键。**按 root 区分**，不是固定 `"default"` —— 水位线存在 DB 里，
    各 root 共用一行的话，在全新空目录下构建会继承真 Wiki 的水位线（"新记忆 0 条"、
    不产概念页）。配置里的 root 用 `"default"`，其它按路径取哈希。
    """
    try:
        resolved = Path(root).resolve()
        if resolved == Path(settings.WIKI_DIR).resolve():
            return "default"
        return "root:" + hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    except OSError:
        return "default"


# ---------------------------------------------------------------- 构建

@dataclass
class BuildResult:
    ok: bool = True
    rebuilt: bool = False
    new_memories: int = 0
    pages_written: int = 0
    page_count: int = 0
    llm_calls: int = 0
    merge_method: str = "exact"
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [f"{'重建' if self.rebuilt else '增量'}："
                f"新记忆 {self.new_memories} 条 · 写页 {self.pages_written} 个 · "
                f"共 {self.page_count} 页 · LLM {self.llm_calls} 次 · "
                f"归并={self.merge_method}"]
        bits += self.notes
        return " | ".join(bits)


def build(*, root: Path | None = None, rebuild: bool = False,
          use_llm: bool | None = None,
          ask_llm: Callable[[str], str] | None = None,
          embed: Callable | None = None) -> BuildResult:
    """构建（或重建）Wiki。

    - `rebuild=True`：清空 `wiki/` 与水印，从 0 全量重跑
    - `use_llm=False`：不调 LLM（综述留占位）—— 离线测试用
    - `ask_llm` / `embed`：可注入，测试时可换成假的
    """
    root = Path(root or settings.WIKI_DIR)
    wiki = root / "wiki"
    use_llm = settings.WIKI_USE_LLM if use_llm is None else use_llm
    result = BuildResult(rebuilt=rebuild)
    skey = _state_key(root)

    (root / "raw" / "days").mkdir(parents=True, exist_ok=True)
    wiki.mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(parents=True, exist_ok=True)

    if rebuild:
        _wipe_wiki(wiki)
        # force=True 不能省：默认有单调性保护，不加的话 `max(0, 27)` 会把重置挡掉。
        save_wiki_state(skey, memory_watermark=0, message_watermark=0,
                        page_count=0, force=True)

    if ask_llm is None and use_llm:
        ask_llm = _default_ask_llm
    if ask_llm is None:
        use_llm = False
    if embed is None:
        embed = _default_embed

    state = get_wiki_state(skey)
    index = P.load_index(wiki)
    if rebuild or not index.get("pages"):
        index = P.rebuild_index(wiki)

    # ---- 1. 每日总结：一天一个来源页 ----
    rollups = _load_rollups()
    watermark_msg = state["message_watermark"]
    with SessionLocal() as s:
        total_mem = s.execute(select(func.count()).select_from(Memory)).scalar() or 0

    # ---- 2. 增量记忆 ----
    memories = _load_memories(state["memory_watermark"])
    result.new_memories = len(memories)
    display = [m for m in memories if _confident(m)]

    # ⚠️ **开放问题是"状态"，不是"本次新处理了什么"。**
    #    必须取库里当前所有低置信、未澄清的记忆，按增量取会破坏幂等。
    low = _load_open_questions()

    # ---- 3. 路由 ----
    merge = resolve_topics([_topic_of(m) for m in display], index, root=root,
                           embed=embed, ask_llm=ask_llm if use_llm else None)
    result.merge_method = merge.method
    if merge.note:
        result.notes.append(merge.note)

    touched: dict[str, P.Page] = {}

    # 3a. 主题 → concepts/
    by_page: dict[str, list] = {}
    for m in display:
        by_page.setdefault(merge.mapping.get(_topic_of(m), _topic_of(m)), []).append(m)
    for canon in sorted(by_page):
        # ⚠️ 这里传的必须是**页面路径**，不是标题 —— 索引按路径存，
        #    传标题会让 `_get_page` 永远返回 None，增量构建整页覆盖已有证据。
        path = P.page_path("concept", canon)
        page = _get_page(wiki, index, path) or \
            P.Page(path=path, title=canon, type="concept")
        _add_evidence(page, by_page[canon], _topic_of)
        # 归并进来的主题名进 aliases —— 下次就能精确命中，不用再靠嵌入
        for alias in sorted({_topic_of(m) for m in by_page[canon]}):
            if alias != page.title and alias not in page.aliases:
                page.aliases.append(alias)
        page.updated = _latest_date(by_page[canon], page.updated)
        page.confidence = _confidence_of(by_page[canon])
        touched[page.path] = page

    # 3b. 日期 → sources/
    # 关联要从**全部概念页**推出来，不是只从本次处理的那批 ——
    # 只算本次的话，重跑时推出来是空的，会把已有页的关联覆盖掉，破坏幂等。
    day_to_concepts = _day_concept_map(wiki, index, touched)
    for day, r in sorted(rollups.items()):
        rel = P.page_path("source", day)
        page = _get_page(wiki, index, rel) or \
            P.Page(path=rel, title=day, type="source")
        page.sources = sorted(set(page.sources) | {f"rollup:{day}"})
        if r["summary"]:
            page.summary = r["summary"].strip()
        page.updated = max(page.updated, day)
        # 来源页 → 引用了这一天的概念页
        page.related = sorted(day_to_concepts.get(day, set()))
        touched[page.path] = page

    # 概念页 → 它的证据散布在哪些来源日（覆盖式赋值，保证确定性）
    for canon in sorted(by_page):
        page = touched[P.page_path("concept", canon)]
        days = sorted({ev.date for ev in page.evidence if ev.date})
        page.related = [P.page_path("source", d) for d in days]

    # 3c. 人物 → entities/（Person 表为空时不产页）
    for person in _load_persons():
        rel = P.page_path("entity", person["name"])
        page = _get_page(wiki, index, rel) or \
            P.Page(path=rel, title=person["name"], type="entity")
        if person["relationship"]:
            page.aliases = sorted(set(page.aliases) | {person["relationship"]})
        if person["notes"] and not page.summary:
            page.summary = person["notes"].strip()
        touched[page.path] = page

    # ---- 4. 综述：证据够多才值得花一次 LLM ----
    for rel in sorted(touched):
        page = touched[rel]
        if not use_llm or page.type == "source":
            continue
        if len(page.evidence) < settings.WIKI_SUMMARY_MIN_EVIDENCE:
            continue
        payload = _summary_payload(page)
        before = _cache_hits(root, "summary", payload)
        page.summary = _cached(root, "summary", payload,
                               lambda p=page: _ask_summary(ask_llm, p))
        if not before:
            result.llm_calls += 1

    # ---- 5. synthesis：跨来源、证据够多 ----
    # ⚠️ 必须排在**综述之后、写页之前**：排在之前会拷到空综述（"空页"），
    #    排在之后没人链它（"孤儿页"）。
    _add_synthesis(touched)

    # ---- 6. 写页 ----
    for rel in sorted(touched):
        P.write_page(wiki, touched[rel])
        index.setdefault("pages", {})[rel] = _index_entry(touched[rel])
        result.pages_written += 1

    # ---- 7. QUESTIONS.md ----
    # **总是渲染**（数据源是库里的当前状态，不是本次增量）。
    _write_questions(wiki, low)

    # ---- 8. index.md / overview.md / log.md ----
    index = P.rebuild_index(wiki) if rebuild else index
    P.save_index(wiki, index)
    new_watermark = max(state["memory_watermark"], _max_id(memories))
    _write_index(wiki, index)
    _write_overview(wiki, index, new_watermark, total_mem=total_mem, low=low)
    # 日志戳用**新**水位线，且不带"本次处理了几条"（否则空跑和实跑写出的戳不同）。
    _append_log(wiki, "build", new_watermark)
    P.write_text(root / "WIKI.md", render_contract())
    P.write_text(root / "README.md", _readme())
    _write_scripts(root)

    # ---- 9. 水位线 ----
    page_count = len(index.get("pages") or {})
    save_wiki_state(skey, memory_watermark=new_watermark,
                    message_watermark=watermark_msg, page_count=page_count)
    result.page_count = page_count
    return result


# ---------------------------------------------------------------- 内部

def _wipe_wiki(wiki: Path) -> None:
    """清空 `wiki/` 但**保留 LLM 缓存**（缓存在 `..` 下的 `.llm_cache/`），
    这样 `--rebuild` 才能逐字节重现。
    """
    if not wiki.exists():
        return
    for f in sorted(wiki.rglob("*"), reverse=True):
        if f.is_file():
            f.unlink(missing_ok=True)
        elif f.is_dir():
            f.rmdir()


def _default_ask_llm(prompt: str) -> str:
    from llm import chat
    return chat([{"role": "user", "content": prompt}], temperature=0.2)


def _default_embed(text: str):
    from llm import embed
    return embed(text)


def _cache_hits(root: Path, name: str, payload: str) -> bool:
    return _cache_path(root, f"{name}-{_cache_key(payload)}").is_file()


def _topic_of(m) -> str:
    return (m.topic or "").strip() or "未分类"


def _confident(m) -> bool:
    """低置信记忆**不进正文页**，只进 `QUESTIONS.md`（§7.3）：正文只放有把握的内容。
    """
    if m.status != MemoryStatus.current:
        return False
    return (m.confidence if m.confidence is not None else 1.0) >= settings.MEMORY_MIN_CONFIDENCE


def _confidence_of(items) -> str:
    kinds = {"stated" if (m.confidence or 0) >= 0.9 else "inferred" for m in items}
    if len(kinds) > 1:
        return "mixed"
    return kinds.pop() if kinds else "stated"


#: 上游曾经把「没有日期」写成过这些**字符串**，库里已有这类历史数据；
#: 不认出来的话页面上会出现 `- None · [memory:16] …`。
_DIRTY_DATES = {"none", "null", "nan", "nat", "-"}


def _clean_date(value) -> str:
    """把日期字符串洗干净；识别不出就返回空串（明确的"不知道"）。"""
    s = ("" if value is None else str(value)).strip()
    if not s or s.lower() in _DIRTY_DATES:
        return ""
    return s.split("~")[-1].strip()


def _evidence_of(m) -> P.Evidence:
    """记忆 → 一条证据。
    ⚠️ **日期字段必须保持"就是一个日期"**，不能带后缀 —— 证据行格式是
    `- <日期> · [锚点] <正文>`，后缀会让 `pages.parse()` 匹配不上，整页锚点都读不出来。
    """
    date = _clean_date(m.date_range)
    note = ""
    if not date:
        date = m.time.strftime("%Y-%m-%d") if m.time else ""
    if not date:
        # 实在没有日期就用 created_at —— 但**标出来**，那是"入库时间"不是"发生时间"
        date = m.created_at.strftime("%Y-%m-%d") if m.created_at else ""
        if date:
            note = "（日期取的是入库时间）"
    text = note + (m.fact or "").strip()
    if m.quote and m.quote.strip() and m.quote.strip() != (m.fact or "").strip():
        text += f"（原话：{m.quote.strip()}）"
    return P.Evidence(ref=f"memory:{m.id}", date=date, text=text)


def _add_evidence(page: P.Page, items, topic_of) -> None:
    have = {e.ref for e in page.evidence}
    for m in items:
        ev = _evidence_of(m)
        if ev.ref not in have:
            page.evidence.append(ev)
            have.add(ev.ref)
    # front-matter 的 `sources:` 要列出这一页依赖的锚点，lint 第 8 项拿它去 DB 里查。
    page.sources = sorted({e.ref for e in page.evidence}, key=P.ref_sort_key)


def _latest_date(items, fallback: str) -> str:
    dates = [(_evidence_of(m).date or "") for m in items]
    cand = [d for d in [*dates, fallback] if d]
    return max(cand) if cand else fallback


def _load_memories(after_id: int) -> list:
    with SessionLocal() as s:
        q = (select(Memory).where(Memory.id > after_id)
             .order_by(Memory.id))
        return list(s.execute(q).scalars())


def _load_open_questions() -> list:
    """库里**当前**仍然低置信、且还没被推翻的记忆，`QUESTIONS.md` 的数据源。
    **它是状态不是事件**：低置信记忆只要还在库里就该一直挂着，按增量算下次重跑
    它会消失，而且会破坏幂等。只取 `MemoryType` 里适合当"问题"的那几类，
    `summary` 类（对话概况）不算。
    """
    from store.db import MemoryType
    skip = {MemoryType.summary.value if hasattr(MemoryType.summary, "value")
            else str(MemoryType.summary)}
    with SessionLocal() as s:
        rows = list(s.execute(select(Memory)
                              .where(Memory.status == MemoryStatus.current)
                              .order_by(Memory.id)).scalars())
    return [m for m in rows
            if not _confident(m) and str(m.type).split(".")[-1] not in skip]


def _load_rollups() -> dict[str, dict]:
    with SessionLocal() as s:
        return {r.date: {"summary": r.summary or "",
                         "watermark": r.watermark_message_id}
                for r in s.execute(select(Rollup).order_by(Rollup.date)).scalars()}


def _load_persons() -> list[dict]:
    with SessionLocal() as s:
        return [{"name": p.name, "relationship": p.relationship or "",
                 "notes": p.notes or ""}
                for p in s.execute(select(Person).order_by(Person.id)).scalars()]


def _max_id(items) -> int:
    return max((m.id for m in items), default=0)


def _get_page(wiki: Path, index: dict, rel: str):
    """已存在的页就读回来（**保留旧的综述和证据**），否则 None。
    ⚠️ `rel` 必须是**页面路径**（`concepts/工作.md`），不是标题 —— 传标题永远返回 None。
    """
    if rel in (index.get("pages") or {}) and (wiki / rel).is_file():
        try:
            return P.read_page(wiki, rel)
        except (ValueError, OSError):
            pass
    return None


def _index_entry(page: P.Page) -> dict:
    return {"title": page.title, "type": page.type,
            "aliases": sorted(set(page.aliases)),
            "sources": sorted(set(page.sources), key=P.ref_sort_key),
            "updated": page.updated}


def _summary_payload(page: P.Page) -> str:
    return json.dumps({"title": page.title, "evidence": [e.render() for e in
                                                         sorted(page.evidence, key=P.Evidence.key)]},
                      ensure_ascii=False)


_SUMMARY_PROMPT = """你在维护一个个人知识库。请为下面这一页写「综述」段：3-5 句，
把这批证据说的是什么讲清楚。

**硬性要求：**
- 只写证据里有的。**没说的不要补。**
- 每个论断后面要能对上证据里的锚点。
- 用第三人称叙述（"对方"），不要用"我"。
- 直接输出正文，不要标题、不要 front-matter、不要解释你在做什么。

页面标题：{title}

证据：
{evidence}"""


def _ask_summary(ask_llm, page: P.Page) -> str:
    return ask_llm(_SUMMARY_PROMPT.format(
        title=page.title,
        evidence="\n".join(e.render() for e in sorted(page.evidence, key=P.Evidence.key))
    )).strip()


def _write_questions(wiki: Path, low: list) -> None:
    lines = ["---", "title: QUESTIONS", "type: meta",
             "aliases: []", "sources: []",
             f"updated: {datetime.now():%Y-%m-%d}", "confidence: stated",
             "graph-excluded: true", "---", "", "# 开放问题", "",
             "低置信或没说清的东西记在这里，**不写进正文页** ——",
             "正文只放有把握的内容，读的人才分得清。", ""]
    if low:
        lines += ["## 待澄清（置信度低于阈值）", ""]
        for m in sorted(low, key=lambda x: x.id):
            lines.append(f"- [memory:{m.id}] {m.fact}"
                         f"（置信度 {m.confidence if m.confidence is not None else '?'}）")
    else:
        lines += ["（暂无。）"]
    lines.append("")
    P.write_text(wiki / "QUESTIONS.md", "\n".join(lines))


def _day_concept_map(wiki: Path, index: dict,
                     touched: dict[str, P.Page]) -> dict[str, set[str]]:
    """`日期 → 引用这一天的概念页路径`。

    从**全部概念页**推（本次改动的 + 磁盘上已有的）：只算本次的话，重跑时推出来
    是空的，会覆盖掉已有页的关联。
    """
    pages: dict[str, P.Page] = {}
    for rel, meta in sorted((index.get("pages") or {}).items()):
        if (meta.get("type") or "") != "concept":
            continue
        if rel in touched:
            pages[rel] = touched[rel]
            continue
        got = _get_page(wiki, index, rel)
        if got is not None:
            pages[rel] = got
    # 本次新产出的概念页可能还没进 index
    for rel, page in touched.items():
        if page.type == "concept":
            pages[rel] = page

    out: dict[str, set[str]] = {}
    for rel, page in pages.items():
        for ev in page.evidence:
            if ev.date:
                out.setdefault(ev.date, set()).add(rel)
    return out


def _add_synthesis(touched: dict[str, P.Page]) -> None:
    """跨来源合成：证据够多**且**跨够多天才生成。就地改 `touched`，
    并给概念页加一条指向合成页的关联（不然合成页是孤儿）。"""
    for rel in sorted(list(touched)):
        page = touched[rel]
        if page.type != "concept":
            continue
        days = {e.date for e in page.evidence if e.date}
        if (len(page.evidence) < settings.WIKI_SYNTHESIS_MIN_EVIDENCE
                or len(days) < settings.WIKI_SYNTHESIS_MIN_DAYS):
            continue
        title = f"{page.title}（合成）"
        syn = P.Page(path=P.page_path("synthesis", title), title=title,
                     type="synthesis", updated=page.updated,
                     confidence=page.confidence, summary=page.summary,
                     related=[rel], aliases=[])
        # 证据要带过去 —— 合成页的论断同样需要锚点，不带会被 lint 报成"空页"。
        syn.evidence = list(page.evidence)
        syn.sources = sorted({e.ref for e in page.evidence}, key=P.ref_sort_key)
        touched[syn.path] = syn
        # 概念页 ← → 合成页，双向
        if syn.path not in page.related:
            page.related = sorted(set(page.related) | {syn.path})


def _write_index(wiki: Path, index: dict) -> None:
    by_type: dict[str, list[tuple[str, str]]] = {}
    for rel, meta in sorted((index.get("pages") or {}).items()):
        by_type.setdefault(meta.get("type") or "?", []).append(
            (meta.get("title") or rel, rel))
    lines = ["# 索引", "",
             "**第一个该读的文件。** 按类型分组，点进去是具体页面。", ""]
    for t in sorted(by_type):
        lines += [f"## {t}（{len(by_type[t])}）", ""]
        for title, rel in sorted(by_type[t]):
            lines.append(f"- [[{rel[:-3]}]] · {title}")
        lines.append("")
    P.write_text(wiki / "index.md", "\n".join(lines))


def _write_overview(wiki: Path, index: dict, watermark: int, *,
                    total_mem: int, low: list) -> None:
    """总览 + Health Dashboard。
    ⚠️ "待处理记忆"是 `总数 - 水位线`，**不是"本次处理了几条"**：写后者的话
    同一水位线重跑这个数就会变，幂等就破了。
    """
    by_type: dict[str, int] = {}
    for meta in (index.get("pages") or {}).values():
        by_type[meta.get("type") or "?"] = by_type.get(meta.get("type") or "?", 0) + 1
    with SessionLocal() as s:
        total_rollup = s.execute(select(func.count()).select_from(Rollup)).scalar() or 0
    counts = " · ".join(f"{k} {v}" for k, v in sorted(by_type.items()))
    pending = max(total_mem - watermark, 0)
    # ⚠️ 传的是 `wiki`（页面那一层），不是 `wiki.parent` ——
    #    `dashboard_line` 自己会取 `.parent` 去找 `outputs/`。
    lint_line = dashboard_line(wiki)
    lines = [
        "---", "title: 总览", "type: meta", "aliases: []", "sources: []",
        f"updated: {datetime.now():%Y-%m-%d}", "confidence: stated",
        "---", "", "# 总览", "",
        "## Health Dashboard", "",
        f"- 页面：{counts or '（空）'}（共 {len(index.get('pages') or {})}）",
        f"- 待处理记忆：{pending}（水位线 memory:{watermark} / 共 {total_mem}）",
        f"- 开放问题：{len(low)}",
        f"- 每日总结：{total_rollup}",
        lint_line, "",
        "## 这个知识库是怎么来的", "",
        "它是 `memory` 表和 `rollup` 表的**派生物**，不是第二份真相。",
        "坏了不用修，`python -m wiki.build --rebuild` 直接重建。",
        "",
    ]
    P.write_text(wiki / "overview.md", "\n".join(lines))


def _append_log(wiki: Path, op: str, watermark: int) -> None:
    """仅追加。**按戳去重** —— 幂等的一道闸门。戳里只有 `(操作, 水位线)`，
    **不含"本次处理了几条"**（否则空跑和实跑写出的戳不同）。
    """
    f = wiki / "log.md"
    header = ["---", "title: 操作日志", "type: meta", "aliases: []", "sources: []",
              f"updated: {datetime.now():%Y-%m-%d}", "confidence: stated",
              "graph-excluded: true", "---", "", "# 操作日志", "",
              "仅追加，永不重写。`graph-excluded: true` 把它排除在关系图之外。", ""]
    body = f.read_text(encoding="utf-8").splitlines() if f.is_file() else header
    stamp = f"- {op} · 水位线 {watermark}"
    if stamp in body:
        return
    body.append(stamp)
    P.write_text(f, "\n".join(body) + "\n")


def _write_scripts(root: Path) -> None:
    """生成 `scripts/` 里的薄包装。**逻辑不在那里**。"""
    from wiki.shims import render_scripts
    agent_dir = str(Path(__file__).resolve().parent.parent)
    for name, text in sorted(render_scripts(agent_dir).items()):
        P.write_text(root / "scripts" / name, text)


def _readme() -> str:
    return """# data/wiki · LLM Wiki

把 `memory` 表里散装的条目，整理成一套**人可读、机器可检索、可重建**的 markdown 知识库。

- `raw/` —— 原始数据，**只读**。LLM 不许改。
- `wiki/` —— LLM 完全拥有。从 `wiki/index.md` 开始读。
- `outputs/` —— lint 报告、查询答案。
- `WIKI.md` —— 行为契约，由 `agent/wiki/contract.py` 自动生成，不要手改。

## 重建

它是 `f(memory 表, rollup 表)` 的纯函数。坏了不用修：

```bash
python -m wiki.build --rebuild
```
"""
