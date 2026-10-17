"""记忆系统：Milvus 双集合 + SQLAlchemy 结构化副本。

- companion_memory：已整合的长期记忆（向量侧只存 fact 文本）
- message_index：消息向量索引（未整合的近期原始对话，双源检索第二源）
- 置信度低于 MEMORY_MIN_CONFIDENCE 的条目只留 SQLite（待审核池），不进向量库
- inference 类条目靠 corroborate() 累积佐证，达阈值提升置信度
- 检索统一用原始向量算真实余弦（Milvus 各版本 COSINE distance 语义不一致）
"""
import math
from datetime import datetime

from pymilvus import MilvusClient, DataType

from config import get_settings
from store.db import Memory, MemoryStatus, MemoryType, SessionLocal, save
from llm import embed, embed_many

settings = get_settings()

# 阈值标定（bge-m3 实测）：应合并的同义复述 ≥0.94，不该合并的 ≤0.74 → 取 0.85
UPDATE_MIN_SIMILARITY = 0.85  # 真实余弦相似度阈值：≥ 视为同一记忆
TRUSTED_CONFIDENCE = 0.8     # 佐证达标后的置信度

_milvus: MilvusClient | None = None


def get_milvus() -> MilvusClient:
    global _milvus
    if _milvus is None:
        _milvus = MilvusClient(settings.MILVUS_URI)
        _ensure_collection(_milvus, settings.COLLECTION_NAME)
        _ensure_collection(_milvus, settings.MESSAGE_INDEX_COLLECTION, is_message_index=True)
        _milvus.load_collection(settings.COLLECTION_NAME)
        _milvus.load_collection(settings.MESSAGE_INDEX_COLLECTION)
    return _milvus


def _ensure_collection(client: MilvusClient, name: str, is_message_index: bool = False) -> None:
    if client.has_collection(name):
        return
    schema = client.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.EMBED_DIM)
    schema.add_field("content", DataType.VARCHAR, max_length=2048)
    schema.add_field("created_at", DataType.VARCHAR, max_length=64)
    schema.add_field("importance", DataType.INT64)
    if is_message_index:
        schema.add_field("message_id", DataType.INT64)
        schema.add_field("reviewed", DataType.INT64)
    client.create_collection(name, schema=schema)
    index = client.prepare_index_params()
    index.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE",
                    params={"M": 16, "efConstruction": 64})
    client.create_index(name, index)


# ---------- 长期记忆写入（schema v4） ----------

def store_memory(
    fact: str,
    type_: MemoryType = MemoryType.stated_fact,
    *,
    abstract: str | None = None,
    quote: str | None = None,
    evidence: list | None = None,
    confidence: float | None = 1.0,
    support_count: int | None = None,
    topic: str | None = None,
    date_range: str | None = None,
    source_ref: str | None = None,
    about_person_id: int | None = None,
    status: MemoryStatus = MemoryStatus.current,
    time=None,
    importance: int = 1,
) -> int:
    """写入记忆：SQLite 全量存；置信度达标者同步进向量库。

    注意 support_count 只允许 inference 类型携带（schema CheckConstraint 约束）。
    """
    mem_id = save(Memory(
        type=type_, status=status, fact=fact, abstract=abstract, quote=quote,
        evidence=evidence or [], confidence=confidence, support_count=support_count,
        topic=topic, date_range=date_range, source_ref=source_ref,
        about_person_id=about_person_id, time=time,
    ))
    if confidence is None or confidence >= settings.MEMORY_MIN_CONFIDENCE:
        _index_memory(fact, importance)
    return mem_id


def _index_memory(text: str, importance: int = 1) -> None:
    client = get_milvus()
    client.insert(settings.COLLECTION_NAME, [{
        "vector": embed(text),
        "content": text,
        "created_at": datetime.now().isoformat(),
        "importance": importance,
    }])


def find_similar_memory(content: str) -> dict | None:
    """语义查找已入库的相似记忆（返回 {id, content, similarity} 或 None）。"""
    hits = search_memory_full(content, top_k=1)
    if hits and hits[0]["similarity"] >= UPDATE_MIN_SIMILARITY:
        return hits[0]
    return None


def _normalize(text: str) -> str:
    return "".join(text.split())


def find_row_by_fact(fact: str) -> int | None:
    """SQLite 精确匹配（归一化空白）→ Memory.id 或 None。

    低置信度条目不进向量库，只能靠这里找回。
    """
    key = _normalize(fact)
    with SessionLocal() as s:
        rows = (s.query(Memory.id, Memory.fact)
                .order_by(Memory.id.desc()).limit(500).all())
        for row_id, row_fact in rows:
            if _normalize(row_fact or "") == key:
                return row_id
    return None


def corroborate(
    fact: str,
    *,
    type_: MemoryType = MemoryType.stated_fact,
    abstract: str | None = None,
    quote: str | None = None,
    evidence: list | None = None,
    confidence: float | None = 1.0,
    topic: str | None = None,
    date_range: str | None = None,
    source_ref: str | None = None,
    time=None,
) -> str:
    """写入或佐证：已有条目 → 累积佐证（inference 达阈值升置信度并晋升进向量库）。

    查找顺序：① SQLite 精确匹配（可覆盖低置信度条目）→ ② 向量相似 → ③ 新建。
    返回 "created" | "corroborated"。
    """
    row_id = find_row_by_fact(fact)
    via_vector = False
    if row_id is None:
        similar = find_similar_memory(fact)
        if similar is not None:
            row_id = find_row_by_fact(similar["content"])
            via_vector = True

    if row_id is None:
        store_memory(fact, type_,
                     abstract=abstract, quote=quote, evidence=evidence,
                     confidence=confidence,
                     support_count=1 if type_ == MemoryType.inference else None,
                     topic=topic, date_range=date_range, source_ref=source_ref, time=time)
        return "created"

    with SessionLocal() as s:
        row = s.get(Memory, row_id)
        if row is None:
            return "corroborated"
        if evidence:
            row.evidence = (row.evidence or []) + evidence
        if row.type == MemoryType.inference:
            row.support_count = (row.support_count or 1) + 1
            if row.support_count >= settings.INFERENCE_SUPPORT_THRESHOLD:
                row.confidence = max(row.confidence or 0.0, TRUSTED_CONFIDENCE)
        fact_text = row.fact
        conf = row.confidence
        s.commit()

    # 佐证后置信度达标 → 晋升进向量库（可被检索）；已在向量库中则跳过
    if (not via_vector) and conf is not None and conf >= settings.MEMORY_MIN_CONFIDENCE:
        _index_memory(fact_text)
    return "corroborated"


def clear_memories() -> None:
    """清空全部记忆（开发/测试用）。"""
    client = get_milvus()
    client.delete(settings.COLLECTION_NAME, filter="id >= 0")
    client.delete(settings.MESSAGE_INDEX_COLLECTION, filter="id >= 0")
    with SessionLocal() as s:
        s.query(Memory).delete()
        s.commit()


# ---------- 检索 ----------

def search_memory_full(query: str, top_k: int = 5) -> list[dict]:
    """记忆库检索，返回 {id, content, similarity} 降序（真实余弦）。"""
    client = get_milvus()
    vec = embed(query)
    res = client.search(
        settings.COLLECTION_NAME, data=[vec], limit=top_k,
        output_fields=["content", "vector"],
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    )
    if not res:
        return []
    hits = []
    for h in res[0]:
        entity = h["entity"]
        hits.append({"id": h["id"], "content": entity["content"],
                     "similarity": _cosine(vec, entity["vector"])})
    hits.sort(key=lambda x: x["similarity"], reverse=True)
    return hits


def recall_memory_dual(query: str, top_k: int = 5) -> list[dict]:
    """双源检索：① 已整合记忆 ② 未整合近期原始对话。返回 [{content, source, similarity}]。"""
    client = get_milvus()
    vec = embed(query)
    results: list[dict] = []

    for coll, source, extra_filter in (
        (settings.COLLECTION_NAME, "已整理记忆", ""),
        (settings.MESSAGE_INDEX_COLLECTION, "近期原始对话", "reviewed == 0"),
    ):
        kwargs = dict(
            collection_name=coll, data=[vec], limit=top_k,
            output_fields=["content", "vector"],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )
        if extra_filter:
            kwargs["filter"] = extra_filter
        try:
            res = client.search(**kwargs)
        except Exception:
            continue
        if not res:
            continue
        for h in res[0]:
            entity = h["entity"]
            results.append({"content": entity["content"], "source": source,
                            "similarity": _cosine(vec, entity["vector"])})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[: top_k * 2]


def recall_memory(query: str, top_k: int = 5) -> list[str]:
    """兼容旧接口：仅记忆库检索，返回内容列表。"""
    return [h["content"] for h in search_memory_full(query, top_k=top_k)]


# ---------- 消息索引（双源检索第二源） ----------

#: 索引失败原因（最近若干次）。失败不静默：调用方可经 `index_failures()` 读出并显示。
_INDEX_FAILURES: list[str] = []


def index_failures() -> list[str]:
    """最近若干次索引失败的原因（空 = 一切正常）。"""
    return list(_INDEX_FAILURES)


def reset_index_failures() -> None:
    _INDEX_FAILURES.clear()


def index_message(message_id: int, content: str, reviewed: int = 0) -> None:
    """消息入库时同步建立向量索引（原始对话可被检索，直到整合后删除索引行）。"""
    client = get_milvus()
    client.insert(settings.MESSAGE_INDEX_COLLECTION, [{
        "vector": embed(content),
        "content": content,
        "created_at": datetime.now().isoformat(),
        "importance": 0,
        "message_id": message_id,
        "reviewed": reviewed,
    }])


def count_indexed_messages() -> int:
    """`message_index` 现有行数；取统计失败时返回 -1。"""
    try:
        stats = get_milvus().get_collection_stats(settings.MESSAGE_INDEX_COLLECTION)
        return int(stats.get("row_count") or 0)
    except Exception:
        return -1


def reindex_messages(*, batch: int = 32, verbose: bool = True,
                     limit: int | None = None) -> dict:
    """把 `message_index` 按 DB 里的消息**重建**一遍（索引没建全或换了嵌入模型时用）。

    **只索引"还没整合"的消息**（review_count 为 0）：已整合的记忆索引行会被删除，
    不该再作为"近期原始对话"源，且必须排除不参与记忆的会话（编码会话）。

    幂等：先按 `message_id` 删掉已有行再重插。一次嵌一批（`embed_many`）。
    """
    from sqlalchemy import or_, select
    from store.db import Message, SessionLocal, _memory_conversation_ids

    with SessionLocal() as s:
        rows = list(s.execute(
            select(Message.id, Message.content, Message.review_count,
                   Message.kind)
            .where(Message.kind == "chat",
                   # 没整合过的才建索引（整合过的索了也会被删）
                   or_(Message.review_count.is_(None), Message.review_count == 0),
                   # 必须排除不参与记忆的会话（编码会话），否则重建会把代码和报错
                   # 塞进索引，陪伴那边就捞得到它们了。
                   Message.conversation_id.in_(_memory_conversation_ids()))
            .order_by(Message.id)).all())
    if limit:
        rows = rows[-limit:]
    total = len(rows)
    if verbose:
        print(f"  待索引消息 {total} 条")

    client = get_milvus()
    done = 0
    for i in range(0, total, batch):
        chunk = rows[i:i + batch]
        ids = [r[0] for r in chunk]
        try:
            client.delete(settings.MESSAGE_INDEX_COLLECTION,
                          filter=f"message_id in {ids}")
        except Exception:
            pass                       # 没建过的 collection 删不了，正常
        vecs = embed_many([r[1] for r in chunk])
        client.insert(settings.MESSAGE_INDEX_COLLECTION, [
            {"vector": v, "content": r[1],
             "created_at": datetime.now().isoformat(),
             "importance": 0, "message_id": r[0],
             "reviewed": r[2] or 0}
            for r, v in zip(chunk, vecs)])
        done += len(chunk)
        if verbose and (i // batch) % 20 == 0:
            print(f"    {done}/{total}")
    if verbose:
        print(f"  重建完成：{done} 条，索引现有 {count_indexed_messages()} 行")
    return {"total": total, "indexed": done, "rows": count_indexed_messages()}


def drop_message_index_rows(message_ids: list[int]) -> None:
    """整合完成后删除消息索引对应行（已整理的消息不再作为"近期原始对话"源）。"""
    if not message_ids:
        return
    client = get_milvus()
    client.delete(settings.MESSAGE_INDEX_COLLECTION,
                  filter=f"message_id in {list(message_ids)}")


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0
