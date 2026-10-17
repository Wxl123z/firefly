"""SQLAlchemy 数据层：SQLite / PostgreSQL 双兼容。

时间戳统一为 DateTime（本地时区 naive，由 `now()` 生成）；Message 的 kind/visible/meta
做渲染分层（全量入库，渲染层只取该看见的）。风格：SQLAlchemy 2.0。
可移植类型：JSONType（PG=JSONB / SQLite=JSON）、BigIntPK（PG=BIGINT / SQLite=INTEGER）。
"""
import enum
from datetime import date as date_cls, datetime, time as time_cls, timedelta
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Table,
    Text,
    create_engine,
    delete,
    inspect,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from config import get_settings

settings = get_settings()

# JSON 可移植类型：PostgreSQL 用 JSONB，SQLite 退化为 JSON
JSONType = JSON().with_variant(JSONB, "postgresql")
# 大整数自增主键：SQLite 仅 INTEGER PRIMARY KEY 自增，故需变体
BigIntPK = BigInteger().with_variant(Integer, "sqlite")

engine = create_engine(
    settings.DB_URL,
    connect_args={"check_same_thread": False} if settings.DB_URL.startswith("sqlite") else {},
    echo=settings.DB_ECHO,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 声明式基类。"""


def now() -> datetime:
    """本地时间（naive）。全库统一用这个生成时间戳。"""
    return datetime.now()


class TimeStampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=now, onupdate=now, nullable=False
    )


# ---------- 消息种类 ----------

class MessageKind(str, enum.Enum):
    chat = "chat"                # 正常对话：渲染进 prompt 历史
    tool_call = "tool_call"      # 以下几类仅留痕，不进 prompt
    tool_result = "tool_result"
    notice = "notice"


# ---------- 模型（表映射） ----------

# 关联表：Table 里只能用 Column（mapped_column 不支持）
message_media = Table(
    "message_media",
    Base.metadata,
    Column("message_id", Integer, ForeignKey("message.id"), primary_key=True),
    Column("media_id", Integer, ForeignKey("media.id"), primary_key=True),
)


class Person(Base):
    """交际圈人物。"""
    __tablename__ = "person"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    # 列名仍为 relationship（兼容既有数据）；属性改名避免遮蔽 sqlalchemy.relationship
    relationship_type: Mapped[Optional[str]] = mapped_column("relationship", String(64))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, nullable=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=now)


class Conversation(Base):
    __tablename__ = "conversation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(String(256))
    other_person_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("person.id")  # NULL=仅与本人的对话
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=now)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    source: Mapped[Optional[str]] = mapped_column(String(64))
    # 编码会话的**工作区绝对路径**。NULL = 不按工作区区分（陪伴会话，以及 v20 之前的老行）。
    workspace: Mapped[Optional[str]] = mapped_column(String(512))

    messages: Mapped[list["Message"]] = relationship(back_populates="conversation")


class Message(Base):
    __tablename__ = "message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("conversation.id"), nullable=False, index=True
    )
    speaker_type: Mapped[str] = mapped_column(String(16), nullable=False)  # owner/assistant/person
    speaker_person_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("person.id")  # speaker_type=person 时指向谁
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    thinking: Mapped[Optional[str]] = mapped_column(Text)

    # 渲染分层
    kind: Mapped[str] = mapped_column(String(16), default="chat", nullable=False, index=True)
    visible: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    meta: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType)  # 工具名/入参哈希/耗时

    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, nullable=False, index=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(128))
    emotion_tag: Mapped[Optional[str]] = mapped_column(String(64))
    importance: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    privacy_level: Mapped[Optional[str]] = mapped_column(String(16), default="general")
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    review_count: Mapped[Optional[int]] = mapped_column(Integer, default=0)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")

    __table_args__ = (
        Index("ix_message_conv_kind_id", "conversation_id", "kind", "id"),
    )


class Media(Base):
    """媒体文件元数据（原文件在磁盘，库内只存路径+属性）。"""
    __tablename__ = "media"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[str] = mapped_column(String(16), nullable=False)   # image/audio/video
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size: Mapped[Optional[int]] = mapped_column(Integer)
    mime_type: Mapped[Optional[str]] = mapped_column(String(64))
    checksum: Mapped[Optional[str]] = mapped_column(String(64), index=True)  # sha256，去重
    duration: Mapped[Optional[float]] = mapped_column(Float)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    summary: Mapped[Optional[str]] = mapped_column(Text)        # 感知层描述（进向量库）
    transcript: Mapped[Optional[str]] = mapped_column(Text)     # 语音转写
    emotion_tags: Mapped[Optional[str]] = mapped_column(String(256))
    speaker_type: Mapped[Optional[str]] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, nullable=False)
    privacy_level: Mapped[Optional[str]] = mapped_column(String(16), default="sensitive")


class MemoryType(str, enum.Enum):
    event = "event"
    discussion = "discussion"
    stated_fact = "stated_fact"
    stated_preference = "stated_preference"
    inference = "inference"
    summary = "summary"


class MemoryStatus(str, enum.Enum):
    current = "current"
    historical = "historical"


class Memory(TimeStampMixin, Base):
    """长期记忆（schema v4）：quote 锚定原文 / support_count 佐证 /
    status 当前vs历史 / confidence 待审核。
    """
    __tablename__ = "memory"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)

    # --- 分类 ---
    type: Mapped[MemoryType] = mapped_column(
        SAEnum(MemoryType, name="memory_type", native_enum=True), nullable=False, index=True
    )
    status: Mapped[MemoryStatus] = mapped_column(
        SAEnum(MemoryStatus, name="memory_status", native_enum=True),
        nullable=False,
        default=MemoryStatus.current,
        server_default=MemoryStatus.current.value,
        index=True,
    )

    # --- 内容 ---
    fact: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    quote: Mapped[Optional[str]] = mapped_column(Text)   # 直接陈述的原文摘录（锚定保真）

    # --- 出处与置信 ---
    evidence: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSONType, default=list)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    support_count: Mapped[Optional[int]] = mapped_column(Integer)  # 仅 inference 使用

    # --- 时间与分类元数据 ---
    time: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)
    date_range: Mapped[Optional[str]] = mapped_column(String(64))
    topic: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    source_ref: Mapped[Optional[str]] = mapped_column(String(255))
    about_person_id: Mapped[Optional[int]] = mapped_column(
        BigIntPK, ForeignKey("person.id", ondelete="SET NULL"), index=True
    )

    __table_args__ = (
        CheckConstraint("type = 'inference' OR support_count IS NULL",
                        name="ck_memory_support_count_inference_only"),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
                        name="ck_memory_confidence_range"),
        Index("ix_memory_type_status", "type", "status"),
        Index("ix_memory_topic_time", "topic", "time"),
    )


class ContextState(TimeStampMixin, Base):
    """每个会话的滚动上下文状态：摘要 + 水位线 + 压缩冷却。"""
    __tablename__ = "context_state"

    conversation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("conversation.id"), primary_key=True
    )
    summary: Mapped[Optional[str]] = mapped_column(Text)
    # 摘要已覆盖到的 message.id（此 id 及之前的内容已经在 summary 里）
    summary_upto_id: Mapped[int] = mapped_column(BigIntPK, default=0, nullable=False)
    last_compress_turn: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    compress_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class Rollup(TimeStampMixin, Base):
    """每日总结：幂等补课，date 为主键。"""
    __tablename__ = "rollup"

    date: Mapped[str] = mapped_column(String(10), primary_key=True)   # YYYY-MM-DD
    summary: Mapped[Optional[str]] = mapped_column(Text)
    watermark_message_id: Mapped[int] = mapped_column(BigIntPK, default=0, nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class WikiState(TimeStampMixin, Base):
    """LLM Wiki 的增量水位线：只存"我们处理到哪了"。

    页面清单（path / title / aliases / 来源 id）是 Wiki 自己的内部结构，不放 DB，存在
    `wiki/.index.json`（机器读）和 `wiki/index.md`（人读）；`--rebuild` 清空 `wiki/` 从 0 重跑。
    """
    __tablename__ = "wiki_state"

    key: Mapped[str] = mapped_column(String(32), primary_key=True)   # 固定 "default"
    memory_watermark: Mapped[int] = mapped_column(BigIntPK, default=0, nullable=False)
    message_watermark: Mapped[int] = mapped_column(BigIntPK, default=0, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


def get_wiki_state(key: str = "default") -> dict:
    with SessionLocal() as s:
        row = s.get(WikiState, key)
        if row is None:
            return {"key": key, "memory_watermark": 0, "message_watermark": 0,
                    "page_count": 0}
        return {"key": row.key, "memory_watermark": row.memory_watermark or 0,
                "message_watermark": row.message_watermark or 0,
                "page_count": row.page_count or 0}


def save_wiki_state(key: str, *, memory_watermark: int, message_watermark: int,
                    page_count: int, force: bool = False) -> None:
    """写水位线。**单调性保护**：默认只许前进不许后退，否则已写进 Wiki 的证据会被再处理一遍。

    `force=True` 时直接赋值，只有 `--rebuild` 该用它（重建需要把水位线推回 0）。
    """
    with SessionLocal() as s:
        row = s.get(WikiState, key)
        if row is None:
            row = WikiState(key=key, memory_watermark=0, message_watermark=0,
                            page_count=0)
            s.add(row)
        if force:
            row.memory_watermark = int(memory_watermark)
            row.message_watermark = int(message_watermark)
        else:
            row.memory_watermark = max(int(memory_watermark), row.memory_watermark or 0)
            row.message_watermark = max(int(message_watermark),
                                        row.message_watermark or 0)
        row.page_count = int(page_count)
        s.commit()


# ---------- 初始化 ----------

def init_db() -> None:
    Base.metadata.create_all(engine)
    _ensure_columns()


#: 建表之后要补的列：{表名: {列名: DDL 类型}}
#: `create_all` 只建新表、不改已有表，所以这里手动补；只做加列，不改类型、不删列、不动数据。
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "conversation": {"workspace": "VARCHAR(512)"},
}


def _ensure_columns() -> None:
    """轻量迁移：给已有的库补上新增的列。幂等，可反复跑。"""
    insp = inspect(engine)
    existing_tables = set(insp.get_table_names())
    for table, columns in _ADDED_COLUMNS.items():
        if table not in existing_tables:
            continue
        have = {c["name"] for c in insp.get_columns(table)}
        for column, ddl in columns.items():
            if column in have:
                continue
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


# ---------- 通用写入 ----------

def save(obj) -> int:
    """通用写入：ORM 对象 → 入库并返回 id。"""
    with SessionLocal() as s:
        s.add(obj)
        s.commit()
        return obj.id


def link_media_to_message(message_id: int, media_id: int) -> None:
    with SessionLocal() as s:
        s.execute(message_media.insert().values(message_id=message_id, media_id=media_id))
        s.commit()


# ---------- 会话 ----------

def get_or_create_conversation(title: str | None = None, *,
                               source: str | None = None,
                               workspace: str | None = None) -> int:
    """取指定来源（+工作区）的默认会话（id 最小者），没有就建一个。

    `source=None` 表示**陪伴会话**：历史数据的 `Conversation.source` 是 NULL，判据不能改；
    编码 Profile 传 `source="coding"`（见 profile.Profile.conversation_source）。

    `workspace` 是编码会话的第二维身份：传路径 → 只匹配 `workspace` 相同的那条，传 `None`
    → 不做工作区过滤，退回"按 source 找最早一条"（陪伴会话传 `None`，工作区不是它的主题）。
    """
    with SessionLocal() as s:
        q = select(Conversation).order_by(Conversation.id).limit(1)
        q = (q.where(Conversation.source.is_(None)) if source is None
             else q.where(Conversation.source == source))
        if workspace is not None:
            q = q.where(Conversation.workspace == workspace)
        row = s.execute(q).scalar_one_or_none()
        if row is None:
            row = Conversation(title=title or f"与{settings.NAME}的对话",
                               source=source, workspace=workspace)
            s.add(row)
            s.commit()
        return row.id


# ---------- 消息写入 ----------

def append_message(
    conversation_id: int,
    speaker_type: str,
    content: str,
    *,
    kind: str = "chat",
    visible: bool = True,
    ts: datetime | None = None,
    meta: dict | None = None,
    thinking: str | None = None,
    model_version: str | None = None,
) -> int:
    """追加一条消息。对话消息 kind=chat/visible=True；工具轨迹 kind=tool_*/visible=False。"""
    return save(Message(
        conversation_id=conversation_id,
        speaker_type=speaker_type,
        content=content,
        kind=kind,
        visible=visible,
        created_at=ts or now(),
        meta=meta,
        thinking=thinking,
        model_version=model_version,
    ))


# ---------- 查询辅助 ----------

# 不参与**长期记忆固化**与**每日总结**的会话来源。
# 与 `profile.Profile.memory_eligible=False` 一一对应（编码 Profile 传 source="coding"）。
# 必须在这里排除：`get_unreviewed_messages` 与 `get_messages_on_day` 都是**跨会话**查询。
NON_MEMORY_SOURCES: tuple[str, ...] = ("coding",)


def _memory_conversation_ids():
    """参与长期记忆与每日总结的 conversation.id 子查询。"""
    return (
        select(Conversation.id)
        .where(or_(Conversation.source.is_(None),
                   Conversation.source.notin_(NON_MEMORY_SOURCES)))
    )


_ROLE_MAP = {"owner": "user", "person": "user", "assistant": "assistant"}


def _to_dict(m: Message) -> dict:
    return {
        "id": m.id,
        "role": _ROLE_MAP.get(m.speaker_type, "user"),
        "speaker_type": m.speaker_type,
        "content": m.content,
        "kind": m.kind,
        "ts": m.created_at,
    }


def get_conversation_messages(conversation_id: int) -> list[dict]:
    with SessionLocal() as s:
        rows = s.execute(
            select(Message).where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        ).scalars().all()
        return [_to_dict(m) for m in rows]


def load_history(
    conversation_id: int,
    *,
    after_id: int = 0,
    before_id: int | None = None,
    limit: int | None = None,
) -> list[dict]:
    """装载可渲染的历史消息（visible 的对话消息，按 id 升序）。

    - after_id: 只取 id > after_id（跳过已被摘要覆盖的部分）
    - before_id: 只取 id <= before_id（目前无调用方）
    - limit: 只取**最后** N 条；被丢弃的更早消息既没进摘要、又落到水位线之下，此后不会再被装载
    """
    limit = limit or settings.MAX_HISTORY_MESSAGES
    with SessionLocal() as s:
        q = (
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.visible.is_(True),
                Message.kind == MessageKind.chat.value,
                Message.speaker_type.in_(("owner", "person", "assistant")),
                Message.id > after_id,
            )
            .order_by(Message.id)
        )
        if before_id is not None:
            q = q.where(Message.id <= before_id)
        rows = s.execute(q).scalars().all()

    if len(rows) > limit:
        dropped = len(rows) - limit
        print(f"[load_history] ⚠️ 会话 {conversation_id}：水位线之后有 {len(rows)} 条消息，"
              f"超过上限 {limit}；最早的 {dropped} 条（id {rows[0].id}..{rows[dropped - 1].id}）"
              f"将被丢弃且不会被摘要。见 plan/待办与缺陷.md D2。")
    rows = rows[-limit:]
    return [_to_dict(m) for m in rows]


def get_unreviewed_messages(limit: int = 3000) -> list[dict]:
    """离线整合的输入：所有尚未被整合的对话消息（review_count=0）。

    只取 kind=chat 的真实对话，且**排除不参与记忆的会话**（编码会话，见 profile.Profile.memory_eligible）。
    """
    with SessionLocal() as s:
        rows = s.execute(
            select(Message)
            .where(
                or_(Message.review_count.is_(None), Message.review_count == 0),
                Message.kind == MessageKind.chat.value,
                Message.speaker_type.in_(("owner", "person", "assistant")),
                Message.conversation_id.in_(_memory_conversation_ids()),
            )
            .order_by(Message.id)
            .limit(limit)
        ).scalars().all()
        return [{"id": m.id, "conversation_id": m.conversation_id,
                 "speaker_type": m.speaker_type, "content": m.content,
                 "created_at": m.created_at} for m in rows]


def mark_messages_reviewed(message_ids: list[int]) -> None:
    """整合完成后标记：reviewed_at + review_count+1。"""
    if not message_ids:
        return
    with SessionLocal() as s:
        s.execute(
            update(Message)
            .where(Message.id.in_(message_ids))
            .values(reviewed_at=now(), review_count=Message.review_count + 1)
        )
        s.commit()


def get_messages_on_day(day: date_cls | str, conversation_id: int | None = None) -> list[dict]:
    """取某一天（本地时间）的全部对话消息，供 summarize_today / daily_rollup 使用。
    `conversation_id=None` 时是**跨会话**查询，会排除不参与记忆的会话（编码会话）。"""
    if isinstance(day, str):
        day = date_cls.fromisoformat(day)
    start = datetime.combine(day, time_cls.min)
    end = start + timedelta(days=1)
    with SessionLocal() as s:
        q = (
            select(Message)
            .where(
                Message.kind == MessageKind.chat.value,
                Message.created_at >= start,
                Message.created_at < end,
            )
            .order_by(Message.id)
        )
        if conversation_id is not None:
            q = q.where(Message.conversation_id == conversation_id)
        else:
            q = q.where(Message.conversation_id.in_(_memory_conversation_ids()))
        rows = s.execute(q).scalars().all()
        return [_to_dict(m) for m in rows]


def list_active_days(limit: int = 60) -> list[date_cls]:
    """有对话记录的日子（降序），供 rollup 补课扫描；排除不参与记忆的会话。"""
    with SessionLocal() as s:
        rows = s.execute(
            select(Message.created_at)
            .where(Message.kind == MessageKind.chat.value,
                   Message.conversation_id.in_(_memory_conversation_ids()))
            .order_by(Message.created_at.desc())
            .limit(limit * 40)
        ).scalars().all()
    days: list[date_cls] = []
    for ts in rows:
        d = ts.date() if isinstance(ts, datetime) else ts
        if d not in days:
            days.append(d)
    return days[:limit]


# ---------- 上下文状态 ----------

def get_context_state(conversation_id: int) -> dict:
    """取会话的滚动上下文状态；不存在则返回初值（不落库）。"""
    with SessionLocal() as s:
        row = s.get(ContextState, conversation_id)
        if row is None:
            return {"summary": "", "summary_upto_id": 0,
                    "last_compress_turn": 0, "compress_count": 0}
        return {"summary": row.summary or "",
                "summary_upto_id": row.summary_upto_id or 0,
                "last_compress_turn": row.last_compress_turn or 0,
                "compress_count": row.compress_count or 0}


def update_context_state(conversation_id: int, **fields) -> None:
    """更新（不存在则创建）会话上下文状态。

    **单调性保护**：`summary_upto_id` 与 `last_compress_turn` 只许前进不许后退（两列会被并发写）。
    """
    allowed = {"summary", "summary_upto_id", "last_compress_turn", "compress_count"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    with SessionLocal() as s:
        row = s.get(ContextState, conversation_id)
        if row is None:
            s.add(ContextState(conversation_id=conversation_id, **fields))
            s.commit()
            return
        if fields.get("summary_upto_id", 0) < (row.summary_upto_id or 0):
            fields.pop("summary_upto_id", None)
            fields.pop("summary", None)          # 摘要与水位线必须同步，一起丢弃
        if fields.get("last_compress_turn", 0) < (row.last_compress_turn or 0):
            fields.pop("last_compress_turn", None)
        for k, v in fields.items():
            setattr(row, k, v)
        s.commit()


# ---------- 每日总结 ----------

def get_rollup(day: date_cls | str) -> dict | None:
    key = day if isinstance(day, str) else day.isoformat()
    with SessionLocal() as s:
        row = s.get(Rollup, key)
        if row is None:
            return None
        return {"date": row.date, "summary": row.summary or "",
                "watermark_message_id": row.watermark_message_id or 0,
                "message_count": row.message_count or 0}


def upsert_rollup(day: date_cls | str, summary: str, watermark_message_id: int,
                  message_count: int) -> None:
    key = day if isinstance(day, str) else day.isoformat()
    with SessionLocal() as s:
        row = s.get(Rollup, key)
        if row is None:
            s.add(Rollup(date=key, summary=summary,
                         watermark_message_id=watermark_message_id,
                         message_count=message_count))
        else:
            row.summary = summary
            row.watermark_message_id = watermark_message_id
            row.message_count = message_count
        s.commit()


def clear_conversation(conversation_id: int) -> None:
    """删除会话的全部消息与上下文状态（开发/测试用）。"""
    with SessionLocal() as s:
        s.execute(delete(Message).where(Message.conversation_id == conversation_id))
        s.execute(delete(ContextState).where(
            ContextState.conversation_id == conversation_id))
        s.commit()
