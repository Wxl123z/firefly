"""ReAct 对话循环：手写 `while` 编排（preload → call_model ⇄ run_tools → commit）。

关键不变量：
- 提示词分层：前缀 = system + 摘要 + history（稳定，缓存命中）；尾部 = 通知 +
  本轮输入；再后面是本轮工具轨迹（回合结束即丢）
- 工具分两层：核心工具走 `tools=` 字段（恒定）；扩展工具由模型自己调
  `search_tools` 加载进 `tools=`，会话内粘性保留
- 不做启发式检索；压缩由模型决定、系统执行；DB 全量入库但只渲染 visible 消息
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict

from core.compress import compress
from config import get_settings
from core.context import assemble_messages, build_notice, counter
from store.db import (
    append_message,
    get_context_state,
    get_or_create_conversation,
    init_db,
    load_history,
    update_context_state,
)
from llm import chat_with_tools
from store.memory import index_message
import core.plan as plan_store
from core.profile import COMPANION, Profile, activate, deactivate
from tools import active_schemas, execute_tool, pack_index

settings = get_settings()

_COMPRESS_TOOL = "compress_context"

#: 这些工具执行成功就算"有进展"（不收敛护栏的判据之一）；`run_shell` 除外，
#: 它按命令是不是只读来判（见 exec.counts_as_progress）。
_PROGRESS_TOOLS = frozenset({"write_file", "edit_file", "todo_write"})

#: 连续多少步毫无进展就提前收尾。
_STALL_LIMIT = 5


# ---------------------------------------------------------------- state

class TurnState(TypedDict, total=False):
    """一个回合的全部状态。跨回合的部分以 Session 为准，这里是本回合的快照。"""
    # 持久层
    conversation_id: int
    history: list[dict]          # [{id, role, content, ts}]
    summary: str
    summary_upto_id: int
    turn_count: int
    last_compress_turn: int

    # 本轮瞬时层
    user_input: str
    scratch: list[dict]          # 本轮 OpenAI 格式消息（工具轨迹，回合结束即丢）
    forced_compress: bool        # 本轮是**系统**强制压的（不是模型调 compress_context）
    loaded_tools: list[str]      # 只有模型自己调 search_tools 才会增长
    notice: str
    trace: list[dict]            # 本轮完整轨迹（落库，不进 prompt）
    tool_calls: list[dict]       # call_model 填、run_tools 消费
    ctx_tokens: int
    ctx_ratio: float
    steps: int
    stall_steps: int             # 连续多少步没有进展（不收敛护栏）
    stop_reason: str             # answer | steps | stall，决定收尾说什么
    plan: list[dict]             # 任务清单（从 plan store 读，每轮渲染进尾部）
    repeat_guard: list[str]      # 重复工具调用护栏（环形缓冲）
    prompt: list[dict]           # 本轮实际发出的 prompt

    # 输出
    response: str


@dataclass
class Session:
    """跨回合的会话状态。⚠️ 模块级单例，同一进程内只服务一个 Profile：换了
    Profile 而 `ready` 还是上一轮的，必须重新 preload。
    """
    profile_name: str = ""
    conversation_id: int = 0
    #: 装载时用的会话身份 `(source, workspace)`。换工作区必须重装。
    conversation_key: tuple[str | None, str | None] = (None, None)
    turn_count: int = 0
    history: list[dict] = field(default_factory=list)
    summary: str = ""
    summary_upto_id: int = 0
    last_compress_turn: int = 0
    # 已加载的扩展工具：**会话内粘性**。tools= 字段一变，前缀缓存整段失效。
    loaded_tools: list[str] = field(default_factory=list)
    ready: bool = False

    def usable_for(self, profile: Profile) -> bool:
        """比的是完整身份 `(Profile 名, source, 工作区)`，不只是 Profile 名。"""
        return (self.ready and self.profile_name == profile.name
                and self.conversation_key == profile.conversation_key())


_session = Session()


def get_session() -> Session:
    return _session


def reset_session() -> None:
    """测试用：丢弃跨回合状态，含 `readset` 的已读记录和任务清单（旧计划会留到下一轮）。"""
    global _session
    if _session.conversation_id:
        plan_store.clear(_session.conversation_id)
    _session = Session()
    from core import readset
    readset.reset()


def _drop_finished_plan(cid: int) -> list[dict]:
    """读计划，返回这一轮该用的计划 —— **已经全部完成的旧计划直接丢掉**。

    全 completed 的计划留在尾部会让模型宣布任务完成、一次工具都不调。
    """
    items = plan_store.load(cid)
    if items and all(x["status"] == "completed" for x in items):
        plan_store.clear(cid)
        return []
    return items


# ---------------------------------------------------------------- 工具函数

def _reply_to_scratch(reply, calls) -> dict:
    return {
        "role": "assistant",
        "content": reply.content or "",
        "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
            for c in calls
        ],
    }


def memory_note_index_failure(exc: Exception) -> None:
    """记一次向量索引失败。**第一次和每第 50 次**打一行，不刷屏也不沉默。"""
    import store.memory as _m
    _m._INDEX_FAILURES.append(f"{type(exc).__name__}: {exc}")
    del _m._INDEX_FAILURES[:-20]
    n = len(_m._INDEX_FAILURES)
    if n == 1 or n % 50 == 0:
        print(f"⚠️ 向量索引失败（第 {n} 次）：{type(exc).__name__}: {exc}")
        print("   （对话不受影响；索引不全会让 retrieve_memory 只搜到一部分。）"
              "修好嵌入后端后跑 `python maintenance/reindex.py` 重建。")


def _shell_hint_for(profile: Profile) -> str:
    """只有这个 Profile 真能执行命令时，才告诉它是什么 shell。

    工具描述是恒定前缀、不能随环境变，所以它只能走尾部通知。
    """
    if "run_shell" not in profile.core_tools:
        return ""
    try:
        from tools.packs.exec import shell_hint
        return shell_hint()
    except Exception:
        return ""


# ---------------------------------------------------------------- 步骤

def preload(state: TurnState, profile: Profile) -> None:
    """把跨回合的会话状态装进 state。**纯读取，零 LLM 调用、零后台线程。**

    会话按 `profile.conversation_key()` 分流：陪伴是 `(None, None)`（历史数据里
    source 就是 NULL，判据不能改），编码是 `("coding", <工作区绝对路径>)`；换了
    Profile 或换了工作区都必须重装（`usable_for()` 守卫）。
    """
    if not _session.usable_for(profile):
        init_db()
        cid = get_or_create_conversation(
            title=profile.conversation_title(),
            source=profile.conversation_source,
            workspace=profile.conversation_workspace(),
        )
        ctx = get_context_state(cid)
        _session.profile_name = profile.name
        _session.conversation_key = profile.conversation_key()
        _session.conversation_id = cid
        _session.history = load_history(cid, after_id=ctx["summary_upto_id"])
        _session.summary = ctx["summary"]
        _session.summary_upto_id = ctx["summary_upto_id"]
        _session.last_compress_turn = ctx["last_compress_turn"]
        _session.turn_count = 0
        _session.loaded_tools = []      # 换 Profile = 换工具集，粘性缓存不跨 Profile
        _session.ready = True

    state["conversation_id"] = _session.conversation_id
    state["history"] = list(_session.history)
    state["summary"] = _session.summary
    state["summary_upto_id"] = _session.summary_upto_id
    state["turn_count"] = _session.turn_count
    state["last_compress_turn"] = _session.last_compress_turn


def call_model(state: TurnState, *, force_applied: bool,
               profile: Profile) -> tuple[str, bool]:
    """组装 prompt → 调模型 → 解析工具调用。

    返回 `(下一步, force_applied)`：`"tools"` 本轮有工具调用，已填进
    `state["tool_calls"]`；`"answer"` 拿到最终回答，已填进 `state["response"]`；
    `"retry"` 强制压缩的系统兜底触发了，需要重新问一次。

    这里不做启发式决策：记忆和扩展工具都由模型自己调工具去取。
    """
    turn = state["turn_count"]
    summary = state["summary"]
    history = state["history"]
    loaded = list(state["loaded_tools"])
    system_prompt = profile.system_prompt()

    # --- 两遍组装：先估算占用率（尾部通知本身也要占 token）---
    base = assemble_messages(system_prompt, history, state["user_input"],
                             summary=summary)
    est_tokens = counter.count_messages(base)

    notice = build_notice(
        ctx_tokens=est_tokens,
        ctx_ratio=est_tokens / max(settings.CTX_WINDOW, 1),
        packs=pack_index(profile),
        loaded_tools=loaded,
        last_compress_turn=state["last_compress_turn"],
        turn_count=turn,
        forced_compress=state.get("forced_compress", False),
        shell_hint=_shell_hint_for(profile),
        plan_text=plan_store.render(state.get("plan") or []),
        steps=state["steps"],
    )
    messages = assemble_messages(system_prompt, history, state["user_input"],
                                 summary=summary, notice=notice,
                                 scratch=state["scratch"])
    ctx_tokens = counter.count_messages(messages)
    ctx_ratio = ctx_tokens / max(settings.CTX_WINDOW, 1)

    state["notice"] = notice
    state["ctx_tokens"] = ctx_tokens
    state["ctx_ratio"] = ctx_ratio
    state["prompt"] = messages

    # --- 80% 强制压缩：用 tool_choice 强制产生压缩调用（每回合只强制一次）---
    # 注意：`compress_context` 必须在本 Profile 的常驻工具里，否则强制会失败
    # （tool_choice 指向未声明的工具会被服务端拒绝，返回 400）。
    forced = (ctx_ratio >= settings.CTX_FORCE_RATIO and not force_applied
              and _COMPRESS_TOOL in profile.core_tools)
    tool_choice = ({"type": "function", "function": {"name": _COMPRESS_TOOL}}
                   if forced else None)

    reply = chat_with_tools(messages, active_schemas(loaded, profile),
                            tool_choice=tool_choice,
                            max_tokens=profile.max_output_tokens)
    if reply.prompt_tokens:
        counter.calibrate(ctx_tokens, reply.prompt_tokens)   # 用真实 usage 校准估算

    if reply.tool_calls:
        state["tool_calls"] = [
            {"id": c.id, "name": c.name, "arguments": c.arguments,
             "fingerprint": c.fingerprint(), "parse_error": c.parse_error}
            for c in reply.tool_calls
        ]
        state["scratch"] = state["scratch"] + [_reply_to_scratch(reply, reply.tool_calls)]
        return "tools", force_applied

    if forced:
        # 兜底：强制了却没调 → 系统直接压（电路断路器，绝不撑爆窗口），压完重新问。
        # 压完必须让模型知道：记进 `forced_compress`，下一轮 notice 里说明这不是它
        # 调用的 `compress_context`。
        _text, did = _compress_now(state, force=True)
        if did:
            state["forced_compress"] = True
        return "retry", True

    state["response"] = reply.content or ""
    return "answer", force_applied


def _made_progress(state: TurnState, marks: list[tuple[str, bool, dict]],
                   reads_before: int) -> bool:
    """这一批工具调用里有没有**实质性进展**？（不收敛护栏的判据）

    `repeat_guard` 只挡得住"完全一样的调用"，挡不住 "`read a.py` → `read b.py`
    → `read a.py`"这种无效循环。判据：**有没有写过东西、有没有读过新文件**。
    ⚠️ 只读的 `run_shell`（`ls` / `git status` / `cat`）**不算进展**，否则模型
    可以靠反复 `ls` 让护栏永远不触发。
    """
    from core import readset
    if len(readset.snapshot()) > reads_before:
        return True
    for name, ok, args in marks:
        if not ok:
            continue
        if name in _PROGRESS_TOOLS:
            return True
        if name == "run_shell":
            try:
                from tools.packs.exec import counts_as_progress
                if counts_as_progress(str(args.get("command", ""))):
                    return True
            except Exception:
                return True
    return False


def run_tools(state: TurnState) -> bool:
    """执行 `state["tool_calls"]` 里的调用。返回本轮是否发生过压缩。"""
    guard = state["repeat_guard"]
    loaded = state["loaded_tools"]
    compressed = False
    marks: list[tuple[str, bool, dict]] = []
    from core import readset
    reads_before = len(readset.snapshot())

    for call in state["tool_calls"]:
        name, args = call["name"], call.get("arguments") or {}

        if call.get("parse_error"):
            result_text = (f"错误：{call['parse_error']}。"
                           '请用 {"name": "工具名", "arguments": {"参数": 值}} 的格式重新调用。')
            ok = False
        elif guard and guard[-1] == call["fingerprint"]:
            # 重复调用护栏：同样的工具 + 同样的参数，结果不会变
            result_text = (f"错误：你刚刚已经用同样的参数调用过 {name} 了，结果不会变。"
                           "请换一种做法，或者直接把结果说出来。")
            ok = False
        else:
            outcome = execute_tool(name, args)
            guard.append(call["fingerprint"])
            ok = outcome.ok

            if outcome.control == "compress":
                result_text, did = _compress_now(
                    state,
                    keep_recent_tokens=outcome.payload.get("keep_recent_tokens"),
                    focus=outcome.payload.get("focus"),
                )
                compressed = compressed or did
            else:
                if outcome.control == "load_tools":
                    for n in outcome.payload.get("loaded", []):
                        if n not in loaded:
                            loaded.append(n)     # 保持加载顺序，便于按 LRU 淘汰
                result_text = outcome.content

        marks.append((name, ok, args))
        state["scratch"].append(
            {"role": "tool", "tool_call_id": call["id"], "content": result_text})
        state["trace"].append({"kind": "tool_result", "name": name,
                               "content": result_text[:500], "ok": ok})

    state["repeat_guard"] = guard[-8:]
    state["loaded_tools"] = loaded
    state["tool_calls"] = []

    # 不收敛护栏：连续若干步没有任何进展就提前收尾
    if _made_progress(state, marks, reads_before):
        state["stall_steps"] = 0
    else:
        state["stall_steps"] = state.get("stall_steps", 0) + 1
    return compressed


def _compress_now(state: TurnState, *, keep_recent_tokens: int | None = None,
                  focus: str | None = None, force: bool = False) -> tuple[str, bool]:
    """执行压缩并把结果写回 state。返回 `(回灌文本, 是否真的压了)`。

    顺序由 `compress()` 保证：先固化事实 → 再合并摘要 → 再移出历史。
    """
    turn = state["turn_count"] + 1
    result = compress(
        state["conversation_id"],
        state["history"],
        turn_count=turn,
        keep_recent_tokens=keep_recent_tokens,
        focus=focus,
        force=force or state["ctx_ratio"] >= settings.CTX_FORCE_RATIO,
        verbose=False,
    )
    if not result.ok:
        return f"没有执行压缩：{result.reason}", False

    state["history"] = result.keep
    state["summary"] = result.summary
    state["summary_upto_id"] = result.summary_upto_id
    state["last_compress_turn"] = turn

    consolidation = result.consolidated or {}
    kept = (f"，其中 {consolidation.get('created', 0)} 条事实已存入长期记忆"
            if consolidation.get("created") else "")
    return (f"已压缩：{result.removed_count} 条早期对话被整理进摘要并移出上下文"
            f"（约省下 {result.removed_tokens} token）{kept}。"
            "最近的对话没有动，可以继续正常回答。"), True


def _stop_message(state: TurnState, profile: Profile) -> str:
    """没拿到回答就收尾时说什么。**不能是静默兜底**：用户得知道停在哪、还剩什么。"""
    reason = state.get("stop_reason", "answer")
    if reason == "steps":
        head = f"（本回合达到步数上限 {profile.max_steps} 停下，任务可能还没做完。）"
    elif reason == "stall":
        head = (f"（连续 {_STALL_LIMIT} 步没有实质进展 —— 既没改东西也没读到新内容，"
                f"提前收尾了。）")
    else:
        head = profile.fallback_reply

    plan = state.get("plan") or []
    if plan:
        c = plan_store.counts(plan)
        head += f"\n当前计划 {c['completed']}/{c['total']} 完成"
        doing = next((x["content"] for x in plan if x["status"] == "in_progress"), "")
        if doing:
            head += f"，停在「{doing}」"
        pend = [x["content"] for x in plan if x["status"] == "pending"]
        if pend:
            head += f"\n还没做：{'、'.join(pend[:5])}"
    return head


def commit(state: TurnState, profile: Profile) -> None:
    """落库 + 更新会话状态。轨迹入库但不进 prompt（kind=tool_*，visible=False）。"""
    cid = state["conversation_id"]
    user_input = state["user_input"]
    # 计划可能在这一轮里被 todo_write 改过 —— 以 store 为准重新读一次
    state["plan"] = plan_store.load(cid)
    response = state.get("response") or _stop_message(state, profile)

    uid = append_message(cid, "owner", user_input, model_version=settings.CHAT_MODEL)
    for item in state["trace"]:
        kind = "notice" if item.get("kind") == "notice" else "tool_result"
        append_message(cid, "assistant", item.get("content", ""), kind=kind, visible=False,
                       meta={"name": item.get("name")})
    aid = append_message(cid, "assistant", response, model_version=settings.CHAT_MODEL)

    # 向量索引按 Profile 门控：编码会话**不建索引**，否则陪伴模式检索记忆会捞到代码。
    if profile.index_messages:
        for mid, text in ((uid, user_input), (aid, response)):
            try:
                index_message(mid, text)
            except Exception as exc:
                # ⚠️ **失败不能静默。** 索引挂了不该影响对话，但也不能一声不吭：
                #    索引长期不全是没有任何其它症状的，检索照样"能用"，只是搜到的少。
                memory_note_index_failure(exc)

    ts = datetime.now()
    state["history"] = state["history"] + [
        {"id": uid, "role": "user", "content": user_input, "ts": ts},
        {"id": aid, "role": "assistant", "content": response, "ts": ts},
    ]
    state["response"] = response
    state["turn_count"] = state["turn_count"] + 1

    update_context_state(cid, summary=state["summary"],
                         summary_upto_id=state["summary_upto_id"],
                         last_compress_turn=state["last_compress_turn"])

    _session.profile_name = profile.name
    _session.conversation_id = cid
    _session.history = state["history"]
    _session.summary = state["summary"]
    _session.summary_upto_id = state["summary_upto_id"]
    _session.last_compress_turn = state["last_compress_turn"]
    _session.turn_count = state["turn_count"]
    # 已加载工具粘性保留（超上限按最久未用淘汰）
    loaded = list(state["loaded_tools"])
    _session.loaded_tools = loaded[-settings.MAX_LOADED_TOOLS:]
    _session.ready = True


# ---------------------------------------------------------------- 循环

def ask(user_input: str, *, profile: Profile | None = None,
        show_trace: bool = False) -> TurnState:
    """跑一个回合：preload → (call_model ⇄ run_tools)* → commit。

    `profile` 默认 COMPANION，整个回合期间是"当前执行上下文"（`profile.activate`）。
    """
    prof = profile if profile is not None else COMPANION
    token = activate(prof)
    try:
        return _ask_inner(user_input, prof, show_trace=show_trace)
    finally:
        deactivate(token)


def _ask_inner(user_input: str, profile: Profile, *, show_trace: bool) -> TurnState:
    state: TurnState = {
        "user_input": user_input,
        "scratch": [],
        "trace": [],
        "tool_calls": [],
        "steps": 0,
        "stall_steps": 0,
        "forced_compress": False,
        "stop_reason": "answer",
        "repeat_guard": [],
        "loaded_tools": [],
        "response": "",
    }
    preload(state, profile)
    state["loaded_tools"] = list(_session.loaded_tools)
    # 计划**从 store 读**，不从对话历史里捞 —— 压缩会把早期的 todo_write 调用移出上下文。
    state["plan"] = _drop_finished_plan(state["conversation_id"])

    force_applied = False
    while True:
        state["steps"] += 1
        action, force_applied = call_model(state, force_applied=force_applied,
                                           profile=profile)

        if action == "answer":
            state["stop_reason"] = "answer"
            break

        if action == "retry":
            continue

        if action == "tools":
            # 步数耗尽就不再执行工具，交给 commit 用兜底文案（避免无限工具箱循环）
            if state["steps"] >= profile.max_steps:
                state["stop_reason"] = "steps"
                break
            # run_tools 是**命令**不是查询：它真的调用工具并改 scratch/trace/
            # loaded_tools，模型调了 compress_context 时还会换掉 history/summary。
            compressed = run_tools(state)
            # 压过了就别再强制，否则"压缩被护栏拒绝、ctx_ratio 仍超阈值"时会空转到步数耗尽
            force_applied = force_applied or compressed
            if state["stall_steps"] >= _STALL_LIMIT:
                state["stop_reason"] = "stall"
                break
            continue

        raise RuntimeError(f"call_model 返回了未知的 action：{action!r}")

    commit(state, profile)

    if show_trace:
        for item in state["trace"]:
            print(f"  · {item.get('kind')} {item.get('name', '')}: "
                  f"{str(item.get('content'))[:120]}")
    return state
