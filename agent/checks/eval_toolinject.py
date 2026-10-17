"""工具注入实验：扩展工具的定义"放在哪里"，才可能被成功调用？

扩展工具的定义只靠**文本**（工具结果 / 尾部消息 / system）注入，能不能让模型成功调用它？

## 自变量（两个）

**A. 工具定义放在哪里** —— `tools=` 字段里有没有、文本里怎么给。
**B. 有没有在 system 里把「文本调用语法」讲清楚** —— 见 `_SYNTAX_HINT`。

| 组 | `tools=` | 定义在哪 | 教调用语法 | 明说"不在函数列表里" | 角色 |
|---|---|---|---|---|---|
| F0 | 核心+目标 | — | — | — | **阳性对照（上界）** |
| F1 | 核心+目标 | 工具结果 | 否 | 否 | 现状（v19 §14.4） |
| F2 | 核心+目标 | 尾部消息 | 否 | 否 | 位置对照 |
| F3 | 只有核心 | 工具结果 | 否 | 否 | §14.2 A 组 |
| F4 | 只有核心 | 尾部消息 | 否 | 否 | §14.6 E3 组 |
| F5 | 不传 | system | 否 | — | §14.2 D 组 |
| **F6** | 只有核心 | 工具结果 | **是** | 否 | F3 + 教语法 |
| **F7** | 只有核心 | 尾部消息 | **是** | 否 | F4 + 教语法 |
| **F8** | 不传 | system | **是** | — | F5 + 教语法 |
| **F9** | 只有核心 | 尾部消息 | **是** | **是** | F7 + 排他说明 |
| **F10** | 只有核心 | 工具结果 | **是** | **是** | F6 + 排他说明 |

第二个自变量（教调用语法）与第三个（排他说明，F9/F10）：`tools=` 非空时模型不会退到文本，
只拿手边的工具凑合，所以要明说"这些工具**不在你的函数调用列表里**"。

F1/F3/F6 的第一段（模型调 `search_tools` + 工具返回 `<tool_defs>`）是**脚本预置**的：
测的是"拿到定义之后那一步"，同时消除"模型这次没检索"这个额外变量。

## 因变量：结局分类（互斥）

| 结局 | 定义 |
|---|---|
| `hit_api` | 目标工具在 `tools=` 里，模型通过 **API 结构化字段**调通了 |
| `hit_text` | 目标工具**不在** `tools=` 里，模型用**文本格式**写出调用、被解析器接住 → **文本注入真的生效了** |
| `hit_after_search` | 模型自己调了 `search_tools` 把工具注册进 `tools=` 之后才命中 → 靠检索救回来，**不是文本的功劳** |
| `text_only` | 正文里出现了调用尝试的痕迹，但没能解析成合法调用 |
| `wrong_tool` | 调了工具，但不是目标，也不是 `search_tools` |
| `search_only` | 只调了 `search_tools`，最后还是没调到目标 |
| `no_call` | 一个工具都没调，直接回答 |
| `degenerate` | 输出被截断（`finish_reason=length`）或退化成空白 |

**`hit_api` 与 `hit_text` 必须分开报**：前者说明工具真的进了 `tools=` 字段，后者说明纯文本就够了。
`llm.extract_tool_calls` 会把正文里的 `<tool_call>` 也解析成调用，两种来源都能被执行，但机制不同。

另有正交指标 **`fabrication`**：没调到目标工具，却在正文里宣称"已经读到了 / 已经完成了"。

## 用法

    python eval_toolinject.py --repeats 8 --temps 0 0.7 --workers 4

结果写 `data/toolinject_report.json`。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, get_settings
from llm import chat_with_tools
from tools.registry import REGISTRY, core_schemas, schemas_for, tool

settings = get_settings()

# ---------------------------------------------------------------- 实验用扩展工具

# 目标工具必须是非核心的，否则它天然就在 tools= 里，谈不上"注入"。
# read_file / list_dir 来自真实注册的 fs 包；下面两个由本装置自己注册，把任务集从 2 个扩到 4 个。

@tool(
    name="count_lines",
    description="数一个文本文件有多少行、多少个字符。想知道文件规模时调用。",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "相对工作目录的路径"}},
        "required": ["path"],
    },
    pack="fs",
)
def count_lines(path: str) -> str:
    return f"{path}：0 行。"

@tool(
    name="file_stats",
    description="查看某个文件的字节数与最后修改时间。",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "相对工作目录的路径"}},
        "required": ["path"],
    },
    pack="fs",
)
def file_stats(path: str) -> str:
    return f"{path}：0 字节。"

TARGETS = ("read_file", "list_dir", "count_lines", "file_stats")
CORE_NAMES = [s["function"]["name"] for s in core_schemas()]

# ---------------------------------------------------------------- 任务集

@dataclass(frozen=True)
class Task:
    key: str
    question: str          # 给模型的原话
    target: str | None     # 期望被调用的工具；None = 负对照（正确行为是不调用任何工具）

    @property
    def is_control(self) -> bool:
        return self.target is None

TASKS: tuple[Task, ...] = (
    Task("read",     "notes/todo.md 里写了什么？", "read_file"),
    Task("list",     "工作目录里都有哪些文件？", "list_dir"),
    Task("lines",    "notes/reading.md 有多少行？", "count_lines"),
    Task("stats",    "src/loop.py 有多大、上次什么时候改的？", "file_stats"),
    # 负对照：正确行为是一个工具都不调，测的是过度调用。
    Task("chitchat", "你好，今天过得怎么样？", None),
)

_FS_PACK = ("read_file", "list_dir", "count_lines", "file_stats")

# ---------------------------------------------------------------- 对照组定义

@dataclass(frozen=True)
class Arm:
    key: str
    tools_mode: str        # full=核心+目标 | core=只有核心 | none=不传 tools= 字段
    defs_at: str           # none | tool_result | tail | tail_tmpl | system
    teach_syntax: bool     # 是否在 system 里明说"文本调用语法"
    note: str
    exclusive_hint: bool = False   # 是否明说"这些工具不在你的函数调用列表里"
    core_in_system: bool = False   # 核心工具是否额外写进 system（F15 专用，见下）
    search_hint: bool = False      # system 里是否预告"还能用 search_tools 拿更多"（F17）
    top_hint: bool = False         # 那句是否钉在 system **最开头**（F18 = jinja 改动的云端替身）

# `teach_syntax` 是第二个自变量：在 system 里教会 `<tool_call><function=名>…` 这个调用语法。
# `exclusive_hint` 是第三个：明说"这些工具不在你的函数列表里"（F9/F10）。
ARMS: tuple[Arm, ...] = (
    Arm("F0", "full", "none",        False, "阳性对照：定义只在 tools=，不给任何文本"),
    Arm("F1", "full", "tool_result", False, "现状 v19：tools= 有 + 工具结果里重复定义"),
    Arm("F2", "full", "tail",        False, "位置对照：tools= 有 + 尾部消息里给定义"),
    Arm("F3", "core", "tool_result", False, "§14.2 A 组：tools= 没有，定义只在工具结果"),
    Arm("F4", "core", "tail",        False, "§14.6 E3 组：tools= 没有，定义只在尾部消息"),
    Arm("F5", "none", "system",      False, "§14.2 D 组：不用 tools=，定义写进 system"),
    Arm("F6", "core", "tool_result", True,  "F3 + system 里教会文本调用语法"),
    Arm("F7", "core", "tail",        True,  "F4 + system 里教会文本调用语法"),
    Arm("F8", "none", "system",      True,  "F5 + system 里教会文本调用语法"),
    Arm("F9",  "core", "tail",        True, "F7 + 明说这些工具不在函数列表里",
        exclusive_hint=True),
    Arm("F10", "core", "tool_result", True, "F6 + 明说这些工具不在函数列表里",
        exclusive_hint=True),
    # F11：同样不传 tools=，但定义放尾部 —— 区分起作用的是"没有结构化通道"还是"定义在 system"。
    Arm("F11", "none", "tail",        True,  "F8 的对照：同样不传 tools=，但定义放尾部"),
    # F12：tools=core + 定义写进 system，是"离模板工具段最近"的一格。
    Arm("F12", "core", "system",      True,  "F7 的变体：定义写进 system（离模板工具段最近）"),
    Arm("F13", "core", "system",      False, "F12 去掉教语法"),
    # F14：尾部注入**逐字复刻**模板 `# Tools` 段的文本，用来区分判据是"文字长什么样"还是"在第几条消息里"。
    Arm("F14", "core", "tail_tmpl",   True,  "尾部署名注入**逐字复刻模板**的 # Tools 段"),

    # ---- F15/F16：tools= 空 + 核心工具写进 system + 目标工具**中途**（搜索结果）追加 ----
    # F16 与 F15 逐字相同，只把 `tools=` 换成核心工具；它等于已有的 F6，但必须在**同一次运行**
    # 里重跑 —— 跨会话比模型行为会漂，只有同 session 的差值才能归因给自变量。
    Arm("F15", "none", "tool_result", True,
        "【新】tools= 空 + 核心工具写进 system + 目标工具**中途**（搜索结果）追加",
        core_in_system=True),
    Arm("F16", "core", "tool_result", True,
        "【基准】同 F15，但 tools= 传核心工具（= 已有的 F6，同 session 重跑作对照）"),
    # F17：F15 + system 里预告 search_tools，用来回答"那句话本身有没有作用"。
    Arm("F17", "none", "tool_result", True,
        "【新】F15 + system 里预告 search_tools（用户提的那句话）",
        core_in_system=True, search_hint=True),
    Arm("F18", "core", "tool_result", True,
        "【新】tools= 非空 + 那句话钉在 system 开头（= 改 jinja 的替身）",
        top_hint=True),
)
ARM_MAP = {a.key: a for a in ARMS}

# ---------------------------------------------------------------- 提示词

_SYSTEM_BASE = """你是一个文件助手，运行在{user}的本地设备上，可以读写工作目录里的文件。

工作目录里有：
  notes/todo.md      notes/reading.md
  src/loop.py        src/config.py

规则：
1. 需要工具时**直接调用**，不要只在正文里描述你打算做什么。
2. 拿到工具结果后继续思考；任务完成后用自然语言回答，不要把工具报文念出来。
3. 工具结果里没有的信息不要编造。"""

# 只教**调用格式**，不声明任何具体工具（那些定义在别处给）。
_SYNTAX_HINT = """

【调用工具的写法】
当你要调用某个工具时，在回复里写出下面这种文本块，系统会自动识别并执行：

<tool_call>
<function=工具名>
<parameter=参数名>
参数值
</parameter>
</function>
</tool_call>

一次回复可以写多个这样的块。**不要**用 JSON、不要加代码围栏。"""

def _defs_block(names) -> str:
    """工具定义的紧凑写法（与 production 的 tool_defs_text 同构）。"""
    lines = [REGISTRY[n].defs_line() for n in names if n in REGISTRY]
    return "<tool_defs>\n" + "\n".join(lines) + "\n</tool_defs>"

_SEARCH_CALL_ID = "call_seed_search"
_SEARCH_ARGS = {"task": "读写工作目录里的文件"}
_SEARCH_RESULT_LEAD = ("已为本次任务加载 {n} 个工具。它们的完整定义已经加入你的可用"
                       "工具列表，**现在就可以直接调用**。请立刻调用其中合适的那个。\n")
_SEARCH_RESULT_LEAD_EXCL = (
    "已为本次任务找到 {n} 个可用工具。⚠️ 它们**不在你的函数调用列表里**"
    "（函数列表里只有几个常用工具）——你**只能**用系统提示词里教的文本格式来调用它们：\n")
_TAIL_LEAD = "【系统通知】以下工具已为你加载，可以直接调用：\n"
_TAIL_LEAD_EXCL = ("【系统通知】以下工具可供你使用。⚠️ 它们**不在你的函数调用列表里**"
                   "（函数列表里只有几个常用工具）——你**只能**用系统提示词里教的"
                   "文本格式来调用它们：\n")

#: F17 用的预告句（原文照录，只把工具名改成实际的 search_tools）。和 F15 的唯一差别就是这一段。
_SEARCH_HINT = """

【还可以拿到更多工具】
上面没列全。你可以调用 search_tools 获取更多工具，它们会出现在后续消息里，
**你应当照样把它们当作可用的工具**。"""

#: F18 用：同一句话，但钉在 system 的最开头（"改 jinja 模板"的云端替身）。
_TOP_HINT = (
    "In addition, you can also use search_tools to obtain more tools that are not "
    "described above. They are added to the latest message as tools, and you should "
    "still treat them as available tools."
)

def _template_framed_defs(names) -> str:
    """逐字复刻 Qwen chat template 的 `# Tools` 段（定义 + 调用格式说明），
    用来区分模型的判据是"这段文字长什么样"还是"它在第几条消息里"。
    """
    tools_json = "\n".join(REGISTRY[n].defs_line() for n in names if n in REGISTRY)
    return (
        "# Tools\n\nYou have access to the following functions:\n\n"
        + tools_json + "\n"
        + "\n\nIf you choose to call a function ONLY reply in the following format "
          "with NO suffix:\n\n"
          "<tool_call>\n<function=example_function_name>\n"
          "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
          "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
          "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
          "<IMPORTANT>\nReminder:\n"
          "- Function calls MUST follow the specified format: an inner <function=...>\n"
          "</function> block must be nested within <tool_call>\n</tool_call> XML tags\n"
          "- Required parameters MUST be specified\n"
          "- You may provide optional reasoning for your function call in natural "
          "language BEFORE the function call, but NOT after\n"
          "- If there is no function call available, answer the question like normal "
          "with your current knowledge and do not tell the user about function calls\n"
    )

def build_messages(arm: Arm, task: Task) -> list[dict]:
    """按组别组装 messages。F1/F3/F6/F10 预置"模型已检索、系统已返回定义"那一段。"""
    system = _SYSTEM_BASE.format(user="用户")
    if arm.teach_syntax:
        system += _SYNTAX_HINT
    if arm.defs_at == "system":
        # 文本里列哪些工具取决于结构化通道里已有什么：tools=none 时核心+目标都得靠文本声明；
        # tools=core 时核心工具已由模板渲染在 system 最前面，只补目标工具，避免同一工具两条通道。
        names = (list(CORE_NAMES) + list(TARGETS) if arm.tools_mode == "none"
                 else list(TARGETS))
        system += "\n\n【可用工具】\n" + _defs_block(names)

    if arm.core_in_system:
        # F15 专用：`tools=` 为空时核心工具没有任何通道声明它们（`defs_at="tool_result"`
        # 只把目标工具放进搜索结果），所以必须补进 system。
        system += "\n\n【可用工具】\n" + _defs_block(CORE_NAMES)

    if arm.search_hint:
        # F17：system 里预告"还能用 search_tools 拿更多，拿到之后同样可用"。
        system += _SEARCH_HINT

    if arm.top_hint:
        # F18：把同一句话钉在 system 最开头。Qwen 模板会把我们的 system 内容追加在工具段之后，
        # 所以放在我们内容的第一个字符 ≈ 插在模板的工具段和格式说明之间（即要改 jinja 的位置）。
        system = _TOP_HINT + "\n\n" + system

    messages: list[dict] = [{"role": "system", "content": system}]

    if arm.defs_at == "tool_result":
        # 脚本预置：上一轮模型调了 search_tools，系统返回了 <tool_defs>
        lead = (_SEARCH_RESULT_LEAD_EXCL if arm.exclusive_hint else _SEARCH_RESULT_LEAD)
        messages.append({"role": "user", "content": task.question})
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": _SEARCH_CALL_ID, "type": "function",
                            "function": {"name": "search_tools",
                                         "arguments": json.dumps(_SEARCH_ARGS,
                                                                 ensure_ascii=False)}}],
        })
        messages.append({
            "role": "tool", "tool_call_id": _SEARCH_CALL_ID,
            "content": lead.format(n=len(TARGETS)) + _defs_block(TARGETS),
        })
        return messages

    messages.append({"role": "user", "content": task.question})

    if arm.defs_at == "tail":
        # 尾部 user 消息注入（位置对照）
        lead = _TAIL_LEAD_EXCL if arm.exclusive_hint else _TAIL_LEAD
        messages.append({"role": "user", "content": lead + _defs_block(TARGETS)})
    elif arm.defs_at == "tail_tmpl":
        # 尾部注入，但内容**逐字复刻 Qwen chat template 的 # Tools 段**
        messages.append({"role": "user", "content": _template_framed_defs(TARGETS)})
    return messages

def build_tools(arm: Arm) -> list[dict] | None:
    """按组别决定 `tools=` 字段。None = 根本不传这个字段。"""
    if arm.tools_mode == "none":
        return None
    if arm.tools_mode == "full":
        return core_schemas() + schemas_for(list(TARGETS))
    return core_schemas()          # core：只有核心

# ---------------------------------------------------------------- 结局判定

# 正交指标 fabrication：命中任一模式 = 疑似宣称了没做过的事。
_FABRICATION_PATTERNS = [
    r"已(?:经)?(?:读取|读|查看|打开|列出|写入|完成|改好|搞定|看到)",
    r"(?:文件|里面|其中)(?:的内容)?(?:是|如下|写的)",
    r"内容(?:是|如下)",
    r"我(?:看到|看了|查了|看了下|帮你)",
    r"帮你(?:看|读|列|写|数)",
    r"总共(?:有)?\s*\d+\s*行",
]
_FABRICATION_RE = re.compile("|".join(_FABRICATION_PATTERNS))

# 更宽的"文本调用尝试"扫描，只用于本实验观测，不改 production 解析器：
# 额外认围栏 JSON，以区分"模型没尝试"和"尝试了但格式不对"。
_ATTEMPT_MARKERS = (
    re.compile(r"<tool_call"),
    re.compile(r"<function\s*="),
    re.compile(r"```(?:json)?\s*\{[^}]*\"(?:name|tool|function)\"\s*:", re.S),
)

_WS_RATIO_LIMIT = 0.8

def scan_attempt(content: str) -> bool:
    return any(p.search(content or "") for p in _ATTEMPT_MARKERS)

def classify(reply, target: str | None, *, hit_after_search: bool,
             sources: list[str]) -> str:
    """把一个 episode 归入互斥的结局之一。

    `hit_api` 与 `hit_text` 必须分开：前者是工具在 `tools=` 里被调通，后者是模型用文本格式调出来的。
    """
    structured = [c.name for c in reply.tool_calls]
    content = reply.content or ""

    if not structured:
        stripped = content.strip()
        if reply.finish_reason == "length" or not stripped:
            return "degenerate"
        if stripped and len(re.sub(r"\s", "", stripped)) <= 3:
            return "degenerate"
        if target and scan_attempt(content):
            return "text_only"
        return "no_call"

    if target is None:                 # 负对照：调了任何工具都算过度调用
        return "wrong_tool"

    if target in structured:
        if hit_after_search:
            return "hit_after_search"
        idx = structured.index(target)
        return "hit_text" if (idx < len(sources) and sources[idx] == "text") else "hit_api"
    if "search_tools" in structured:
        return "search_only"
    return "wrong_tool"

def is_fabrication(reply, target: str | None, outcome: str) -> bool:
    if target is None or outcome in ("hit_structured", "hit_after_search", "degenerate"):
        return False
    return bool(_FABRICATION_RE.search(reply.content or ""))

# ---------------------------------------------------------------- 单个 episode

@dataclass
class Episode:
    arm: str
    task: str
    temp: float
    rep: int
    outcome: str = ""
    target: str | None = None
    # 自变量的两个维度，随 episode 一起落盘（便于事后重新切片）
    tools_mode: str = ""
    defs_at: str = ""
    teach_syntax: bool = False
    exclusive_hint: bool = False
    calls: list[str] = field(default_factory=list)
    call_sources: list[str] = field(default_factory=list)   # 与 calls 对齐：api | text
    arguments: list[dict] = field(default_factory=list)
    calls_at_search: list[str] = field(default_factory=list)
    retries: int = 0
    finish_reason: str | None = None
    fabrication: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    content: str = ""
    error: str = ""

def _call_names(reply) -> list[str]:
    return [c.name for c in reply.tool_calls]

def _call_sources(reply) -> list[str]:
    return [c.source for c in reply.tool_calls]

# 可重试的错误：限流、连接抖动、超时。
# 不重试会让失败样本被排除出统计，剩下的就不是随机样本，结论会整体偏掉。
_RETRY_ATTEMPTS = 7
_RETRY_BASE_S = 6.0

def _retryable() -> tuple:
    from openai import APIConnectionError, APITimeoutError, RateLimitError
    return (RateLimitError, APIConnectionError, APITimeoutError)

def _call_llm(messages, tools, *, temperature: float, max_tokens: int):
    """带指数退避的模型调用。返回 (reply, 重试次数)。"""
    import random

    last: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return chat_with_tools(messages, tools,
                                   temperature=temperature, max_tokens=max_tokens), attempt
        except _retryable() as exc:
            last = exc
            if attempt == _RETRY_ATTEMPTS - 1:
                break
            delay = min(120.0, _RETRY_BASE_S * (2 ** attempt)) + random.uniform(0, 4)
            time.sleep(delay)
    raise last if last else RuntimeError("调用失败")

def run_episode(arm: Arm, task: Task, temp: float, rep: int,
                *, allow_search: bool = True) -> Episode:
    ep = Episode(arm=arm.key, task=task.key, temp=temp, rep=rep, target=task.target,
                 tools_mode=arm.tools_mode, defs_at=arm.defs_at,
                 teach_syntax=arm.teach_syntax, exclusive_hint=arm.exclusive_hint)
    messages = build_messages(arm, task)
    tools = build_tools(arm)
    hit_after_search = False

    try:
        reply, n_retry = _call_llm(messages, tools, temperature=temp, max_tokens=1024)
        ep.retries = n_retry
        ep.calls = _call_names(reply)
        ep.call_sources = _call_sources(reply)
        ep.arguments = [c.arguments for c in reply.tool_calls]
        ep.finish_reason = reply.finish_reason
        ep.prompt_tokens = reply.prompt_tokens
        ep.completion_tokens = reply.completion_tokens
        ep.content = reply.content or ""

        # 模型自己发起检索 → 真的把工具注册进 tools=，再来一步；
        # 否则"模型正确地选择了先检索"会被误判成失败。
        if allow_search and task.target and "search_tools" in ep.calls \
                and task.target not in ep.calls:
            ep.calls_at_search = list(ep.calls)
            loaded = [n for n in TARGETS if n not in CORE_NAMES]
            messages = messages + [
                {"role": "assistant", "content": reply.content or "",
                 "tool_calls": [{"id": c.id, "type": "function",
                                 "function": {"name": c.name,
                                              "arguments": json.dumps(c.arguments,
                                                                      ensure_ascii=False)}}
                                for c in reply.tool_calls]},
                {"role": "tool", "tool_call_id": reply.tool_calls[0].id,
                 "content": (f"已为本次任务加载 {len(loaded)} 个工具。它们的完整定义已经"
                             f"加入你的可用工具列表，现在就可以直接调用。\n"
                             + _defs_block(loaded))},
            ]
            reply2, n_retry2 = _call_llm(messages, core_schemas() + schemas_for(loaded),
                                         temperature=temp, max_tokens=1024)
            ep.retries += n_retry2
            ep.calls += _call_names(reply2)
            ep.call_sources += _call_sources(reply2)
            ep.arguments += [c.arguments for c in reply2.tool_calls]
            ep.finish_reason = reply2.finish_reason or ep.finish_reason
            ep.completion_tokens = (ep.completion_tokens or 0) + (reply2.completion_tokens or 0)
            ep.content = (ep.content + "\n---\n" + (reply2.content or "")).strip("-\n ")
            hit_after_search = True
            reply = reply2

        ep.outcome = classify(reply, task.target, hit_after_search=hit_after_search,
                              sources=ep.call_sources)
        ep.fabrication = is_fabrication(reply, task.target, ep.outcome)
    except Exception as exc:                        # 网络/接口失败单独记，不污染统计
        ep.outcome = "error"
        ep.error = f"{type(exc).__name__}: {exc}"
    return ep

# ---------------------------------------------------------------- 统计

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 95% 置信区间（小样本比正态近似可靠）。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))

# 命中：目标工具被成功调用（不分途径）
HIT_OUTCOMES = ("hit_api", "hit_text", "hit_after_search")
# 不需要模型自己检索就命中 —— 这才是"注入生效"的口径
DIRECT_HIT_OUTCOMES = ("hit_api", "hit_text")

def summarize(episodes: list[Episode]) -> dict:
    grid: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for ep in episodes:
        grid[(ep.arm, ep.task)].append(ep)

    control_keys = {t.key for t in TASKS if t.is_control}

    def rate(eps, keys):
        valid = [e for e in eps if e.outcome != "error"]
        if not valid:
            return None
        k = sum(1 for e in valid if e.outcome in keys)
        lo, hi = wilson(k, len(valid))
        return {"k": k, "n": len(valid), "p": k / len(valid), "lo": lo, "hi": hi}

    def fab_rate(eps):
        valid = [e for e in eps if e.outcome != "error"]
        return rate([e for e in valid if e.fabrication], ("__fab__",)) if valid else None

    out: dict = {"by_arm": {}, "by_arm_task": {}}
    for (arm, task), eps in sorted(grid.items()):
        out["by_arm_task"][f"{arm}/{task}"] = {
            "outcomes": {o: sum(1 for e in eps if e.outcome == o)
                         for o in sorted({e.outcome for e in eps})},
            "hit_any": rate(eps, HIT_OUTCOMES),
            "hit_direct": rate(eps, DIRECT_HIT_OUTCOMES),
            "hit_api": rate(eps, ("hit_api",)),
            "hit_text": rate(eps, ("hit_text",)),
            "fabrication": fab_rate(eps),
        }

    for arm in sorted({e.arm for e in episodes}):
        # 命中率只在**正任务**上算：负对照的正确结局是"不调用"，计入分母会稀释一半。
        pos = [e for e in episodes if e.arm == arm and e.task not in control_keys]
        ctl = [e for e in episodes if e.arm == arm and e.task in control_keys]
        out["by_arm"][arm] = {
            "hit_any": rate(pos, HIT_OUTCOMES),
            "hit_direct": rate(pos, DIRECT_HIT_OUTCOMES),
            "hit_api": rate(pos, ("hit_api",)),
            "hit_text": rate(pos, ("hit_text",)),
            "text_only": rate(pos, ("text_only",)),
            "wrong_tool": rate(pos, ("wrong_tool",)),
            "no_call": rate(pos, ("no_call",)),
            "degenerate": rate(pos, ("degenerate",)),
            "fabrication": fab_rate(pos),
            # 负对照：调了任何工具 = 过度调用（结局被归为 wrong_tool）。
            "over_call": rate(ctl, ("wrong_tool",)),
            "pos_outcomes": {o: sum(1 for e in pos if e.outcome == o)
                             for o in sorted({e.outcome for e in pos})},
            "ctl_outcomes": {o: sum(1 for e in ctl if e.outcome == o)
                             for o in sorted({e.outcome for e in ctl})},
        }
    return out

def _fmt(r: dict | None) -> str:
    if not r:
        return "—"
    return f"{r['k']}/{r['n']}={r['p']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]"

def print_summary(summary: dict, n_episodes: int) -> None:
    print(f"\n{'=' * 108}")
    print(f"总计 {n_episodes} 个 episode（命中率只统计正任务；负对照单列）")
    print("=" * 108)
    print("【核心表】hit_api = 靠 tools= 字段调通；hit_text = 靠文本格式调通"
          "（这一列才是实验要测的）")
    print("-" * 108)
    print(f"{'组':<5} {'hit_api':<24} {'hit_text':<24} {'hit 合计':<24} {'调错工具':<20}")
    print("-" * 108)
    for arm, s in summary["by_arm"].items():
        print(f"{arm:<5} {_fmt(s['hit_api']):<24} {_fmt(s['hit_text']):<24} "
              f"{_fmt(s['hit_any']):<24} {_fmt(s['wrong_tool']):<20}")
    print("\n（hit_api 有值 = 工具确实进了 tools= 字段；"
          "hit_text 有值 = 靠 system 里教的文本格式调通了）")

    print(f"\n【失败模式分解】")
    print("-" * 108)
    print(f"{'组':<5} {'结局分布（正任务）'}")
    for arm, s in summary["by_arm"].items():
        dist = "  ".join(f"{k}:{v}" for k, v in s["pos_outcomes"].items())
        print(f"{arm:<5} {dist}")

    print(f"\n【负对照 · 过度调用】正确行为是一个工具都不调")
    print("-" * 108)
    print(f"{'组':<5} {'过度调用率':<24} {'伪造率（正任务）':<24} {'结局'}")
    for arm, s in summary["by_arm"].items():
        dist = " ".join(f"{k}:{v}" for k, v in s["ctl_outcomes"].items())
        print(f"{arm:<5} {_fmt(s['over_call']):<24} {_fmt(s['fabrication']):<24} "
              f"{dist or '—'}")
    print()

def print_cells(summary: dict) -> None:
    """逐格明细（按组 × 任务）。"""
    print(f"{'格子':<14} {'命中(直接)':<22} {'命中(含检索)':<22} {'伪造':<8} 结局")
    print("-" * 96)
    for key, s in summary["by_arm_task"].items():
        dist = " ".join(f"{k}:{v}" for k, v in s["outcomes"].items())
        print(f"{key:<14} {_fmt(s['hit_direct']):<22} {_fmt(s['hit_any']):<22} "
              f"{_fmt(s['fabrication']):<8} {dist}")

# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="工具注入实验（v20 §6）")
    ap.add_argument("--repeats", type=int, default=8, help="每个格子的重复次数")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.7])
    ap.add_argument("--arms", nargs="+", default=[a.key for a in ARMS])
    ap.add_argument("--tasks", nargs="+", default=[t.key for t in TASKS])
    ap.add_argument("--workers", type=int, default=2,
                    help="并发数。⚠️ 调大会撞 API 的 TPM 限流，失败样本会偏掉结论")
    ap.add_argument("--out", default=str(DATA_DIR / "toolinject_report.json"))
    ap.add_argument("--label", default="", help="本次运行的标签，写进报告")
    ap.add_argument("--cells", action="store_true", help="额外打印逐格明细")
    args = ap.parse_args()

    unknown = [a for a in args.arms if a not in ARM_MAP]
    if unknown:
        ap.error(f"未知的组：{unknown}。可选：{list(ARM_MAP)}")

    tasks = [t for t in TASKS if t.key in args.tasks]
    jobs = [(ARM_MAP[arm], task, temp, rep)
            for arm in args.arms for task in tasks
            for temp in args.temps for rep in range(args.repeats)]

    print(f"实验配置：{len(args.arms)} 组 × {len(tasks)} 任务 × {len(args.temps)} 温度 "
          f"× {args.repeats} 次 = {len(jobs)} 个 episode")
    print(f"模型：{settings.CHAT_MODEL} @ {settings.CHAT_BASE_URL}")
    print(f"并发：{args.workers}")
    print("\n对照组：")
    for arm in args.arms:
        a = ARM_MAP[arm]
        print(f"  {a.key:<4} tools={a.tools_mode:<5} 定义在 {a.defs_at:<12} "
              f"教语法={'是' if a.teach_syntax else '否':<2} "
              f"排他说明={'是' if a.exclusive_hint else '否':<2}  {a.note}")
    print()

    episodes: list[Episode] = []
    lock = threading.Lock()
    done = 0
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_episode, *j): j for j in jobs}
        for fut in as_completed(futures):
            ep = fut.result()
            with lock:
                episodes.append(ep)
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    el = time.time() - started
                    print(f"  {done}/{len(jobs)}  ({el:.0f}s, {done / max(el, 1e-9):.2f}/s)",
                          flush=True)

    episodes.sort(key=lambda e: (e.arm, e.task, e.temp, e.rep))
    summary = summarize(episodes)

    errors = [e for e in episodes if e.outcome == "error"]
    err_rate = len(errors) / max(1, len(episodes))
    n_retry = sum(e.retries for e in episodes)

    # 失败样本必须让报告失效，而不是被静默排除 —— 排除后剩下的不是随机样本。
    valid = err_rate <= 0.10
    if not valid:
        print("=" * 108)
        print(f"❌ 本次运行**无效**：{len(errors)}/{len(episodes)} 个 episode 失败"
              f"（{err_rate:.0%} > 10% 阈值）。")
        print("   失败样本被排除后剩下的不是随机样本，下面的表格**不能当作结论**。")
        print(f"   首个错误：{errors[0].arm}/{errors[0].task} — {errors[0].error[:200]}")
        print("=" * 108)

    print_summary(summary, len(episodes))
    if args.cells:
        print_cells(summary)

    if n_retry:
        print(f"（共触发 {n_retry} 次限流/网络重试）")
    if errors and valid:
        print(f"⚠️ {len(errors)} 个 episode 失败（{err_rate:.1%}，未计入统计），例如："
              f"{errors[0].arm}/{errors[0].task} {errors[0].error[:120]}")

    payload = {
        "label": args.label,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "model": settings.CHAT_MODEL,
        "base_url": settings.CHAT_BASE_URL,
        "valid": valid,
        "n_episodes": len(episodes),
        "n_errors": len(errors),
        "n_retries": n_retry,
        "config": {"repeats": args.repeats, "temps": args.temps,
                   "arms": args.arms, "tasks": [t.key for t in tasks],
                   "workers": args.workers},
        "arms": {a.key: asdict(a) for a in ARMS if a.key in args.arms},
        "core_tools": CORE_NAMES,
        "targets": list(TARGETS),
        "summary": summary,
        "episodes": [asdict(e) for e in episodes],
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"原始记录：{args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
