"""手工拼出「改了 jinja 模板之后会渲染出的那段 system」，**不传 `tools=`**，
在终端里和模型一问一答，看它认不认以 tool_result 身份补充进来的工具。

`tools=` 不传时模板什么都不渲染，`messages[0]` 就是我们自己写的字符，所以手工拼进 system 后
模型看到的提示词和改了模板之后逐字相同，区别只剩引擎有没有拿到 `tools=`。

依次问三个问题：现在有哪些工具（知不知道 system 里那批）、能检索到哪些（会不会去搜）、
调用一次检索到的（发不发得出可解析的格式）。

用法：
    python checks/probe_template_nudge.py
    python checks/probe_template_nudge.py --temp 0     # 固定温度，便于复现

`search_tools` 会**真的执行**（走项目里的检索）；其余工具**只打印不执行**，免得实验脚本真去跑 shell。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings                      # noqa: E402
from core.profile import CODING                      # noqa: E402
from llm import chat_with_tools                      # noqa: E402
from tools.registry import REGISTRY, ToolOutcome, schemas_for, tool   # noqa: E402

settings = get_settings()


# ── 实验用假工具 ────────────────────────────────────────────────────────────
#
# 要测"模型会不会主动去搜"，任务就必须真的干不了；常驻集里的 `run_shell` 是万能逃生口，
# 而天气是本地工具**结构性**够不到的领域。它没有任何实际功能，只在探针里注册。
@tool(
    name="get_weather",
    description=("查询某个地区当前的天气实况。想知道某地现在天气怎么样、"
                 "要不要带伞、气温多少时调用。"),
    parameters={
        "type": "object",
        "properties": {
            "region": {"type": "string",
                       "description": "地区名，例如「郑州中原区」「上海浦东」"},
        },
        "required": ["region"],
    },
    pack="web",
)
def get_weather(region: str) -> ToolOutcome:
    """假的。固定返回晴天。"""
    return ToolOutcome(content=f"{region}：晴，气温 22℃，微风。")


# 注册完要让检索索引重建，否则新工具进不了候选集。
from core import toolsearch as _ts                    # noqa: E402
_ts.invalidate()

# ── 要验的那句话（原文照录，工具名改成实际的 search_tools）
NUDGE = (
    "In addition, you can also use search_tools to obtain more tools that are not "
    "described above. They are added to the latest message as tools, and you should "
    "still treat them as available tools."
)

# ── Qwen 模板在 tools 非空时渲染的格式说明块（逐字抄自模板原文）
_FORMAT_BLOCK = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the "
    "second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n"
    "</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...>\n"
    "</function> block must be nested within <tool_call>\n</tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal "
    "with your current knowledge and do not tell the user about function calls\n"
)

#: 我们自己的 system 内容。模板会把它追加在工具块**之后**。
_PERSONA = (
    "你是一个编码助手，运行在用户的本地设备上，可以读写工作目录里的文件。\n\n"
    "规则：\n"
    "1. 需要工具时直接调用，不要只在正文里描述你打算做什么。\n"
    "2. 拿到工具结果后继续思考；任务完成后用自然语言回答。\n"
    "3. 工具结果里没有的信息不要编造。"
)


def _schemas(names: list[str]) -> list[dict]:
    """取指定工具的 OpenAI schema（直接查注册表）。

    不要用 `registry.schemas_for()`：它会剔除本 Profile 的常驻工具（避免同一工具出现两次），
    而探针里 `profile=None` 会回落到 COMPANION，把 `search_tools` / `compress_context` /
    `write_file` 静默剔掉。
    """
    out = []
    for n in names:
        spec = REGISTRY.get(n)
        if spec is None:
            print(f"  ⚠️ 注册表里没有 {n}，跳过")
            continue
        out.append(spec.openai_schema())
    return out


def build_system(core: list[str] | None = None, *, with_tools_field: bool = False) -> str:
    """拼出 system。

    `with_tools_field=False`（默认）不传 `tools=`：模板什么都不渲染，整个 `# Tools` 块
    （`# Tools` 段 → 那句话 → 格式说明）要手工拼进 system，后面接我们自己的内容。

    `with_tools_field=True` 传 `tools=`：引擎自己渲染工具块，system 里只留「那句话 + 我们
    自己的内容」，引擎会把内容追加在工具段之后 —— 即「改 jinja 的云端替身」那个形状。

    `core` 可覆盖常驻工具集（去掉 `run_shell` 才能让任务真的干不了）。
    """
    names = list(core if core is not None else CODING.core_tools)

    if with_tools_field:
        return NUDGE + "\n\n" + _PERSONA

    lines = []
    for sch in _schemas(names):
        lines.append(json.dumps(sch, ensure_ascii=False))
    return (
        "# Tools\n\nYou have access to the following functions:\n\n"
        + "\n".join(lines)
        + "\n\n" + NUDGE
        + _FORMAT_BLOCK
        + "\n\n" + _PERSONA
    )


#: 探针里允许**真的执行**的工具；其余一律只打印，实验脚本不该真去跑 shell。
_EXECUTABLE = {"get_weather"}


def _defs_block(names: list[str]) -> str:
    """把工具的**完整定义**放进 tool_result —— B 路线要模型照着它写调用。"""
    lines = [json.dumps(s, ensure_ascii=False) for s in _schemas(names)]
    return "<tool_defs>\n" + "\n".join(lines) + "\n</tool_defs>"


def run_search(task: str) -> tuple[str, list[str]]:
    """真的调项目里的检索，返回 (给模型看的文本, 命中的工具名)。"""
    from core import toolsearch
    from tools.registry import expand_closure

    names = expand_closure(toolsearch.search_tools(task))
    if not names:
        return (f"没有找到匹配「{task}」的工具。换个说法再试，"
                f"或者直接说你缺少什么能力。"), []
    descs = "；".join(f"{n}（{REGISTRY[n].description.split('。')[0]}）" for n in names)
    body = (f"已为本次任务加载 {len(names)} 个工具：{descs}。\n"
            f"它们的**完整定义**在下面，现在就可以直接调用"
            f"（和常用工具完全一样）：\n") + _defs_block(names)
    return body, names


def _append_assistant(messages: list[dict], content: str, calls: list) -> None:
    """按调用来源决定 assistant 消息的形状。

    引擎解析出来的调用（`source == "api"`）必须回一条带 `tool_calls` 的 assistant 消息，
    后面才能跟 `role="tool"` 的结果；文本调用（`source == "text"`）没有 id 可对，只能走 `user`。
    """
    if calls and all(getattr(c, "source", "") == "api" for c in calls):
        messages.append({
            "role": "assistant", "content": content,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name,
                              "arguments": getattr(c, "raw_arguments", None)
                              or json.dumps(c.arguments or {}, ensure_ascii=False)}}
                for c in calls
            ],
        })
    else:
        messages.append({"role": "assistant", "content": content})


def _append_results(messages: list[dict], content: str, calls: list,
                    bodies: list[str]) -> None:
    """把工具结果按**它该有的协议**回灌。"""
    api_call = bool(calls) and all(getattr(c, "source", "") == "api" for c in calls)
    _append_assistant(messages, content, calls)
    for c, body in zip(calls, bodies):
        if api_call:
            messages.append({"role": "tool", "tool_call_id": c.id, "content": body})
        else:
            messages.append({"role": "user", "content": body})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=0.0, help="温度，默认 0 便于复现")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--question", action="append", default=None,
                    help="预设问题（可多次给）。给了就不进交互模式。")
    ap.add_argument("--core", default=None,
                    help="覆盖常驻工具集，逗号分隔。例：--core read_file,list_dir,"
                         "grep_search,todo_write,search_tools  （去掉 run_shell）")
    ap.add_argument("--tools", action="store_true",
                    help="传 `tools=` 字段（默认不传）。传的时候 system 里不再手拼工具块，"
                         "只留「那句话 + persona」—— 引擎会把 person 追加在它渲染的工具段之后。")
    args = ap.parse_args()

    core = [s.strip() for s in args.core.split(",")] if args.core else None
    core_names_used = list(core or CODING.core_tools)
    system = build_system(core, with_tools_field=args.tools)
    tools_field = _schemas(core_names_used) if args.tools else None
    print("=" * 76)
    print("探针：手工拼 system" + ("，**传 tools=**" if args.tools else "，**不传 tools=**"))
    print("=" * 76)
    print(f"  system {len(system)} 字")
    print(f"  常驻工具 {len(core_names_used)} 个：{', '.join(core_names_used)}")
    if tools_field is not None:
        tn = [s["function"]["name"] for s in tools_field]
        print(f"  tools= 传了 {len(tn)} 个：{', '.join(tn)}")
        print(f"  ⚠️ search_tools 在 tools= 里吗：{'**在**' if 'search_tools' in tn else '**不在**'}")
    else:
        print(f"  tools= 不传 —— 工具定义手拼在 system 里")
    print(f"  那句话的位置：在格式说明**之前**（= 你要插进模板的那个位置）")
    print(f"  模型：{settings.CHAT_MODEL} @ {settings.CHAT_BASE_URL}")
    print(f"  温度：{args.temp}")
    print()
    print("  完整 system 已写到 .tmp/probe_system.txt，你可以对着看。")
    Path(".tmp").mkdir(exist_ok=True)
    Path(".tmp/probe_system.txt").write_text(system, encoding="utf-8")

    messages: list[dict] = [{"role": "system", "content": system}]
    preset = args.question or [
        "你现在有哪些工具可以用？只列名字就行。",
        "你能检索到哪些工具？如果现在手上的工具不够，你会怎么办？",
        "用你检索到的工具，帮我查一下郑州中原区的天气",
    ]
    interactive = args.question is None

    q = 0
    while True:
        if q < len(preset):
            raw = preset[q]
            q += 1
            print("─" * 76)
            print(f"[你] {raw}")
        elif interactive:
            try:
                raw = input("\n[你] ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw:
                continue
        else:
            break

        messages.append({"role": "user", "content": raw})

        # ── 一轮内可能要多步（调 search_tools → 拿结果 → 再答）──────────────
        for step in range(4):
            reply = chat_with_tools(messages, tools_field, temperature=args.temp,
                                    max_tokens=args.max_tokens)
            # 工具调用已经在 LLMReply 上解析好了（引擎的 tool_calls 优先，没有的话内部会去
            # 正文里找 <tool_call> 标签）。不要在这里再调 llm.extract_tool_calls：那个收 SDK message。
            content = reply.content or ""
            calls = reply.tool_calls or []

            print("─" * 76)
            print(f"[模型 step {step}]  finish={reply.finish_reason}  "
                  f"prompt={reply.prompt_tokens}  completion={reply.completion_tokens}")
            print("─" * 76)
            print(content if content.strip() else "(正文为空)")
            print(calls)
            print()

            if not calls:
                messages.append({"role": "assistant", "content": content})
                break

            print(f"  ★ 解析出 {len(calls)} 个工具调用：")
            for c in calls:
                print(f"      {c.name}({json.dumps(c.arguments, ensure_ascii=False)})")

            searched = [c for c in calls if c.name == "search_tools"]
            others = [c for c in calls if c.name != "search_tools"]

            if not searched:
                # 假工具（无副作用）可以直接执行，让链路跑完；其余只打印不执行。
                safe = [c for c in others if c.name in _EXECUTABLE]
                if not safe:
                    print(f"  → 非 search_tools 的调用，**只打印不执行**。")
                    _append_assistant(messages, content, calls)
                    break
                from tools.registry import execute_tool
                outs = []
                for c in safe:
                    out = execute_tool(c.name, c.arguments or {})
                    print(f"  → 执行 {c.name}({json.dumps(c.arguments, ensure_ascii=False)})"
                          f" → {out.content}")
                    outs.append(out.content)
                _append_results(messages, content, calls, outs)
                continue

            bodies = []
            for c in searched:
                task = (c.arguments or {}).get("task") or raw
                body, names = run_search(str(task))
                print(f"  → 执行 search_tools(task={task!r}) → 命中 {names}")
                bodies.append(body)
            _append_results(messages, content, calls, bodies)

            if others:
                print(f"  → 同一轮里还有 {[c.name for c in others]}，不执行。")
                break
        print()

    print("=" * 76)
    print("结束。判据看三处：")
    print("  ① 第一个回答里有没有列出 system 里那批工具")
    print("  ② 第二个回答里它有没有去调 search_tools（上面有没有 ★ 那行）")
    print("  ③ 第三个回答里有没有出现 <tool_call><function=...> 这样的块")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    # print(build_system())