"""CLI 入口。

    python main.py                      # 陪伴 Profile，交互对话
    python main.py --profile coding      # 编码 Profile（工作区默认 = 项目根）
    python main.py --profile coding --workspace D:\\some\\repo
    python main.py --trace               # 显示本回合工具轨迹
    python main.py --show-prompt         # 打印每轮实际发送的完整 prompt

Profile 决定：system 提示词、常驻工具集、沙箱允许根、步数/输出上限、会话归属。

退出时会跑离线任务：压缩 → 记忆固化 → 每日总结。
（编码会话不参与后两步，见 `db.NON_MEMORY_SOURCES`。）
"""
from __future__ import annotations

import argparse
import sys
import time

from config import get_settings
from core.loop import ask, get_session
from core.profile import BY_NAME, COMPANION, with_workspace

settings = get_settings()

_CMDS = "/exit 退出   /trace 轨迹   /prompt 打印prompt   /ctx 上下文"


def _banner(prof) -> str:
    return (f"\n╭──────────────────────────────────────────────╮\n"
            f"│  {prof.label}  ·  agent v20\n"
            f"│  模型 {settings.CHAT_MODEL}\n"
            f"│  常驻工具 {len(prof.core_tools)} 个 · 步数上限 {prof.max_steps} "
            f"· 输出上限 {prof.max_output_tokens}\n"
            f"│  工作区 {', '.join(str(r) for r in prof.workspace_roots)}\n"
            f"│  {_CMDS}\n"
            f"╰──────────────────────────────────────────────╯\n")


def _ctx_line() -> str:
    from core.context import counter
    s = get_session()
    tokens = counter.count_messages(
        [{"role": "system", "content": " "}] +
        [{"role": t["role"], "content": t["content"]} for t in s.history])
    return (f"上下文：{len(s.history)} 条历史 / 约 {tokens:,} token"
            f"（窗口 {settings.CTX_WINDOW:,}，{tokens / settings.CTX_WINDOW:.0%}）"
            f"· 压缩次数状态见 DB · token 计数来源：{counter.describe()}")


def dump_prompt(messages: list[dict]) -> None:
    """把这一轮实际发给模型的完整 messages 原样打出来（复盘用）。

    四段式的分界会以 ──── 标出：system / 历史 / 本轮输入 / 本轮临时消息（工具轨迹）。
    """
    if not messages:
        print("（没有记录到 prompt）")
        return
    print("\n" + "═" * 72)
    print(f"本轮实际发送的 messages（共 {len(messages)} 条）")
    print("═" * 72)
    for i, m in enumerate(messages):
        role = m.get("role")
        tag = {0: "① system（恒定，永不改变）"}.get(i, "")
        if role == "user" and i == len(messages) - 1:
            tag = "③ 本轮输入（尾部通知 + 原文）"
        elif role == "user":
            tag = "② 历史（追加式，带稳定时间前缀）"
        elif role in ("tool",) or m.get("tool_calls"):
            tag = "④ 本轮临时消息（工具轨迹，回合结束即丢）"
        print(f"\n─── [{i}] {role} {tag} " + "─" * 20)
        content = m.get("content") or ""
        print(content if len(content) < 4000 else content[:4000] + "\n…（截断）")
        for tc in (m.get("tool_calls") or []):
            print(f"  ↳ tool_call: {tc['function']['name']}({tc['function']['arguments']})")
    print("\n" + "═" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(BY_NAME), default="companion",
                        help="用哪个 Profile（默认 companion）")
    parser.add_argument("--workspace", default=None,
                        help="覆盖 Profile 的工作区（必须是已存在的目录）")
    parser.add_argument("--trace", action="store_true", help="显示工具轨迹")
    parser.add_argument("--show-prompt", action="store_true",
                        help="每轮打印实际发送的完整 prompt（复盘代码流程用）")
    args = parser.parse_args()

    prof = BY_NAME[args.profile]
    if args.workspace:
        try:
            prof = with_workspace(prof, args.workspace)
        except NotADirectoryError as exc:
            print(f"⚠️  {exc}")
            return 2

    print(_banner(prof))
    print("（首次启动会装载会话历史，纯读取，不发 LLM 请求）\n")

    show_trace = args.trace
    show_prompt = args.show_prompt
    started = time.time()
    first = True

    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/trace":
            show_trace = not show_trace
            print(f"（轨迹显示：{'开' if show_trace else '关'}）")
            continue
        if user_input == "/prompt":
            show_prompt = not show_prompt
            print(f"（prompt 打印：{'开' if show_prompt else '关'}）")
            continue
        if user_input == "/ctx":
            print(_ctx_line())
            continue

        try:
            result = ask(user_input, profile=prof)
        except Exception as exc:
            print(f"\n⚠️  出错：{type(exc).__name__}: {exc}\n")
            continue

        if show_prompt:
            dump_prompt(result.get("prompt") or [])

        if first:
            print(f"（预加载耗时 {time.time() - started:.1f}s）")
            first = False

        if show_trace:
            for item in result.get("trace", []):
                name = f" {item.get('name')}" if item.get("name") else ""
                print(f"  · {item.get('kind')}{name}: {str(item.get('content'))[:160]}")
            print(f"  · ctx {result.get('ctx_tokens', 0):,} token "
                  f"({result.get('ctx_ratio', 0):.0%}) · steps {result.get('steps')}")

        print(f"\n{prof.label} > {result.get('response', '')}\n")

    _shutdown()


def _shutdown() -> None:
    """退出时的离线整理：压缩 → 记忆固化 → 每日总结。

    压缩必然改写历史前缀、使缓存失效，退出时才是没有缓存可炸的时刻。
    """
    if not settings.OFFLINE_ON_EXIT:
        print("再见啦～")
        return
    print("\n[退出] 整理中…（压缩上下文 / 固化记忆 / 每日总结）")
    try:
        from offline.run import run
        run(verbose=True)
    except Exception as exc:
        print(f"  离线整理失败：{type(exc).__name__}: {exc}")
    print("再见啦～")


if __name__ == "__main__":
    sys.exit(main())
