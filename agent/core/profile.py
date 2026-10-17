"""运行时 Profile：同一套循环 / 注册表 / 上下文管理，装成不同的 agent。

一个 Profile 描述"人格 + 能力边界 + 运行约束"。`activate()` 用 ContextVar 记
"本回合在哪个 Profile 下跑"（工具函数签名固定，没法从 `execute_tool` 一路传参）：
它一次 `ask()` 内恒定，只有 `ask()` 入口写、出口重置，需要它的地方读 `active()`。
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config import get_settings

settings = get_settings()


@dataclass(frozen=True)
class Profile:
    """一个 agent 的"人格 + 能力边界 + 运行约束"。"""

    name: str
    system_prompt: Callable[[], str]        # 前缀第一段。**必须逐字节稳定**
    core_tools: tuple[str, ...]             # 本 Profile 的常驻工具；顺序即 tools= 的渲染顺序
    packs: tuple[str, ...]                  # 本 Profile 能看见的工具包（能力边界）
    workspace_roots: tuple[Path, ...]       # 沙箱允许根（可多个）
    max_steps: int                          # 单回合工具步数上限
    max_output_tokens: int                  # 单次输出上限
    fallback_reply: str                     # 步数耗尽/空回复时的兜底文案
    # 写保护：这些**前缀**下的路径只读（相对根的 posix 路径）。
    # ⚠️ 以下字段都必须带默认值：dataclass 要求"无默认"的排在"有默认"的前面。
    protected_paths: tuple[str, ...] = ()
    approval: str = "destructive"           # auto | destructive | all_writes
    # 会话归属：None = 陪伴会话（历史数据里 source 是 NULL，不能改判据）
    conversation_source: str | None = None
    index_messages: bool = True             # 是否把消息嵌进向量索引（供 retrieve_memory）
    memory_eligible: bool = True            # 是否参与记忆固化 + 每日总结
    label: str = ""                         # CLI 横幅用

    def is_protected(self, rel_posix: str) -> bool:
        """这个相对路径是不是写保护的？

        `protected_paths` 里的项是**前缀**（不是 glob）：`"agent"` 护住 `agent`
        本身和 `agent/**`。
        """
        rel = rel_posix.strip("/")
        for pref in self.protected_paths:
            p = pref.strip("/")
            if p and (rel == p or rel.startswith(p + "/")):
                return True
        return False

    def conversation_workspace(self) -> str | None:
        """会话身份里的工作区那一维。`None` = **不按工作区区分**。

        判据是"这个 Profile 有没有自己的 `conversation_source`"：编码有 → 工作区
        进身份（同一项目的不同副本因此互不串台）；陪伴没有 → 工作区只是它写笔记的
        地方，不进身份，否则那条主对话会被工作区劈成两条。路径先 `resolve()`，
        否则 `D:\\a\\proj` 和 `D:\\a\\..\\a\\proj` 会被当成两个项目。
        """
        if not self.conversation_source or not self.workspace_roots:
            return None
        try:
            return str(Path(self.workspace_roots[0]).resolve())
        except OSError:
            return str(self.workspace_roots[0])

    def conversation_title(self) -> str | None:
        """新会话的标题（纯显示用）。`None` = 用默认标题。"""
        ws = self.conversation_workspace()
        return f"编码 · {ws}" if ws else None

    def conversation_key(self) -> tuple[str | None, str | None]:
        """会话身份 = `(source, workspace)`。`Session.usable_for` 比的就是它。"""
        return (self.conversation_source, self.conversation_workspace())


# ---------------------------------------------------------------- 两个 Profile

def _companion_prompt() -> str:
    from personas.persona import build_system_prompt
    return build_system_prompt()


def _coding_prompt() -> str:
    from personas.coding_persona import build_system_prompt
    return build_system_prompt()


COMPANION = Profile(
    name="companion",
    system_prompt=_companion_prompt,
    # ⚠️ 顺序即 tools= 字段里的渲染顺序。改顺序 = 改前缀 = 缓存纪元失效。
    core_tools=("retrieve_memory", "search_tools", "compress_context",
                "summarize_today", "web_search", "write_file"),
    packs=("memory", "wiki", "web", "fs", "code", "schedule", "meta"),
    # ⚠️ Wiki 根挂进沙箱允许根，`read_file("wiki/concepts/xx.md")` 就能直接读。
    #    根取 `WIKI_DIR` 而非 `WIKI_DIR/wiki`，相对路径才和文档结构一致；
    #    **必须追加在最后**：`conversation_workspace()` 取的是 `[0]`。
    workspace_roots=(settings.AGENT_WORKSPACE, Path(settings.WIKI_DIR)),
    # 对话中 Wiki 只读（能写就成了第二个记忆出口）；`scripts/` 和 `WIKI.md` 一并护住。
    protected_paths=("wiki", "raw", "scripts", "WIKI.md"),
    approval=settings.APPROVAL_POLICY,
    max_steps=settings.MAX_REACT_STEPS,
    max_output_tokens=settings.MAX_OUTPUT_TOKENS,
    fallback_reply="（……我好像走神了，你再说一遍好不好？）",
    conversation_source=None,
    index_messages=True,
    memory_eligible=True,
    label=settings.NAME,
)

CODING = Profile(
    name="coding",
    system_prompt=_coding_prompt,
    # ⚠️ read_file 必须先于 edit_file：`edit_file` 的读后写保护靠 read_file 登记。
    core_tools=("read_file", "edit_file", "write_file", "list_dir", "grep_search",
                "run_shell", "todo_write", "search_tools", "compress_context"),
    # ⚠️ **不含 "memory"**：编码 Profile 不该检索到 retrieve_memory / summarize_today，
    # 也不该读陪伴攒下的私人记忆。
    packs=("fs", "code", "exec", "vcs", "plan", "web", "meta"),
    workspace_roots=(settings.CODING_WORKSPACE,),
    # 默认**不许改自己**：把 loop.py 改坏之后下一次就启动不起来，而那时正想让它修回来。
    protected_paths=() if settings.ALLOW_SELF_EDIT else ("agent",),
    approval=settings.APPROVAL_POLICY,
    max_steps=settings.CODING_MAX_STEPS,
    max_output_tokens=settings.CODING_MAX_OUTPUT_TOKENS,
    fallback_reply="（本回合步数用尽了，任务可能还没完成。）",
    # 编码会话**入库但隔离**：能跨进程续接、能压缩，但不进向量索引 / 长期记忆。
    conversation_source="coding",
    index_messages=False,
    memory_eligible=False,
    label="coding",
)

ALL: tuple[Profile, ...] = (COMPANION, CODING)
BY_NAME = {p.name: p for p in ALL}


# ---------------------------------------------------------------- 激活机制

_active: ContextVar[Profile] = ContextVar("dsh_active_profile", default=COMPANION)


def active() -> Profile:
    """当前执行上下文里的 Profile。默认 COMPANION（保证老调用行为不变）。"""
    return _active.get()


def activate(profile: Profile) -> Token:
    """激活一个 Profile，返回用于还原的 token。**只有 `loop.ask()` 该调它。**"""
    return _active.set(profile)


def deactivate(token: Token) -> None:
    _active.reset(token)


def with_workspace(profile: Profile, workspace: str | Path) -> Profile:
    """派生一个工作区被覆盖的 Profile（`--workspace` 用）。

    用 `dataclasses.replace` 派生而不是就地改：`CODING` 是模块级单例，就地改会
    污染同进程里的其它调用。
    **目录必须已存在**，不自动创建：`--workspace D:\\typo` 应当场报错。
    """
    from dataclasses import replace

    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"--workspace 指向的不是一个存在的目录：{root}")
    return replace(profile, workspace_roots=(root,), label=f"{profile.label} @ {root}")
