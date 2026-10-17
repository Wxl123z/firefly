"""审批策略（HITL）：哪些操作要问人、怎么问、拒绝了怎么回。

一个会编造的模型 + 没有审批的写工具 = 数据损坏，所以写操作默认要过人这一关。

- 会让进程**永远回不来**的（`vim` / `gdb` / `tail -f` / 裸解释器进 REPL）→
  **直接拒绝**：拿审批去问它没有意义，用户点了 y 进程照样挂住。这类在工具层
  就拒掉（见 `exec.py` 的 `BLOCKED`），到不了这里。
- 不可逆或高风险的（`rm -rf` / `git push --force` / `git reset --hard`）→ 问一句。

三档策略：`AUTO` 全自动不问（演示 / CI）；`DESTRUCTIVE` **只问 destructive**
（默认，相当于危险黑名单）；`ALL_WRITES` 任何非只读操作都问（相当于只读白名单）。
"""
from __future__ import annotations

import enum
import sys
from dataclasses import dataclass
from typing import Callable


class ApprovalPolicy(str, enum.Enum):
    AUTO = "auto"
    DESTRUCTIVE = "destructive"
    ALL_WRITES = "all_writes"


class Risk(str, enum.Enum):
    """操作的风险等级。**由工具自己判定** —— 它最清楚自己要干什么。"""
    SAFE = "safe"                # 只读，无副作用
    WRITE = "write"              # 改状态，但可逆、影响面小
    DESTRUCTIVE = "destructive"  # 不可逆，或影响面大


@dataclass
class ApprovalRequest:
    tool: str
    key: str                     # 稳定标识"这一类操作"（用于"本会话总是允许"）
    summary: str                 # 给人看的一句话
    detail: str = ""             # 更详细的内容（命令原文 / diff）
    reason: str = ""             # 为什么被判定需要审批


@dataclass
class ApprovalResult:
    allowed: bool
    reason: str = ""             # 拒绝时**回灌给模型**的话（不是异常）


# ---------------------------------------------------------------- 交互

# "本会话总是允许"的记忆：{(tool, key)}
_always: set[tuple[str, str]] = set()


def reset() -> None:
    """会话重置时清空"总是允许"的记忆。"""
    _always.clear()


def _terminal_asker(req: ApprovalRequest) -> bool:
    """终端交互。**非交互环境（管道 / EOF）默认拒绝** —— fail closed。"""
    if not sys.stdin or not sys.stdin.isatty():
        print("  （非交互环境，默认拒绝）")
        return False
    print()
    print("─" * 68)
    print(f"⚠️ 需要确认：{req.summary}")
    if req.reason:
        print(f"   原因：{req.reason}")
    if req.detail:
        for line in req.detail.splitlines()[:24]:
            print(f"   │ {line}")
        extra = req.detail.count("\n") + 1 - 24
        if extra > 0:
            print(f"   │ …（还有 {extra} 行）")
    print("─" * 68)
    try:
        ans = input("  [y] 允许一次   [a] 本会话总是允许   [n] 拒绝  > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if ans in ("a", "all", "总是"):
        _always.add((req.tool, req.key))
        return True
    return ans in ("y", "yes", "是")


#: 可替换的"问人"实现。测试与非交互场景换成别的。
asker: Callable[[ApprovalRequest], bool] = _terminal_asker


def use_asker(fn: Callable[[ApprovalRequest], bool] | None) -> None:
    """替换/还原问人实现（测试用）。"""
    global asker
    asker = fn or _terminal_asker


# ---------------------------------------------------------------- 判定

def needs_approval(risk: Risk, policy: ApprovalPolicy) -> bool:
    if policy is ApprovalPolicy.AUTO:
        return False
    if policy is ApprovalPolicy.ALL_WRITES:
        return risk is not Risk.SAFE
    return risk is Risk.DESTRUCTIVE


def require(summary: str, *, risk: Risk, tool: str, key: str,
            detail: str = "", reason: str = "",
            policy: ApprovalPolicy = ApprovalPolicy.DESTRUCTIVE) -> ApprovalResult:
    """按策略决定要不要问，要问就问。返回能不能做。

    拒绝**不是异常**：回灌一段可读的话，让模型自己决定是解释、换做法、还是放弃。
    """
    if not needs_approval(risk, policy):
        return ApprovalResult(True)
    if (tool, key) in _always:
        return ApprovalResult(True)

    req = ApprovalRequest(tool=tool, key=key, summary=summary,
                          detail=detail, reason=reason)
    if asker(req):
        return ApprovalResult(True)
    return ApprovalResult(
        False,
        reason=(f"用户拒绝了这次操作（{summary}）。\n"
                f"不要重试同一个操作。可以：说明你为什么要这么做、"
                f"换一种影响更小的做法、或者直接问用户想怎么办。"))
