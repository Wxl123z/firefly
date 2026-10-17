"""文件工具沙箱：所有路径必须落在当前 Profile 的允许根（`workspace_roots`，可多个）内。

- 相对路径 → 按允许根依次尝试，优先选已经存在的那个；都不存在则用第一个根
- 绝对路径必须落在某个允许根内；`..` 穿越在 resolve 之后比对前缀校验，越界直接拒绝
- 报错时列出所有允许根（否则模型会反复试同一个越界路径）；**这里不创建目录**：
  根目录的存在性由 config / `profile.with_workspace()` 负责
"""
from pathlib import Path

from core.profile import Profile, active


class SandboxError(Exception):
    """路径越界或非法。"""


def roots(profile: Profile | None = None) -> tuple[Path, ...]:
    """当前 Profile 的允许根（已 resolve）。"""
    prof = profile if profile is not None else active()
    return tuple(Path(r).resolve() for r in prof.workspace_roots)


def workspace(profile: Profile | None = None) -> Path:
    """主根（第一个）。单根场景的简写，老的 `workspace()` 调用点行为不变。"""
    return roots(profile)[0]


def _within(raw: str, root: Path) -> Path | None:
    """把 raw 按 root 解析；不在 root 内则返回 None。"""
    t = Path(raw)
    t = (root / t).resolve() if not t.is_absolute() else t.resolve()
    return t if (t == root or root in t.parents) else None


def _roots_hint(all_roots: tuple[Path, ...]) -> str:
    return "、".join(str(r) for r in all_roots)


def resolve(path: str, *, must_exist: bool = False,
            profile: Profile | None = None) -> Path:
    """把模型给的路径解析成某个允许根内的绝对路径。"""
    raw = (path or "").strip().strip('"').strip("'")
    if not raw:
        raise SandboxError("路径为空。请给出相对工作目录的路径，例如 notes/todo.md")

    all_roots = roots(profile)

    if Path(raw).is_absolute():
        for root in all_roots:
            hit = _within(raw, root)
            if hit is None:
                continue
            if must_exist and not hit.exists():
                raise SandboxError(f"文件不存在：{relative(hit, profile)}")
            return hit
        raise SandboxError(f"路径越界：{raw} 不在任何允许的工作目录内。"
                           f"允许的根：{_roots_hint(all_roots)}")

    # 相对路径优先选"那个根下真的有这个文件"的：多根场景（仓库 + wiki）下，模型说
    # `concepts/换工作.md` 时该找真的存在的那个，而不是永远拼到第一个根下。
    first: Path | None = None
    for root in all_roots:
        hit = _within(raw, root)
        if hit is None:
            continue
        if first is None:
            first = hit
        if hit.exists():
            return hit
    if first is not None:
        if must_exist:
            raise SandboxError(f"文件不存在：{relative(first, profile)}"
                               f"（所有允许的根里都没有：{_roots_hint(all_roots)}）")
        return first

    raise SandboxError(f"路径越界：只允许访问工作目录内的文件。"
                       f"允许的根：{_roots_hint(all_roots)}")


def relative(path: Path, profile: Profile | None = None) -> str:
    """转成相对于**命中的那个根**的路径。"""
    p = Path(path)
    for root in roots(profile):
        try:
            return str(p.relative_to(root)).replace("\\", "/")
        except ValueError:
            continue
    return str(p)


def assert_writable(target: Path, profile: Profile | None = None) -> None:
    """写操作前调用。落在 Profile 的写保护路径下就抛 `SandboxError`。
    硬拒绝而非审批：这类"把自己改坏、下次就起不来"的操作不该交给一轮对话里的确认按钮。
    """
    prof = profile if profile is not None else active()
    if not prof.protected_paths:
        return
    rel = relative(target, prof)
    if prof.is_protected(rel):
        raise SandboxError(
            f"路径受保护：{rel} 在 `{'、'.join(prof.protected_paths)}` 下，当前 Profile "
            f"不允许写。这是防止 agent 改坏自己（改坏了下次就起不来了）。"
            f"确实要改请用 --allow-self-edit 或改配置 ALLOW_SELF_EDIT。")
