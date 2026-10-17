"""代码检索工具包（code）：按文件名找、按内容搜、按符号看结构。

- `glob_files`：路径模式匹配，找有哪些文件叫这个名字
- `grep_search`：正则扫内容，找哪里出现了这段文字
- `code_symbols`：用 Python `ast` 看定义了哪些类和函数

**`code_symbols` 只做 Python**：动态属性、`getattr`、跨文件类型推断都看不了。
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from tools.registry import ToolOutcome, tool
from tools.sandbox import relative, resolve

# ---------------------------------------------------------------- 通用护栏

MAX_FILES = 5000            # 一次最多扫多少文件
MAX_FILE_BYTES = 1_000_000  # 超过这个大小的文件跳过
MAX_TOTAL_BYTES = 200 * 1024 * 1024
SNIFF_BYTES = 4096

# 硬编码忽略（.gitignore 之外总要挡掉的）
HARD_IGNORE = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist",
               "build", ".idea", ".vscode", ".mypy_cache", ".pytest_cache",
               ".ruff_cache", "site-packages", ".tox", ".next", "target"}


# ---------------------------------------------------------------- glob / gitignore

def _glob_to_regex(pat: str) -> str:
    """glob → 正则。`*` 不跨 `/`，`**` 跨。"""
    out: list[str] = []
    i, n = 0, len(pat)
    while i < n:
        c = pat[i]
        if c == "*":
            if pat.startswith("**/", i):
                out.append("(?:.*/)?")      # `**/x` 里的 `**/` 还允许零层
                i += 3
            elif pat.startswith("**", i):
                out.append(".*")
                i += 2
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = i + 1
            if j < n and pat[j] in "!^":
                j += 1
            if j < n and pat[j] == "]":
                j += 1
            while j < n and pat[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape("["))
                i += 1
            else:
                inner = pat[i + 1:j]
                if inner.startswith("!"):
                    inner = "^" + inner[1:]
                out.append("[" + inner + "]")
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


@dataclass
class GitIgnore:
    """**极简** .gitignore 匹配器：支持 `#` 注释、空行、`!` 取反、结尾 `/`（只匹配目录）、
    开头或中间的 `/`（锚定到根）、`*`（不跨 `/`）、`**`、`?`、`[...]`。
    不支持嵌套 `.gitignore`、全局排除文件、反斜杠转义、`\\` 续行 —— 静默按"不忽略"处理。
    """
    rules: list[tuple[re.Pattern, bool, bool]] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> "GitIgnore":
        gi = cls()
        f = root / ".gitignore"
        if not f.is_file():
            return gi
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return gi
        for raw in text.splitlines():
            line = raw.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            negate = line.startswith("!")
            if negate:
                line = line[1:]
            dir_only = line.endswith("/")
            if dir_only:
                line = line[:-1]
            anchored = "/" in line
            if line.startswith("/"):
                line = line[1:]
            if not line:
                continue
            body = _glob_to_regex(line)
            # 命中目录时，里面的东西也该被忽略 → 尾部允许 `(?:/.*)?`
            rx = (f"^{body}(?:/.*)?$" if anchored
                  else f"(?:^|/){body}(?:/.*)?$")
            gi.rules.append((re.compile(rx), negate, dir_only))
        return gi

    def ignored(self, rel_posix: str, is_dir: bool) -> bool:
        """后面的规则覆盖前面的（和 git 一致）。"""
        verdict = False
        for rx, negate, dir_only in self.rules:
            if dir_only and not is_dir:
                continue
            if rx.search(rel_posix):
                verdict = not negate
        return verdict


def _iter_files(root: Path, ignore: GitIgnore, stats: "WalkStats",
                *, pattern: str | None = None) -> Iterator[tuple[Path, str]]:
    """遍历 root 下的文件，产出 (绝对路径, 相对 posix 路径)。边遍历边剪枝。
    不按扩展名过滤 —— 按后缀筛会静默漏掉 `Makefile`、`Dockerfile` 这类文件；
    该挡的是二进制和超大文件（各有护栏和计数）。
    """
    rx = re.compile(_glob_to_regex(pattern) + "$") if pattern else None
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        rel_dir = d.relative_to(root).as_posix() if d != root else ""
        # 剪枝：把要跳过的子目录从 dirnames 里摘掉（os.walk 就不会进去）
        keep = []
        for name in dirnames:
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if name in HARD_IGNORE or ignore.ignored(rel, True):
                continue
            keep.append(name)
        dirnames[:] = sorted(keep)

        for name in sorted(filenames):
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if name in HARD_IGNORE or ignore.ignored(rel, False):
                continue
            p = d / name
            if rx is not None and not rx.search(rel) and not rx.search(name):
                continue
            stats.scanned += 1
            if stats.scanned > MAX_FILES:
                stats.truncated = True
                return
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > MAX_FILE_BYTES:
                stats.skipped_big += 1
                continue
            total_bytes += size
            if total_bytes > MAX_TOTAL_BYTES:
                stats.truncated = True
                return
            yield p, rel


@dataclass
class WalkStats:
    scanned: int = 0
    skipped_big: int = 0
    skipped_binary: int = 0
    truncated: bool = False

    def note(self) -> str:
        bits = []
        if self.skipped_big:
            bits.append(f"跳过 {self.skipped_big} 个大文件（>{MAX_FILE_BYTES // 1000}KB）")
        if self.skipped_binary:
            bits.append(f"跳过 {self.skipped_binary} 个二进制文件")
        if self.truncated:
            bits.append(f"⚠️ 达到上限（{MAX_FILES} 个文件）后停止扫描，结果**不完整**")
        return f"（已扫描 {self.scanned} 个文件" + ("；" + "；".join(bits) if bits else "") + "）"


def _is_binary(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            return b"\x00" in f.read(SNIFF_BYTES)
    except OSError:
        return True


def _read_text(p: Path) -> str:
    try:
        return p.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return ""


# ---------------------------------------------------------------- glob_files

@tool(
    name="glob_files",
    description=("按文件名/路径模式找文件，例如 `**/*.py`、`agent/tools/*.py`、`*.md`。"
                 "结果按修改时间倒序（最近改的更可能是相关的）。"
                 "想按内容找用 grep_search。"),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string",
                        "description": "glob 模式。`*` 不跨目录，`**` 跨目录"},
            "path": {"type": "string", "description": "在哪个目录下找，默认工作区根",
                     "default": "."},
            "limit": {"type": "integer", "description": "最多返回多少条，默认 100",
                      "default": 100},
        },
        "required": ["pattern"],
    },
    pack="code",
)
def glob_files(pattern: str, path: str = ".", limit: int = 100) -> ToolOutcome:
    root = resolve(path, must_exist=True)
    if not root.is_dir():
        return ToolOutcome(content=f"{relative(root)} 不是目录。", ok=False)
    if not pattern.strip():
        return ToolOutcome(content="错误：pattern 不能为空。比如 `**/*.py`。", ok=False)

    ignore = GitIgnore.load(root)
    stats = WalkStats()
    hits: list[tuple[float, str, int]] = []
    for p, rel in _iter_files(root, ignore, stats, pattern=pattern):
        try:
            st = p.stat()
        except OSError:
            continue
        hits.append((st.st_mtime, rel, st.st_size))

    hits.sort(reverse=True)
    limit = max(1, int(limit))
    shown = hits[:limit]
    if not shown:
        return ToolOutcome(content=f"没找到匹配 `{pattern}` 的文件。{stats.note()}")

    lines = []
    for mtime, rel, size in shown:
        ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"{rel}  ({size:,} 字节, {ts})")
    more = "" if len(hits) <= limit else f"\n…（还有 {len(hits) - limit} 个未显示）"
    return ToolOutcome(content=(f"匹配 `{pattern}` 的文件（{len(hits)} 个，"
                                f"按修改时间倒序）：\n" + "\n".join(lines) + more
                                + f"\n{stats.note()}"))


# ---------------------------------------------------------------- grep_search

@tool(
    name="grep_search",
    description=("按内容正则搜索工作区里的文件。找「哪里用到了 X」「这个字符串出现在哪」时调用。"
                 "想按文件名找用 glob_files；想看某个文件的结构用 code_symbols。"),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "正则表达式"},
            "path": {"type": "string", "description": "在哪个目录下搜，默认工作区根",
                     "default": "."},
            "glob": {"type": "string",
                     "description": "只看匹配这个 glob 的文件，例如 `*.py`、`**/*.md`"},
            "output_mode": {"type": "string", "enum": ["content", "files", "count"],
                            "description": "content 带上下文行 / files 只列文件名 / count 只报个数",
                            "default": "content"},
            "ignore_case": {"type": "boolean", "description": "忽略大小写，默认否",
                            "default": False},
            "context_lines": {"type": "integer", "description": "每条命中前后各带几行，默认 0",
                              "default": 0},
            "head_limit": {"type": "integer", "description": "最多返回多少条命中，默认 100",
                           "default": 100},
        },
        "required": ["pattern"],
    },
    pack="code",
)
def grep_search(pattern: str, path: str = ".", glob: str | None = None,
                output_mode: str = "content", ignore_case: bool = False,
                context_lines: int = 0, head_limit: int = 100) -> ToolOutcome:
    """实现只有一条路径：内置 Python 扫描（剪枝之后单次搜索约 0.02–0.09 秒）。若要把
    ripgrep 加回来，必须显式选后端并写明 —— 不能让"装没装 rg"静默改变查询结果。
    """
    root = resolve(path, must_exist=True)
    if not pattern:
        return ToolOutcome(content="错误：pattern 不能为空。", ok=False)
    try:
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return ToolOutcome(content=f"错误：正则写错了 —— {exc}。"
                                   f"注意特殊字符要转义（`\\.` 匹配点号，`\\(` 匹配括号）。",
                           ok=False)

    head_limit = max(1, int(head_limit))
    context_lines = max(0, int(context_lines))
    return _grep_impl(root, rx, pattern, glob=glob, output_mode=output_mode,
                      context_lines=context_lines, head_limit=head_limit)


def _scope_map(path: Path) -> dict[int, str]:
    """行号 → 最内层定义名。只对 .py 有效，解析不了就返回空表。
    附在 grep 结果里，模型才不会在"这行在哪个函数里"这类问题上编造函数名。
    """
    if path.suffix.lower() != ".py":
        return {}
    try:
        tree = ast.parse(_read_text(path), filename=str(path))
    except (SyntaxError, ValueError, RecursionError):
        return {}
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(node, "end_lineno", None) or node.lineno
            spans.append((node.lineno, end, node.name))
    out: dict[int, str] = {}
    for lineno in range(1, (max((e for _, e, _ in spans), default=0)) + 1):
        inside = [(e - s, name) for s, e, name in spans if s <= lineno <= e]
        if inside:
            out[lineno] = min(inside)[1]
    return out


def _grep_impl(root: Path, rx: re.Pattern, pattern: str, *, glob: str | None,
               output_mode: str, context_lines: int,
               head_limit: int) -> ToolOutcome:
    ignore = GitIgnore.load(root)
    stats = WalkStats()
    grep_rx = re.compile(_glob_to_regex(glob) + "$") if glob else None

    hits: list[str] = []
    files_hit: list[str] = []
    total = 0
    for p, rel in _iter_files(root, ignore, stats, pattern=None):
        if grep_rx is not None and not grep_rx.search(rel) and not grep_rx.search(p.name):
            continue
        if _is_binary(p):
            stats.skipped_binary += 1
            continue
        text = _read_text(p)
        if pattern not in text and not rx.search(text):
            continue        # 便宜的预筛：纯字面量能挡掉绝大多数文件
        lines = text.splitlines()
        file_hits = 0
        scopes = _scope_map(p) if output_mode == "content" else {}
        for i, line in enumerate(lines):
            if not rx.search(line):
                continue
            file_hits += 1
            total += 1
            if output_mode == "content" and len(hits) < head_limit:
                # 附上"这行属于哪个定义"——grep 本身给不了，而模型会因此瞎猜
                where = scopes.get(i + 1)
                line_txt = line if len(line) <= 160 else line[:160] + "…"
                suffix = f"    «在 {where} 里»" if where else ""
                if context_lines:
                    lo = max(0, i - context_lines)
                    hi = min(len(lines), i + context_lines + 1)
                    for j in range(lo, hi):
                        mark = ":" if j == i else "-"
                        ctx_suffix = suffix if j == i else ""
                        hits.append(f"{rel}{mark}{j + 1}{mark}{lines[j]}{ctx_suffix}")
                    hits.append("--")
                else:
                    hits.append(f"{rel}:{i + 1}:{line_txt}{suffix}")
        if file_hits:
            files_hit.append(rel)
            if output_mode == "files" and len(files_hit) >= head_limit:
                break

    head = f"命中 `{pattern}`"
    if not total:
        return ToolOutcome(content=f"没有命中 `{pattern}`。{stats.note()}")

    if output_mode == "files":
        body = "\n".join(files_hit[:head_limit])
        more = "" if len(files_hit) <= head_limit else f"\n…（还有 {len(files_hit) - head_limit} 个）"
        return ToolOutcome(content=f"{head}：{total} 处，涉及 {len(files_hit)} 个文件\n"
                                   f"{body}{more}\n{stats.note()}")
    if output_mode == "count":
        return ToolOutcome(content=f"{head}：{total} 处，涉及 {len(files_hit)} 个文件\n"
                                   f"{stats.note()}")

    body = "\n".join(hits[:head_limit * (1 + 2 * max(context_lines, 1))])
    more = (f"\n…（命中 {total} 处，只显示了前 {len(hits)} 行，"
            f"缩小范围或调整 head_limit）" if total > len(hits) else "")
    return ToolOutcome(content=f"{head}：{total} 处，涉及 {len(files_hit)} 个文件\n"
                               f"{body}{more}\n{stats.note()}")


# ---------------------------------------------------------------- code_symbols

def _sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        return ast.unparse(node.args)
    except Exception:
        return "(…)"


def _first_doc(node) -> str:
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    return doc.strip().splitlines()[0][:70]


def _outline_file(path: Path, rel: str) -> list[str] | str:
    """返回渲染好的行；解析失败则返回错误说明字符串。"""
    try:
        tree = ast.parse(_read_text(path), filename=str(path))
    except SyntaxError as exc:
        return f"{rel}: 解析失败（第 {exc.lineno} 行 {exc.msg}）"
    except (ValueError, RecursionError) as exc:
        return f"{rel}: 解析失败（{type(exc).__name__}）"

    rows: list[tuple[int, str, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            rows.append((node.lineno, f"{kind} {node.name}{_sig(node)}", _first_doc(node)))
        elif isinstance(node, ast.ClassDef):
            rows.append((node.lineno, f"class {node.name}", _first_doc(node)))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    kind = "async def" if isinstance(sub, ast.AsyncFunctionDef) else "def"
                    rows.append((sub.lineno, f"  {kind} {sub.name}{_sig(sub)}",
                                 _first_doc(sub)))
    if not rows:
        return f"{rel}: 没有任何顶层定义。"
    width = len(str(max(r[0] for r in rows)))
    out = [f"{rel}（{len(rows)} 个定义）"]
    for lineno, sig, doc in rows:
        out.append(f"  {lineno:>{width}}  {sig}" + (f"    # {doc}" if doc else ""))
    return out


def _collect_py(root: Path, stats: WalkStats, ignore: GitIgnore) -> list[tuple[Path, str]]:
    if root.is_file():
        return [(root, root.name)] if root.suffix == ".py" else []
    return [(p, rel) for p, rel in _iter_files(root, ignore, stats, pattern="**/*.py")]


@tool(
    name="code_symbols",
    description=("看 Python 代码的结构：outline 列出文件/目录里的类和函数（带行号）；"
                 "find 按名字找定义；references 找某个名字出现在哪些地方。"
                 "**只支持 Python**，而且只做静态语法分析 —— "
                 "动态属性、getattr、跨文件类型推断都看不出来。"),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["outline", "find", "references"],
                       "description": "outline 看结构 / find 找定义 / references 找引用"},
            "path": {"type": "string", "description": "文件或目录，默认工作区根",
                     "default": "."},
            "name": {"type": "string",
                     "description": "find / references 用的符号名，支持 `类名.方法名`"},
            "limit": {"type": "integer", "description": "最多处理多少个文件，默认 200",
                      "default": 200},
        },
        "required": ["action"],
    },
    pack="code",
)
def code_symbols(action: str, path: str = ".", name: str | None = None,
                 limit: int = 200) -> ToolOutcome:
    root = resolve(path, must_exist=True)
    ignore = GitIgnore.load(root)
    stats = WalkStats()
    files = _collect_py(root, stats, ignore)
    if not files:
        return ToolOutcome(content=f"{relative(root)} 下没有 .py 文件。", ok=False)
    files = files[:max(1, int(limit))]

    if action == "outline":
        blocks = []
        for p, rel in files:
            got = _outline_file(p, rel)
            blocks.append(got if isinstance(got, str) else "\n".join(got))
        head = (f"{relative(root)} 的结构"
                f"（{len(files)} 个 .py 文件）")
        tail = ("\n\n（用 read_file 看具体实现；用 grep_search 找某个名字在哪用到）")
        return ToolOutcome(content=f"{head}：\n\n" + "\n\n".join(blocks) + tail)

    if not name:
        return ToolOutcome(content=f"错误：action={action} 需要 name 参数。", ok=False)

    # ---- 先建一张"定义了哪些符号"的表（find 和 references 都要）----
    want_cls, _, want_meth = name.partition(".")
    defs: list[str] = []
    for p, rel in files:
        try:
            tree = ast.parse(_read_text(p), filename=str(p))
        except (SyntaxError, ValueError, RecursionError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == want_cls:
                if not want_meth:
                    defs.append(f"{rel}:{node.lineno}  class {node.name}"
                                + (f"    # {_first_doc(node)}" if _first_doc(node) else ""))
                else:
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                                and sub.name == want_meth:
                            defs.append(f"{rel}:{sub.lineno}  "
                                        f"{want_cls}.{sub.name}{_sig(sub)}")
            elif not want_meth and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == want_cls:
                defs.append(f"{rel}:{node.lineno}  def {node.name}{_sig(node)}")

    if action == "find":
        if not defs:
            return ToolOutcome(content=f"没找到名为 `{name}` 的定义。"
                                       f"（{len(files)} 个 .py 文件；只认静态定义，"
                                       f"`getattr` / 动态赋值出来的名字看不到）")
        return ToolOutcome(content=f"`{name}` 的定义：\n" + "\n".join(defs))

    # ---- references：纯文本匹配，必须自报不准 ----
    plain = name.split(".")[-1]
    rx = re.compile(rf"\b{re.escape(plain)}\b")
    hits: list[str] = []
    for p, rel in files:
        for i, line in enumerate(_read_text(p).splitlines(), 1):
            if rx.search(line):
                hits.append(f"{rel}:{i}: {line.strip()[:120]}")
                break
        if len(hits) >= 80:
            break
    note = (f"\n\n⚠️ 这是**文本匹配**，不是精确的引用分析："
            f"注释、字符串、同名局部变量都会被算进来。"
            f"真正要看调用关系请打开文件确认。")
    if not hits:
        return ToolOutcome(content=f"没找到 `{name}` 的任何出现。{note}")
    head = (f"`{name}` 出现在 {len(hits)} 个文件里"
            + (f"（定义在：{defs[0]}）" if defs else "（没找到它的定义）"))
    return ToolOutcome(content=head + "：\n" + "\n".join(hits) + note)
