"""文件工具包（fs）：读 / 写 / 精确编辑 / 列目录。

三个关键约束：
1. **读文件带行号和行分页**，不按字符截断 —— 字符截断会把一行切两半，
   而行号给了模型和错误信息一个共同坐标。
2. **精确编辑用内容寻址（`old_string` → `new_string`），不做模糊匹配**：
   匹配不上就报错，绝不猜位置。
3. **读后写保护**：没在本会话读过、或读过之后被改过的文件，拒绝编辑。
"""
from __future__ import annotations

import ast
import difflib
from pathlib import Path

from core import readset
from tools.registry import ToolOutcome, tool
from tools.sandbox import SandboxError, assert_writable, relative, resolve

_MAX_READ_LINES = 2000          # 单次最多读多少行
_MAX_LINE_CHARS = 2000          # 单行超过这个长度就截断（防 minified 文件一行吃掉全部预算）
_SNIFF_BYTES = 8192             # 判二进制时看前多少字节
_DIFF_LINES = 12                # 编辑成功时最多展示多少行 diff


# ---------------------------------------------------------------- 底层读写

class _ReadError(Exception):
    """文件读不了（给模型看的理由在消息里）。"""


def _read_bytes(path: Path) -> bytes:
    if not path.exists():
        raise _ReadError(f"{path.name} 不存在。")
    if path.is_dir():
        raise _ReadError(f"{path.name} 是目录，不是文件。可以用 list_dir 看里面有什么。")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise _ReadError(f"读不了 {path.name}：{exc}")
    if b"\x00" in raw[:_SNIFF_BYTES]:
        raise _ReadError(f"{path.name} 看起来是二进制文件（前 {_SNIFF_BYTES} 字节里有 NUL），"
                         f"不能当文本处理。")
    return raw


def _decode(raw: bytes) -> tuple[str, str]:
    """bytes → 文本。返回 (文本, 警告)。用 `raw.decode()` 而非 `Path.read_text()`（后者会把 CRLF 翻成 LF）。"""
    text = raw.decode("utf-8", errors="replace")
    bad = text.count("\ufffd")
    warn = ""
    if bad:
        warn = (f"\n⚠️ 这个文件可能不是 UTF-8（有 {bad} 个字节解不出来，已用 U+FFFD 占位）。"
                f"下面那些位置的内容**不可信**，不要照抄。")
    return text, warn


def _newline_style(text: str) -> str:
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _match_newlines(needle: str, style: str) -> str:
    """把模型给的字符串的行尾，对齐到文件实际的行尾风格。这不是模糊匹配，而是无损、
    确定的标准化（内容仍须逐字相同）。不做这步，模型在 CRLF 文件上会 100% 匹配失败。
    """
    needle = needle.replace("\r\n", "\n")
    return needle.replace("\n", "\r\n") if style == "\r\n" else needle


def _display(text: str) -> list[str]:
    """把内容拆成"给人看的行"（统一行尾，再 split）。"""
    return text.replace("\r\n", "\n").split("\n")


def _numbered(lines: list[str], start: int, total: int) -> str:
    width = len(str(max(total, 1)))
    out = []
    for i, line in enumerate(lines, start=start):
        if len(line) > _MAX_LINE_CHARS:
            extra = len(line) - _MAX_LINE_CHARS
            line = line[:_MAX_LINE_CHARS] + f"  …（本行还有 {extra} 字符）"
        out.append(f"{i:>{width}}\t{line}")
    return "\n".join(out)


def _tail_trim(lines: list[str]) -> list[str]:
    """末尾换行会 split 出一个空串，那不是真的"第 N 行"。"""
    return lines[:-1] if len(lines) > 1 and lines[-1] == "" else lines


# ---------------------------------------------------------------- read_file

@tool(
    name="read_file",
    description=("读一个文本文件，带行号。想看文件里写了什么、或在编辑之前确认原文时调用。"
                 "文件很长时用 offset/limit 翻页，返回里会告诉你还剩多少行。"),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作目录的路径"},
            "offset": {"type": "integer", "description": "从第几行开始（从 1 数），默认 1",
                       "default": 1},
            "limit": {"type": "integer",
                      "description": f"最多读多少行，默认 {_MAX_READ_LINES}",
                      "default": _MAX_READ_LINES},
        },
        "required": ["path"],
    },
    pack="fs",
)
def read_file(path: str, offset: int = 1, limit: int = _MAX_READ_LINES) -> ToolOutcome:
    target = resolve(path, must_exist=True)
    try:
        raw = _read_bytes(target)
    except _ReadError as exc:
        return ToolOutcome(content=f"错误：{exc}", ok=False)

    text, warn = _decode(raw)
    lines = _tail_trim(_display(text))
    total = len(lines)

    offset = max(1, int(offset))
    limit = max(1, int(limit))
    start = min(offset, total) if total else 1
    chunk = lines[start - 1:start - 1 + limit]
    end = start + len(chunk) - 1 if chunk else start - 1

    rel = relative(target)
    if not chunk:
        return ToolOutcome(content=f"{rel} 是空的（0 行）。{warn}")

    body = f"{rel}（共 {total} 行，显示 {start}-{end}）：\n{_numbered(chunk, start, total)}"
    if end < total:
        body += f"\n\n（还有 {total - end} 行，用 offset={end + 1} 继续读）"

    # 读全了才算"见过"。分页只读到一部分时**不标记** —— "没看全就编辑"正是要挡的情况。
    if start == 1 and end == total and not warn:
        readset.mark_read(target)

    return ToolOutcome(content=body + warn)


# ---------------------------------------------------------------- list_dir

@tool(
    name="list_dir",
    description="列出工作目录（或其子目录）里的文件与文件夹。",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作目录的路径，默认根目录",
                     "default": "."},
        },
        "required": [],
    },
    pack="fs",
)
def list_dir(path: str = ".") -> ToolOutcome:
    target = resolve(path, must_exist=True)
    if not target.is_dir():
        return ToolOutcome(content=f"{relative(target)} 不是目录。", ok=False)
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    if not entries:
        return ToolOutcome(content=f"{relative(target)} 是空的。")
    lines = []
    for p in entries[:200]:
        if p.is_dir():
            lines.append(f"[目录] {p.name}/")
        else:
            lines.append(f"[文件] {p.name}  ({p.stat().st_size} 字节)")
    more = "" if len(entries) <= 200 else f"\n…（还有 {len(entries) - 200} 项未显示）"
    return ToolOutcome(content=f"{relative(target)} 内容：\n" + "\n".join(lines) + more)


# ---------------------------------------------------------------- edit_file

def _closest_region(needle_lines: list[str], file_lines: list[str],
                    *, window: int = 3) -> tuple[int, int] | None:
    """找文件里与 needle 最像的一段（返回 [start, end)，0-based）。
    用最长公共**连续**块当锚点再向两边扩一点。代价 O(n·m)：文件太大就返回 None。
    """
    if not needle_lines or not file_lines:
        return None
    if len(needle_lines) * len(file_lines) > 4_000_000:
        return None
    sm = difflib.SequenceMatcher(None, needle_lines, file_lines, autojunk=False)
    m = sm.find_longest_match(0, len(needle_lines), 0, len(file_lines))
    if m.size < 2:              # 只对上 1 行不算线索，噪声太大
        return None
    return max(0, m.b - window), min(len(file_lines), m.b + m.size + window)


def _whitespace_insensitive_hits(needle_lines: list[str],
                                 file_lines: list[str]) -> list[int]:
    """把两边每行都 strip 掉再找，返回命中处起始行号（0-based）。只用来解释失败原因，绝不执行替换。"""
    n = len(needle_lines)
    if n == 0:
        return []
    probe = [ln.strip() for ln in needle_lines]
    stripped = [ln.strip() for ln in file_lines]
    return [i for i in range(len(stripped) - n + 1)
            if stripped[i:i + n] == probe]


@tool(
    name="edit_file",
    description=("在文件里做精确替换：把 old_string 换成 new_string。"
                 "old_string 必须与文件里的原文**逐字一致**（含缩进），"
                 "并且在文件里只出现一次 —— 否则会报错，不会猜你想改哪一处。"
                 "改之前必须先 read_file 看过这个文件。"),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作目录的路径"},
            "old_string": {"type": "string",
                           "description": "要被替换的原文，必须与文件里的逐字一致（含缩进）。"
                                          "带上前后各一行上下文可以保证唯一"},
            "new_string": {"type": "string",
                           "description": "替换成的新内容；空字符串表示删除这段"},
            "replace_all": {"type": "boolean",
                            "description": "old_string 出现多次时是否全部替换，默认否",
                            "default": False},
        },
        "required": ["path", "old_string", "new_string"],
    },
    pack="fs",
)
def edit_file(path: str, old_string: str, new_string: str,
              replace_all: bool = False) -> ToolOutcome:
    target = resolve(path, must_exist=True)
    rel = relative(target)

    # ---- 前置拒绝。每一条都给"下一步怎么做"，而不是只说不行 ----
    if old_string == "":
        return ToolOutcome(
            content="错误：old_string 不能为空（空字符串会匹配到文件开头）。"
                    "想新建文件用 write_file；想插入内容，把插入点前后的原文一起写进 old_string。",
            ok=False)
    if old_string == new_string:
        return ToolOutcome(content="错误：old_string 和 new_string 一模一样，这次编辑什么都不会改。"
                                   "如果只是想确认内容，用 read_file。", ok=False)

    # 顺序要紧：写保护排在读后写检查前面 —— 否则对一个根本不许写的路径会先提示
    # "先去 read_file"，模型跑完一趟读回来还是不能写。
    try:
        assert_writable(target)
    except SandboxError as exc:
        return ToolOutcome(content=f"错误：{exc}", ok=False)

    reason = readset.stale_reason(target)
    if reason:
        return ToolOutcome(content=f"错误：{reason}", ok=False)

    try:
        raw = _read_bytes(target)
    except _ReadError as exc:
        return ToolOutcome(content=f"错误：{exc}", ok=False)
    text, warn = _decode(raw)

    style = _newline_style(text)
    needle = _match_newlines(old_string, style)
    repl = _match_newlines(new_string, style)

    hits = text.count(needle)
    if hits == 0:
        return ToolOutcome(content=_not_found_message(rel, text, needle) + warn, ok=False)
    if hits > 1 and not replace_all:
        return ToolOutcome(content=_ambiguous_message(rel, text, needle, hits), ok=False)

    new_text = text.replace(needle, repl)
    start_line = text[:text.find(needle)].count("\n") + 1

    try:
        target.write_bytes(new_text.encode("utf-8"))
    except OSError as exc:
        return ToolOutcome(content=f"错误：写不了 {rel}：{exc}", ok=False)

    # 写回之后同步"已读"指纹，否则下一次编辑会被自己刚才的改动判成"被外部改过"
    readset.mark_read(target)

    old_lines = _display(needle)
    new_lines = _display(repl)
    where = f"第 {start_line} 行" if hits == 1 else f"{hits} 处（从第 {start_line} 行起）"
    out = (f"已编辑 {rel}（{where}）：{len(old_lines)} 行 → {len(new_lines)} 行\n\n"
           f"{_format_diff(old_lines, new_lines)}")

    syntax = _syntax_warning(target, new_text)
    if syntax:
        out += f"\n\n{syntax}"
    return ToolOutcome(content=out + warn)


def _not_found_message(rel: str, text: str, needle: str) -> str:
    file_lines = _display(text)
    needle_lines = _display(needle)
    parts = [f"错误：在 {rel} 里找不到 old_string（逐字比对，一字不差才算）。", ""]

    # 先看是不是"内容对得上、只是缩进/空白不同" —— 最常见的原因，能说清就别让模型猜。
    ws_hits = _whitespace_insensitive_hits(needle_lines, file_lines)
    if ws_hits:
        i = ws_hits[0]
        parts.append(f"⚠️ 但**忽略每行首尾空白后**能对上（第 {i + 1} 行起），"
                     f"所以多半是缩进或空格不一致。文件里的原文是：")
        parts.append(_numbered(file_lines[i:i + len(needle_lines)], i + 1, len(file_lines)))
        parts.append("")
        parts.append("照抄上面的缩进（**制表符和空格是不同的字符**）。")
        return "\n".join(parts)

    region = _closest_region(needle_lines, file_lines)
    if region:
        s, e = region
        parts.append(f"文件共 {len(file_lines)} 行。最相似的一段在第 {s + 1}-{e} 行：")
        parts.append(_numbered(file_lines[s:e], s + 1, len(file_lines)))
        parts.append("")
    else:
        parts.append(f"文件共 {len(file_lines)} 行，没找到明显相似的一段。")

    parts.append("常见原因：")
    parts.append("1. 缩进和文件里的不一致（**制表符 ≠ 空格**）—— 先 read_file 照抄原文")
    parts.append("2. 文件在你读过之后被改过 —— 重新 read_file 一次")
    parts.append("3. old_string 里写了文件里没有的内容（比如你记忆里的版本）")
    return "\n".join(parts)


def _ambiguous_message(rel: str, text: str, needle: str, hits: int) -> str:
    file_lines = _display(text)
    needle_lines = _display(needle)
    n = len(needle_lines)
    where: list[int] = []
    pos = 0
    while len(where) < 8:
        i = text.find(needle, pos)
        if i < 0:
            break
        where.append(text[:i].count("\n") + 1)
        pos = i + len(needle)

    parts = [f"错误：old_string 在 {rel} 里出现了 {hits} 次，无法确定你要改哪一处。",
             f"出现的行号：{', '.join(str(x) for x in where)}"
             + ("…" if hits > len(where) else ""),
             ""]
    i = where[0] - 1
    parts.append(f"第 {where[0]} 行附近长这样：")
    parts.append(_numbered(file_lines[max(0, i - 1):i + n + 1], max(1, i), len(file_lines)))
    parts.append("")
    parts.append("两个办法：")
    parts.append("· 把 old_string 加长，**带上前后各一行上下文**，让它唯一")
    parts.append("· 如果确实要全部替换，用 replace_all=true")
    return "\n".join(parts)


def _format_diff(old_lines: list[str], new_lines: list[str]) -> str:
    out = [f"- {ln}" for ln in old_lines[:_DIFF_LINES]]
    if len(old_lines) > _DIFF_LINES:
        out.append(f"- …（省略 {len(old_lines) - _DIFF_LINES} 行）")
    out += [f"+ {ln}" for ln in new_lines[:_DIFF_LINES]]
    if len(new_lines) > _DIFF_LINES:
        out.append(f"+ …（省略 {len(new_lines) - _DIFF_LINES} 行）")
    return "\n".join(out)


def _syntax_warning(target: Path, text: str) -> str:
    """改完之后 .py 文件还 parse 得动吗？不阻断、不回滚，只报告。"""
    if target.suffix.lower() != ".py":
        return ""
    try:
        ast.parse(text)
    except SyntaxError as exc:
        return (f"⚠️ 语法检查没通过：第 {exc.lineno} 行 {exc.msg}。"
                f"改动**已经写进去了** —— 如果是手误，现在改回来。")
    return ""


# ---------------------------------------------------------------- write_file

@tool(
    name="write_file",
    description=("把内容写入文件。mode=create 只用于新建（文件已存在会报错）；"
                 "mode=overwrite 整份覆盖；mode=append 追加到末尾。"
                 "⚠️ **覆盖一个已存在的文件之前必须先 read_file 看过它** —— "
                 "盲写会把里面的东西全丢掉。只改一小部分请用 edit_file。"),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作目录的路径"},
            "content": {"type": "string", "description": "要写入的完整内容"},
            "mode": {"type": "string", "enum": ["overwrite", "append", "create"],
                     "description": ("overwrite 覆盖（需先读过）/ append 追加 / "
                                     "create 仅新建，默认 overwrite"),
                     "default": "overwrite"},
        },
        "required": ["path", "content"],
    },
    pack="fs",
)
def write_file(path: str, content: str, mode: str = "overwrite") -> ToolOutcome:
    target = resolve(path)
    existed = target.exists()
    rel = relative(target)

    try:
        assert_writable(target)
    except SandboxError as exc:
        return ToolOutcome(content=f"错误：{exc}", ok=False)

    if mode == "create" and existed:
        return ToolOutcome(
            content=f"错误：{rel} 已经存在。mode=create 只用于新建 —— "
                    f"想改它请先 read_file 再 edit_file；确实要整份覆盖就用 mode=overwrite。",
            ok=False)

    # 覆盖一个已存在的文件 = 盲写，所以要求先 read_file 读过。append 是增量，不在此列。
    if mode != "append" and existed:
        reason = readset.stale_reason(target)
        if reason:
            return ToolOutcome(
                content=f"错误：{reason}\n"
                        f"（整份覆盖一个已存在的文件属于盲写 —— 先 read_file 确认它现在是什么，"
                        f"再决定是覆盖还是用 edit_file 改局部。）",
                ok=False)

    old_text = ""
    if existed:
        try:
            old_text = _decode(target.read_bytes())[0]
        except OSError as exc:
            return ToolOutcome(content=f"错误：读不了 {rel}：{exc}", ok=False)

    new_text = old_text + content if mode == "append" and existed else content

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(new_text.encode("utf-8"))
    except OSError as exc:
        return ToolOutcome(content=f"错误：写不了 {rel}：{exc}", ok=False)
    readset.mark_read(target)

    new_lines = _tail_trim(_display(new_text))
    if mode == "append" and existed:
        head = (f"已追加到 {rel}（+{len(content)} 字，"
                f"现在共 {len(new_lines)} 行）")
    elif not existed:
        head = f"已新建 {rel}（{len(new_lines)} 行）"
    else:
        head = f"已覆盖 {rel}：{len(_tail_trim(_display(old_text)))} 行 → {len(new_lines)} 行"

    syntax = _syntax_warning(target, new_text)
    return ToolOutcome(content=f"{head}\n\n{syntax}" if syntax else head)
