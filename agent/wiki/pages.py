"""Wiki 页面的读写：front-matter、页面索引、原子写。

**产出必须逐字节可重现**：front-matter 键顺序固定、列表字段排序后再写、换行统一 `\\n`、
写入用临时文件 + 原子替换、`updated` 取来源里最新的日期而不是 `now()`。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

#: front-matter 的键顺序。**固定**，不依赖 dict 插入顺序。
FM_ORDER = ("title", "type", "aliases", "sources", "updated", "confidence",
            "status", "graph-excluded")

#: 文件名里不能出现的字符（Windows 更严：`\/:*?"<>|`）
_BAD_FILENAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

#: Windows 保留设备名，避免 `con.md` 这种名字
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}

INDEX_FILE = ".index.json"


def ref_sort_key(ref: str) -> tuple:
    """锚点的排序键：`memory:9` 要排在 `memory:13` 前面（`sorted()` 是字符串序）；
    front-matter 的 `sources:` 和正文证据都用它。"""
    kind, _, num = str(ref).partition(":")
    try:
        return (kind, 0, int(num), "")
    except ValueError:
        return (kind, 1, 0, num)


@dataclass
class Evidence:
    """一条证据。渲染成 `- 2026-09-20 · [memory:1234] 事实`。"""
    ref: str            # "memory:1234" / "rollup:2026-09-20"
    date: str           # YYYY-MM-DD
    text: str

    def key(self) -> tuple:
        """排序 + 去重都用它。**这是幂等的第一道闸门。**"""
        return (self.date, ref_sort_key(self.ref))

    def render(self) -> str:
        return f"- {self.date} · [{self.ref}] {self.text}"


@dataclass
class Page:
    path: str                       # 相对 wiki/ 的 posix 路径，如 "concepts/换工作.md"
    title: str
    type: str                       # concept | entity | source | synthesis
    aliases: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    updated: str = ""               # YYYY-MM-DD
    confidence: str = "stated"      # stated | inferred | mixed
    status: str = ""
    graph_excluded: bool = False
    summary: str = ""               # 「综述」段正文
    evidence: list[Evidence] = field(default_factory=list)
    related: list[str] = field(default_factory=list)   # [[...]] 目标

    # ---------------------------------------------------------------- 渲染
    def front_matter(self) -> dict:
        data: dict = {
            "title": self.title,
            "type": self.type,
            "aliases": sorted(set(self.aliases)),
            "sources": sorted(set(self.sources), key=ref_sort_key),
            "updated": self.updated,
            "confidence": self.confidence,
        }
        if self.status:
            data["status"] = self.status
        data["graph-excluded"] = bool(self.graph_excluded)
        # 按 FM_ORDER 重排；不在表里的键排到最后
        return {k: data[k] for k in FM_ORDER if k in data} | \
               {k: v for k, v in data.items() if k not in FM_ORDER}

    def render(self) -> str:
        fm = yaml.safe_dump(self.front_matter(), allow_unicode=True,
                            sort_keys=False, default_flow_style=False,
                            width=1000).rstrip("\n")
        parts = ["---", fm, "---", "", f"# {self.title}", ""]

        parts += ["## 综述", ""]
        parts += [demote_headings(self.summary).strip()
                  or "（还没有足够的证据形成综述。）", ""]

        parts += ["## 证据", ""]
        if self.evidence:
            seen, rows = set(), []
            for ev in sorted(self.evidence, key=Evidence.key):
                if ev.key() in seen:          # 按 (日期, 锚点) 去重 —— 幂等的第二道闸门
                    continue
                seen.add(ev.key())
                rows.append(ev.render())
            parts += rows
        else:
            parts += ["（暂无证据。）"]
        parts.append("")

        parts += ["## 关联", ""]
        parts += [" · ".join(f"[[{r}]]" for r in sorted(set(self.related)))
                  if self.related else "（暂无关联。）"]
        parts.append("")
        return "\n".join(parts)


# ---------------------------------------------------------------- 解析

_FM_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?(.*)$", re.S)
_EV_RE = re.compile(r"^-\s*(\d{4}-\d{2}-\d{2})\s*·\s*\[([^\]]+)\]\s*(.*)$")


def parse(text: str, path: str = "") -> Page:
    """把一份 markdown 读回 `Page`。解析不了就抛 —— 不静默产出半成品。"""
    m = _FM_RE.match(text.lstrip("\ufeff"))
    if not m:
        raise ValueError(f"{path}: 没有 front-matter（开头必须是 `---`）")
    meta = yaml.safe_load(m.group(1)) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: front-matter 不是键值对")
    body = m.group(2)

    page = Page(
        path=path,
        title=str(meta.get("title") or ""),
        type=str(meta.get("type") or ""),
        aliases=[str(a) for a in (meta.get("aliases") or [])],
        sources=[str(s) for s in (meta.get("sources") or [])],
        updated=str(meta.get("updated") or ""),
        confidence=str(meta.get("confidence") or "stated"),
        status=str(meta.get("status") or ""),
        graph_excluded=bool(meta.get("graph-excluded")),
    )
    page.summary = _section(body, "综述")
    page.related = re.findall(r"\[\[([^\]]+)\]\]", _section(body, "关联"))
    for line in _section(body, "证据").splitlines():
        em = _EV_RE.match(line.strip())
        if em:
            page.evidence.append(Evidence(ref=em.group(2), date=em.group(1),
                                          text=em.group(3).strip()))
    return page

def demote_headings(text: str, levels: int = 2) -> str:
    """把嵌入内容里的标题降级，免得把所在小节截断。
    ⚠️ 解析靠 `^##\\s` 停小节，所以**嵌入任意 markdown 都必须先降级它的标题**，
    否则小节会被截成空的，页面被 lint 报成"空页"。"""
    out = []
    for line in (text or "").splitlines():
        m = re.match(r"^(#{1,6})(\s+)", line)
        if m:
            line = "#" * min(len(m.group(1)) + levels, 6) + line[len(m.group(1)):]
        out.append(line)
    return "\n".join(out)


def _section(body: str, name: str) -> str:
    """取 `## <name>` 到下一个 `## ` 之间的内容。"""
    m = re.search(rf"^##\s*{re.escape(name)}\s*$\r?\n(.*?)(?=^##\s|\Z)",
                  body, re.S | re.M)
    if not m:
        return ""
    txt = m.group(1).strip()
    return "" if txt.startswith("（") and txt.endswith("）") else txt


# ---------------------------------------------------------------- 文件名

def slug(title: str) -> str:
    """标题 → 文件名。中文照留（Wiki 是给人看的），只清掉非法字符。"""
    name = _BAD_FILENAME.sub("-", (title or "").strip()).strip(". ")
    name = re.sub(r"\s+", " ", name)
    if not name:
        name = "untitled"
    if name.lower() in _RESERVED:
        name += "_"
    return name[:80]


def page_path(page_type: str, title: str) -> str:
    from wiki.contract import PAGE_TYPES
    folder = PAGE_TYPES.get(page_type, "misc")
    return f"{folder}/{slug(title)}.md"


# ---------------------------------------------------------------- 原子写

def write_text(path: Path, text: str) -> None:
    """原子写：先写临时文件再替换，避免中途崩了留下**半个文件**（front-matter 会解析失败）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_page(root: Path, page: Page) -> None:
    write_text(root / page.path, page.render())


def read_page(root: Path, rel: str) -> Page:
    return parse((root / rel).read_text(encoding="utf-8"), rel)


# ---------------------------------------------------------------- 页面索引

def load_index(root: Path) -> dict:
    """页面清单。**存在 Wiki 侧，不进 DB**。"""
    f = root / INDEX_FILE
    if not f.is_file():
        return {"pages": {}}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"pages": {}}
    return data if isinstance(data, dict) and "pages" in data else {"pages": {}}


def save_index(root: Path, index: dict) -> None:
    write_text(root / INDEX_FILE,
               json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def rebuild_index(root: Path) -> dict:
    """从磁盘上的页面文件重建索引。`--rebuild` 之后 / 索引丢了时用。"""
    pages: dict[str, dict] = {}
    for f in sorted(root.rglob("*.md")):
        rel = f.relative_to(root).as_posix()
        if rel.startswith("templates/") or rel in ("index.md", "log.md",
                                                   "overview.md", "QUESTIONS.md"):
            continue
        try:
            p = parse(f.read_text(encoding="utf-8"), rel)
        except (ValueError, OSError):
            continue
        pages[rel] = {"title": p.title, "type": p.type, "aliases": p.aliases,
                      "sources": p.sources, "updated": p.updated}
    return {"pages": pages}


def all_pages(root: Path, index: dict) -> list[str]:
    """索引里登记过的页面路径（排序，保证确定性）。"""
    return sorted((index.get("pages") or {}).keys())
