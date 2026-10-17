"""Wiki 一致性检查，九项：
1 死链 · 2 孤儿页（没有任何页链接到它，`index.md` 除外）· 3 未登记 / 4 幽灵条目
（`index.md` 与磁盘文件对不上）· 5 空页 · 6 别名冲突 · 7 陈旧页（所有来源记忆都
`historical` 了而页面没标注）· 8 悬空锚点（`sources:` 里的 `memory:<id>` 库里没有）·
9 计数不符（`overview.md` 的统计与实际不符）。

⚠️ **lint 必须能失败**：`t_wiki_lint` 会往临时 Wiki 注入坏页再要求它报出来 ——
一个从不报错的检查器会让一份坏掉的 Wiki 看起来是好的。

**lint 只报告，不自动修**：自动修会让"检查通过"变得廉价，规则有 bug 时还会批量破坏正确的页。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from store.db import Memory, MemoryStatus, SessionLocal
from wiki import pages as P

#: 不参与"孤儿页"和"未登记"检查的元页面
META_PAGES = {"index.md", "log.md", "overview.md", "QUESTIONS.md"}

LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")


@dataclass
class Issue:
    check: str
    page: str
    detail: str


@dataclass
class LintReport:
    issues: list[Issue] = field(default_factory=list)
    pages_checked: int = 0

    def by_check(self) -> dict[str, list[Issue]]:
        out: dict[str, list[Issue]] = {}
        for i in self.issues:
            out.setdefault(i.check, []).append(i)
        return out

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        if not self.issues:
            return f"✅ 没有问题（检查了 {self.pages_checked} 页）"
        parts = " · ".join(f"{k} {len(v)}" for k, v in sorted(self.by_check().items()))
        return f"⚠️ {len(self.issues)} 项问题（{parts}）"

    def to_dict(self) -> dict:
        return {"checked_at": datetime.now().isoformat(timespec="seconds"),
                "pages_checked": self.pages_checked,
                "issue_count": len(self.issues),
                "by_check": {k: len(v) for k, v in sorted(self.by_check().items())},
                "issues": [{"check": i.check, "page": i.page, "detail": i.detail}
                           for i in self.issues]}


# ---------------------------------------------------------------- 检查

def _content_pages(root: Path) -> dict[str, P.Page]:
    """所有内容页（排除元页面），路径 → Page。解析不了的**也记下来**。"""
    out: dict[str, P.Page] = {}
    for f in sorted(root.rglob("*.md")):
        rel = f.relative_to(root).as_posix()
        if rel.startswith("templates/") or rel in META_PAGES:
            continue
        try:
            out[rel] = P.parse(f.read_text(encoding="utf-8"), rel)
        except (ValueError, OSError):
            continue
    return out


def _link_targets(page: P.Page) -> set[str]:
    """这一页链接到哪些页（规范化成路径）。"""
    out = set()
    for raw in LINK_RE.findall(page.render()):
        t = raw.strip().strip("/")
        out.add(t if t.endswith(".md") else t + ".md")
    return out


def lint(wiki_root: Path, *, db_check: bool = True) -> LintReport:
    """检查一个 Wiki。

    ⚠️ `wiki_root` 是**页面所在的那一层**（`<WIKI_DIR>/wiki`），不是 `WIKI_DIR`
    本身 —— 后者下面还有 `raw/`、`outputs/`、`scripts/`。传错的话相对路径会带上
    `wiki/` 前缀，和索引里的键对不上，每页会同时被报成"孤儿页"和"未登记"。
    """
    wiki_root = Path(wiki_root)
    # 允许直接传 WIKI_DIR：自动下沉一层
    if (wiki_root / "wiki" / "index.md").is_file() or \
            (wiki_root / "wiki" / P.INDEX_FILE).is_file():
        wiki_root = wiki_root / "wiki"

    rep = LintReport()
    pages = _content_pages(wiki_root)
    rep.pages_checked = len(pages)

    index = P.load_index(wiki_root)
    registered = set((index.get("pages") or {}).keys())
    on_disk = set(pages)

    # 1 死链 / 2 孤儿页
    linked: set[str] = set()
    for rel, page in sorted(pages.items()):
        for target in sorted(_link_targets(page)):
            linked.add(target)
            if target not in on_disk:
                rep.issues.append(Issue("死链", rel, f"[[{target[:-3]}]] 指向不存在的页"))
    for rel in sorted(on_disk):
        if rel not in linked:
            # 只从"谁都没链到它"这个意义上算孤儿；index.md 不算链接来源，
            # 否则所有登记过的页都会被 index 救活。
            rep.issues.append(Issue("孤儿页", rel, "没有任何页链接到它"))

    # 3 未登记 / 4 幽灵条目
    for rel in sorted(on_disk - registered):
        rep.issues.append(Issue("未登记", rel, "文件存在但 index 里没有"))
    for rel in sorted(registered - on_disk):
        rep.issues.append(Issue("幽灵条目", rel, "index 里有但文件不存在"))

    # 5 空页
    for rel, page in sorted(pages.items()):
        if not page.evidence and not page.summary.strip():
            rep.issues.append(Issue("空页", rel, "综述和证据都空"))

    # 6 别名冲突
    alias_owner: dict[str, str] = {}
    for rel, page in sorted(pages.items()):
        for name in [page.title, *page.aliases]:
            key = name.strip().lower()
            if not key:
                continue
            prev = alias_owner.get(key)
            if prev and prev != rel:
                rep.issues.append(Issue("别名冲突", rel, f"`{name}` 已被 {prev} 占用"))
            else:
                alias_owner[key] = rel

    if db_check:
        _db_checks(rep, pages)

    # 9 计数不符
    _count_check(rep, wiki_root, pages, registered)
    return rep


def _db_checks(rep: LintReport, pages: dict[str, P.Page]) -> None:
    """7 陈旧页 / 8 悬空锚点 —— 这两项要查库。"""
    with SessionLocal() as s:
        rows = {m.id: m for m in s.execute(select(Memory)).scalars()}

    for rel, page in sorted(pages.items()):
        ids = [int(x.split(":")[1]) for x in page.sources
               if x.startswith("memory:") and x.split(":")[1].isdigit()]
        # 8 悬空锚点
        for mid in sorted(set(ids)):
            if mid not in rows:
                rep.issues.append(Issue("悬空锚点", rel,
                                        f"[memory:{mid}] 在库里不存在"))
        # 7 陈旧页
        if ids and all(rows[i].status == MemoryStatus.historical
                       for i in ids if i in rows) and "retracted" not in page.status:
            rep.issues.append(Issue("陈旧页", rel,
                                    "所有来源记忆都已被推翻，页面却没标 status: retracted"))


def _count_check(rep: LintReport, root: Path, pages: dict, registered: set) -> None:
    """9 计数不符：`overview.md` 写的数字和实际对不上。"""
    f = root / "overview.md"
    if not f.is_file():
        rep.issues.append(Issue("计数不符", "overview.md", "文件不存在"))
        return
    text = f.read_text(encoding="utf-8")
    m = re.search(r"共\s*(\d+)\s*）", text)
    if m and int(m.group(1)) != len(registered):
        rep.issues.append(Issue(
            "计数不符", "overview.md",
            f"写的是 {m.group(1)} 页，实际 {len(registered)} 页"))


# ---------------------------------------------------------------- 输出

def write_report(wiki_root: Path, rep: LintReport) -> Path:
    """报告写到 **`WIKI_DIR/outputs/`**，不在 `wiki/` 里 —— `wiki/` 是 LLM 的地盘。
    """
    wiki_root = Path(wiki_root)
    out_dir = wiki_root.parent / "outputs"
    lines = [
        "---", "title: lint 报告", "type: meta", "aliases: []", "sources: []",
        f"updated: {datetime.now():%Y-%m-%d}", "confidence: stated",
        "graph-excluded: true", "---", "",
        "# lint 报告", "",
        f"- 检查时间：{datetime.now():%Y-%m-%d %H:%M}",
        f"- 检查页数：{rep.pages_checked}",
        f"- 结果：{rep.summary()}", "",
    ]
    if rep.issues:
        lines += ["| 检查 | 页面 | 说明 |", "|---|---|---|"]
        lines += [f"| {i.check} | `{i.page}` | {i.detail} |" for i in rep.issues]
    else:
        lines += ["没有发现问题。", ""]
    lines += ["", "> 这份报告由 `wiki/lint.py` 生成，**只报告不自动修** ——",
              "> 自动修会让\"检查通过\"变得廉价：坏掉的东西被悄悄改回去，没人知道曾经坏过。", ""]
    P.write_text(out_dir / "lint.md", "\n".join(lines))

    # 结果存一份机器可读的，让 `build` 能把它渲染进 Dashboard
    P.write_text(out_dir / "lint.json",
                 json.dumps(rep.to_dict(), ensure_ascii=False, indent=2) + "\n")
    # 顺手把 Dashboard 那一行刷新掉，跑完 lint 就能看到结果
    refresh_dashboard(wiki_root)
    return out_dir / "lint.md"


def dashboard_line(wiki_root: Path) -> str:
    """`overview.md` 里那一行 lint 状态。**`build` 和 `lint` 共用这一个函数** ——
    两处各写一遍必然漂移，而漂移的 Dashboard 显示的是过期的好消息。
    """
    rep = read_last_result(wiki_root)
    if not rep:
        return "- lint：还没跑过（`python scripts/lint.py`）"
    n = rep.get("issue_count", 0)
    if n:
        detail = " · ".join(f"{k} {v}"
                            for k, v in sorted((rep.get("by_check") or {}).items()))
        line = f"- lint：⚠️ {n} 项问题（{detail}）"
    else:
        line = "- lint：✅ 没有问题"
    return line + f"　（{rep.get('checked_at', '')[:16]}）"


def refresh_dashboard(wiki_root: Path) -> bool:
    """把 lint 结果写进 `overview.md` 的那一行（跑完 lint 不用再跑一次 `build`）。
    """
    f = Path(wiki_root) / "overview.md"
    if not f.is_file():
        return False
    text = f.read_text(encoding="utf-8")
    new = re.sub(r"^- lint：.*$", dashboard_line(wiki_root), text,
                 count=1, flags=re.M)
    if new == text:
        return False
    P.write_text(f, new)
    return True


def read_last_result(wiki_root: Path) -> dict | None:
    """上一次 lint 的结果（机器可读）。`wiki_root` 是**页面那一层**。"""
    f = Path(wiki_root).parent / "outputs" / "lint.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main(root: Path | None = None) -> int:
    """`scripts/lint.py` 的入口。有问题返回 1，干净返回 0。"""
    from config import get_settings
    wiki_root = Path(root or get_settings().WIKI_DIR)
    if (wiki_root / "wiki").is_dir():
        wiki_root = wiki_root / "wiki"
    rep = lint(wiki_root)
    write_report(wiki_root, rep)
    print(rep.summary())
    for i in rep.issues[:40]:
        print(f"  [{i.check}] {i.page}: {i.detail}")
    if len(rep.issues) > 40:
        print(f"  …还有 {len(rep.issues) - 40} 条")
    print(f"报告已写入 {wiki_root.parent / 'outputs' / 'lint.md'}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
