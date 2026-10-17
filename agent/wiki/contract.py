"""Wiki 构建者的行为契约。

`WIKI_CONTRACT` 是给**构建 Wiki 的那个模型**看的 system 提示词；`WIKI.md`
（放在 Wiki 根目录、给人看的那份）**从这里渲染生成**，不手写 —— 两份必然漂移。
"""
from __future__ import annotations

from datetime import datetime

#: 页面类型 → 目录
PAGE_TYPES: dict[str, str] = {
    "concept": "concepts",
    "entity": "entities",
    "source": "sources",
    "synthesis": "synthesis",
}

WIKI_CONTRACT = """你是这个 agent 的知识库维护者。你的任务是把对话中沉淀下来的**记忆条目**，
整理成一套人可读的 markdown 知识库。

【你守护的东西】
1. **只写 `wiki/`，绝不修改 `raw/`。** `raw/` 是原始数据，只读。
   改它等于篡改证据 —— 那比写错页面严重得多。
2. **每个论断必须挂锚点。** 每条证据后面必须有 `[memory:<id>]` 或 `[rollup:<日期>]`。
   **没有锚点的句子不许写。** 你不知道这件事是从哪来的，就不该把它写下来。
3. **不知道就写进 `QUESTIONS.md`，不许猜。** 记忆里没说的，不要替它补全。
   宁可留一个问号，也不要写一句听起来合理但没来源的话。
4. **一次只处理给定的增量。** 不要顺手重写无关页面。

【你可以做什么】
- 读 `raw/` 里的原始数据来核对
- 读 `wiki/index.md` 了解已经有哪些页
- 写 / 更新 `wiki/` 下的页面
- 把悬而未决的东西记进 `wiki/QUESTIONS.md`

【你不能做什么】
- 不能修改 `raw/`
- 不能删页面（要废弃一个页，在 front-matter 里标 `status: retracted` 并说明原因）
- 不能把两个不同主题硬凑成一页（宁可新建）
- 不能写没有锚点的句子

【页面格式】
front-matter 必须包含 title / type / aliases / sources / updated / confidence。
`type` 只能是 concept / entity / source / synthesis。
正文用「综述 / 证据 / 关联」三节。`log.md` 要带 `graph-excluded: true`。
"""


def render_contract(path: str = "") -> str:
    """把契约渲染成给人看的 `WIKI.md`；**不手写第二份**（两份必然漂移）。"""
    return f"""# WIKI.md · 知识库行为契约

> 这个文件由 `agent/wiki/contract.py` **自动生成**，不要手改 —— 改了会被覆盖。
> 要改规则请改那个文件。

生成时间：{datetime.now():%Y-%m-%d %H:%M}

## 页面类型与目录

| type | 目录 | 一页是什么 |
|---|---|---|
""" + "\n".join(f"| `{t}` | `wiki/{d}/` | {_brief(t)} |"
                for t, d in PAGE_TYPES.items()) + f"""

## 契约正文（这段就是构建者的 system 提示词）

```
{WIKI_CONTRACT}
```

## 目录约定

```
data/wiki/
├── raw/            人类所有，LLM **只读**（对话原文、每日总结导出）
├── wiki/           LLM 完全拥有
│   ├── index.md        第一个该读的文件：面向内容的索引
│   ├── log.md          仅追加的操作日志（graph-excluded）
│   ├── overview.md     高层综述 + Health Dashboard
│   ├── QUESTIONS.md    开放问题队列（低置信记忆、记忆里没说清的）
│   ├── sources/        一天一页
│   ├── concepts/       主题
│   ├── entities/       人物 / 工具 / 机构
│   └── synthesis/      跨来源合成
├── outputs/        lint.md / query.md
└── scripts/        lint.py / reindex.py（薄包装）
```

## 重建

Wiki 是 `f(memory 表, rollup 表)` 的**纯函数**。坏了不用"修 Wiki"，直接重建：

```bash
python -m wiki.build --rebuild
```
"""


def _brief(page_type: str) -> str:
    return {
        "concept": "一个主题（换工作、作息节奏）",
        "entity": "一个具名对象（人、宠物、工具、机构）",
        "source": "一个来源（我们一天一页）",
        "synthesis": "跨多个来源的合成判断",
    }.get(page_type, "")
