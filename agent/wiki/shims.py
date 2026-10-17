"""`data/wiki/scripts/lint.py` 的模板。

**薄包装，不是第二份实现**：逻辑全在 `agent/wiki/` 下，重复一份必然和主实现漂移。"""
from __future__ import annotations

import sys
from pathlib import Path

SHIM = '''#!/usr/bin/env python
"""{title}

由 `agent/wiki/build.py` 自动生成 —— **不要手改**，改了下次构建会被覆盖。
真正的实现在 `agent/wiki/` 下。
"""
import sys
from pathlib import Path

sys.path.insert(0, {agent_dir!r})

{body}
'''

SCRIPTS = {
    "lint.py": ("对 Wiki 跑九项一致性检查", '''
from wiki.lint import main

sys.exit(main(Path(__file__).resolve().parent.parent))
'''),
    "reindex.py": ("重建 Wiki 页面的嵌入索引", '''
from wiki.index import reindex

reindex(Path(__file__).resolve().parent.parent)
'''),
}


def render_scripts(agent_dir: str) -> dict[str, str]:
    return {name: SHIM.format(title=title, agent_dir=agent_dir, body=body.strip())
            for name, (title, body) in SCRIPTS.items()}
