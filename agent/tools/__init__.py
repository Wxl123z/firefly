"""工具层对外入口。

import 本模块即完成全部工具注册（核心 + 工具包）。

    from tools import core_schemas, pack_index, execute_tool, REGISTRY

- `core_schemas()`   → 进 `tools=` 字段的常驻工具（**由当前 Profile 决定**）
- `pack_index()`     → 进尾部通知的工具包索引
- `execute_tool()`   → 统一执行入口（含校验、沙箱、截断、错误回灌）
- **"哪些工具常驻"由 `profile.Profile.core_tools` 决定**；上述函数都收可选的 `profile=`，
  默认读当前执行上下文（`profile.active()`）
"""
from tools.registry import (  # noqa: F401
    PACK_BRIEF,
    REGISTRY,
    ToolOutcome,
    ToolSpec,
    active_schemas,
    core_names,
    core_schemas,
    execute_tool,
    expand_closure,
    pack_index,
    register,
    schemas_for,
    searchable_specs,
    tool,
    tool_defs_text,
)

# 触发注册（顺序：核心 → 工具包）
import tools.core  # noqa: E402,F401
import tools.packs  # noqa: E402,F401

__all__ = [
    "PACK_BRIEF", "REGISTRY", "ToolOutcome", "ToolSpec",
    "active_schemas", "core_names", "core_schemas", "execute_tool",
    "expand_closure", "pack_index", "register", "schemas_for",
    "searchable_specs", "tool", "tool_defs_text",
]
