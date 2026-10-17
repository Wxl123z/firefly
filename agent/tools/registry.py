"""工具注册表：spec 定义 + 参数校验 + 统一执行入口。

- 工具分 core（常驻，走 `tools=` 字段）/ pack（经 search_tools 延迟注入）/ 内部三类
- **「哪些是 core」由 `profile.Profile.core_tools` 决定，不由工具自己声明**
- 参数校验用 pydantic 动态建模，失败时把可读的中文错误回灌给模型（自愈）
- `execute_tool` 永不抛异常；返回 `ToolOutcome`，可带控制信号供循环路由
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ValidationError, create_model

from core.context import mask_tool_result
from core.profile import Profile, active

# 工具包索引里的短描述（粗粒度地图，进尾部通知）
PACK_BRIEF: dict[str, str] = {
    "memory": "记忆读写",
    "web": "联网",
    "fs": "文件",
    "code": "代码检索",
    "exec": "执行命令",
    "vcs": "git",
    "plan": "任务清单",
    "wiki": "知识库",
    "schedule": "定时与日程",
    "meta": "元操作（检索工具、压缩上下文）",
}


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    pack: str = "core"
    searchable: bool = True          # 是否参与 search_tools 检索（core 已常驻，不再检索）
    brief: str = ""                  # 工具包索引里的短描述
    # 依赖闭包：检索命中本工具时，一并把 requires 里的工具补齐（必须同现的工具）。
    requires: tuple[str, ...] = ()

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def defs_line(self) -> str:
        """延迟注入用的紧凑定义（<tool_defs> 块内一行一个 JSON）。"""
        return json.dumps(
            {"name": self.name, "description": self.description,
             "parameters": self.parameters},
            ensure_ascii=False,
        )

    def search_text(self) -> str:
        """检索用的文档文本。"""
        props = ", ".join((self.parameters.get("properties") or {}).keys())
        return f"{self.name} [{self.pack}] {self.description} 参数: {props}"


@dataclass
class ToolOutcome:
    content: str                              # 回灌给模型的工具结果
    control: str | None = None                # "compress" | "load_tools" | None
    payload: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    elapsed_ms: int = 0


REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    if spec.name in REGISTRY:
        raise ValueError(f"工具名重复注册：{spec.name}")
    REGISTRY[spec.name] = spec
    return spec


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    *,
    pack: str = "core",
    searchable: bool = True,
    brief: str = "",
    requires: tuple[str, ...] = (),
):
    """装饰器：把函数注册成工具。是否常驻由 Profile 的 `core_tools` 决定；`searchable=False` 表示永不参与检索。"""
    def deco(fn: Callable) -> Callable:
        register(
            ToolSpec(name=name, description=description, parameters=parameters,
                     func=fn, pack=pack, searchable=searchable,
                     brief=brief, requires=requires),
        )
        return fn
    return deco


def expand_closure(names: list[str]) -> list[str]:
    """依赖闭包：把 requires 里的工具补齐（传递闭包），保持顺序、去重。"""
    out: list[str] = []
    pending = list(names)
    while pending:
        n = pending.pop(0)
        if n in out or n not in REGISTRY:
            continue
        out.append(n)
        pending.extend(REGISTRY[n].requires)
    return out


# ---------- 对外结构 ----------
# 这些函数都收一个可选的 `profile`，默认 `None` → 用当前执行上下文里的 Profile
# （`profile.active()`），所以 `core_schemas()` 这类老调用点行为不变。

def _prof(profile: Profile | None) -> Profile:
    return profile if profile is not None else active()


def core_names(profile: Profile | None = None) -> tuple[str, ...]:
    """本 Profile 的常驻工具名（只保留确实注册过的）。"""
    return tuple(n for n in _prof(profile).core_tools if n in REGISTRY)


def core_schemas(profile: Profile | None = None) -> list[dict]:
    """常驻工具的 OpenAI schema（进 `tools=` 字段，本 Profile 内恒定）。"""
    return [REGISTRY[n].openai_schema() for n in core_names(profile)]


def schemas_for(names: list[str], profile: Profile | None = None) -> list[dict]:
    """指定工具的 OpenAI schema：并入 `tools=` 字段（模型眼中的动作空间），代价是前缀缓存失效。"""
    core = set(core_names(profile))     # 已经是常驻的不要再加一遍，否则 tools= 里出现两次
    out = []
    for n in names:
        spec = REGISTRY.get(n)
        if spec is not None and n not in core:
            out.append(spec.openai_schema())
    return out


def active_schemas(loaded: list[str], profile: Profile | None = None) -> list[dict]:
    """当前该传给模型的完整工具集：常驻（恒定）+ 已加载扩展（会话内粘性）。"""
    return core_schemas(profile) + schemas_for(loaded, profile)


def pack_index(profile: Profile | None = None) -> list[tuple[str, str]]:
    """工具包索引（进尾部通知）：只列出本 Profile 下确实有可检索工具的包（常驻工具不算，白名单外的包不出现）。"""
    prof = _prof(profile)
    core = set(core_names(prof))
    allowed = set(prof.packs)
    packs: list[str] = []
    for spec in REGISTRY.values():
        if (spec.searchable and spec.name not in core
                and spec.pack in allowed and spec.pack not in packs):
            packs.append(spec.pack)
    return [(p, PACK_BRIEF.get(p, p)) for p in sorted(packs)]


def searchable_specs(profile: Profile | None = None,
                     pack: str | None = None) -> list[ToolSpec]:
    """可被 search_tools 检索到的工具：排除常驻工具与 `profile.packs` 白名单之外的包（能力边界）。"""
    prof = _prof(profile)
    core = set(core_names(prof))
    allowed = set(prof.packs)
    specs = [s for s in REGISTRY.values()
             if s.searchable and s.name not in core and s.pack in allowed]
    if pack:
        specs = [s for s in specs if s.pack == pack]
    return specs


def tool_defs_text(names: list[str]) -> str:
    """把指定工具的完整定义打包成 <tool_defs> 块（延迟注入的载体）。"""
    lines = [REGISTRY[n].defs_line() for n in names if n in REGISTRY]
    if not lines:
        return ""
    return ("<tool_defs>\n" + "\n".join(lines) + "\n</tool_defs>\n"
            "以上工具已加载，可直接调用（格式与常用工具相同）。")


# ---------- 参数校验 ----------

_TYPE_MAP: dict[str, Any] = {
    "string": str, "integer": int, "number": float,
    "boolean": bool, "array": list, "object": dict,
}

_model_cache: dict[str, type[BaseModel]] = {}

# 中文别名：模型偶尔会用错的字段名 → 归一化（自愈的一部分）
_PARAM_ALIASES: dict[str, str] = {}


def _build_model(spec: ToolSpec) -> type[BaseModel]:
    if spec.name in _model_cache:
        return _model_cache[spec.name]
    props = spec.parameters.get("properties") or {}
    required = set(spec.parameters.get("required") or [])
    fields: dict[str, Any] = {}
    for pname, pdef in props.items():
        py_type = _TYPE_MAP.get(pdef.get("type", "string"), Any)
        if pdef.get("enum"):
            py_type = Literal[tuple(pdef["enum"])]  # type: ignore[valid-type]
        if pname in required:
            fields[pname] = (py_type, ...)
        else:
            fields[pname] = (Optional[py_type], pdef.get("default", None))
    model = create_model(f"{spec.name}_args", **fields)
    _model_cache[spec.name] = model
    return model


def _normalize_args(spec: ToolSpec, args: dict) -> dict:
    """轻量归一化：剥掉 None 的可选参数、修正常见别名、还原被字符串化的 array / object 值。"""
    props = spec.parameters.get("properties") or {}
    out: dict[str, Any] = {}
    for k, v in (args or {}).items():
        key = _PARAM_ALIASES.get(k, k)
        if key not in props and len(props) == 1 and k in ("input", "query", "text", "q", "task"):
            key = next(iter(props))     # 单参数工具：模型换名字也能救回来
        if v is None:
            continue
        if (props.get(key) or {}).get("type") in ("array", "object") and isinstance(v, str):
            v = _maybe_json(v, props[key]["type"])
        out[key] = v
    return out


def _maybe_json(text: str, declared: str) -> Any:
    """把被字符串化的 array/object 还原回去；还原不了就原样返回，让校验去报错。"""
    t = text.strip()
    if not t or t[0] not in "[{":
        return text
    try:
        value = json.loads(t)
    except json.JSONDecodeError:
        return text
    if declared == "array" and not isinstance(value, list):
        return text
    if declared == "object" and not isinstance(value, dict):
        return text
    return value


def _format_validation_error(spec: ToolSpec, exc: ValidationError) -> str:
    props = spec.parameters.get("properties") or {}
    required = spec.parameters.get("required") or []
    detail = "；".join(
        f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()
    )
    return (f"参数校验失败：{detail}\n"
            f"该工具参数：{json.dumps(props, ensure_ascii=False)}\n"
            f"必填：{required}\n请修正后重新调用。")


# ---------- 执行 ----------

def execute_tool(name: str, arguments: dict | str | None) -> ToolOutcome:
    """统一执行入口。永不抛异常——失败也返回可读文本，让模型自愈。"""
    started = time.time()
    spec = REGISTRY.get(name)
    if spec is None:
        # 提示候选：优先在本 Profile 还没常驻的工具里找（该提示的是"你还能加载什么"）。
        candidates = [s.name for s in searchable_specs()]
        near = [c for c in candidates if name.lower() in c.lower() or c.lower() in name.lower()]
        if near:
            hint = f"你是不是想调用：{', '.join(near[:5])}？"
        else:
            hint = "该能力可能还没加载：先用 search_tools 检索，或直接说你缺少什么能力。"
        return ToolOutcome(content=f"错误：没有名为 {name} 的工具。{hint}",
                           ok=False, elapsed_ms=_ms(started))

    if isinstance(arguments, str):
        arguments = _safe_json(arguments)
    if not isinstance(arguments, dict):
        return ToolOutcome(content=f"错误：{name} 的参数必须是 JSON 对象，收到 {type(arguments).__name__}。",
                           ok=False, elapsed_ms=_ms(started))

    args = _normalize_args(spec, arguments)
    try:
        args = _build_model(spec).model_validate(args).model_dump(exclude_none=True)
    except ValidationError as exc:
        return ToolOutcome(content=_format_validation_error(spec, exc),
                           ok=False, elapsed_ms=_ms(started))

    try:
        result = spec.func(**args)
    except TypeError as exc:                       # 签名不匹配（模型传了多余参数）
        return ToolOutcome(content=f"错误：{name} 调用参数不匹配（{exc}）。请检查参数名。",
                           ok=False, elapsed_ms=_ms(started))
    except Exception as exc:                       # 工具内部异常 → 回灌，让模型换策略
        return ToolOutcome(content=f"错误：{name} 执行失败：{type(exc).__name__}: {exc}",
                           ok=False, elapsed_ms=_ms(started))

    elapsed = _ms(started)
    if isinstance(result, ToolOutcome):
        result.elapsed_ms = elapsed
        # 控制类结果（工具定义等）不能截断，否则 <tool_defs> 里的 JSON 会被截坏
        if result.control is None:
            result.content = mask_tool_result(result.content or "")
        return result
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    return ToolOutcome(content=mask_tool_result(text), elapsed_ms=elapsed)


def _ms(started: float) -> int:
    return int((time.time() - started) * 1000)


def _safe_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.S)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
