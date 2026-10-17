"""LLM 封装：OpenAI 兼容聊天 + 文本 tool_call 解析 + 本地 embed（bge-m3）。

`chat_with_tools` 返回结构化 `LLMReply`（content / tool_calls / usage），`tool_choice`
可透传。`extract_tool_calls` 把两条工具调用来源（API 的 `tool_calls` 字段、正文里的
`<tool_call>{...}</tool_call>`）统一成同一结构，调用方无需区分。
"""
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

import requests
from openai import OpenAI

from config import get_settings

settings = get_settings()

_client = OpenAI(
    base_url=settings.CHAT_BASE_URL,
    api_key=settings.CHAT_API_KEY,
    timeout=180.0,      # 单次调用上限：云端 27B 长输出可能到 1-2 分钟
    max_retries=1,
)

# 文本形式的工具调用（Qwen 原生格式）
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
# 容错：模型漏写 </tool_call> 时退化为「<tool_call> 后跟一段负载」
_LOOSE_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\}|<function\s*=[^>]+>.*?</function>)\s*(?=<tool_call>|$)", re.S)

# Qwen / vLLM 的 XML 工具调用写法：<function=名><parameter=键>值</parameter></function>
_XML_FUNC_RE = re.compile(r"<function\s*=\s*([^>\s]+)\s*>(.*?)</function>", re.S)
_XML_PARAM_RE = re.compile(r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>", re.S)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    source: str = "api"      # api | text
    parse_error: str | None = None

    def fingerprint(self) -> str:
        """用于重复调用护栏：工具名 + 入参哈希。"""
        blob = json.dumps(self.arguments, sort_keys=True, ensure_ascii=False)
        return f"{self.name}:{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class LLMReply:
    content: str = ""                                   # 自然语言部分（已剔除 <tool_call> 块）
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    raw: Any = None


def _is_local() -> bool:
    url = settings.CHAT_BASE_URL or ""
    return "localhost" in url or "127.0.0.1" in url


def _extra_body() -> dict:
    """按端点类型附加私有参数。"""
    body: dict = {}
    if _is_local():
        body["num_ctx"] = settings.OLLAMA_NUM_CTX
    style = (settings.THINKING_PARAM_STYLE or "none").lower()
    if style == "top_level":
        body["enable_thinking"] = settings.ENABLE_THINKING
    elif style == "chat_template_kwargs":
        body["chat_template_kwargs"] = {"enable_thinking": settings.ENABLE_THINKING}
    return body


def _extract_text_tool_calls(content: str) -> tuple[str, list[ToolCall]]:
    """从回复正文里抠出工具调用，返回（剩余正文, 工具调用列表）。

    三级回退：闭合的 `<tool_call>…</tool_call>` → 漏了闭合标签但负载可解析
    → 裸的 `<function=名>…</function>` 组。
    """
    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []

    for i, m in enumerate(_TOOL_CALL_RE.finditer(content)):
        spans.append(m.span())
        calls.append(_build_text_call(m.group(1), i))

    if not calls:  # 闭合标签缺失时的容错路径
        loose = list(_LOOSE_TOOL_CALL_RE.finditer(content))
        # 只在确实存在未闭合的 <tool_call> 时才走这条路
        if loose and content.count("<tool_call>") > content.count("</tool_call>"):
            for i, m in enumerate(loose):
                spans.append(m.span())
                calls.append(_build_text_call(m.group(1), i))

    if not calls:
        # 兜底：裸的 XML 调用（无 <tool_call> 外壳）。
        # 要求 `<function=…>` 里至少有一个 `<parameter=…>`，
        # 否则"正文里提到某个函数名"会被误判成一次调用。
        for i, m in enumerate(_XML_FUNC_RE.finditer(content)):
            if "<parameter" not in m.group(2):
                continue
            spans.append(m.span())
            calls.append(_build_text_call(m.group(0), i))

    if not spans:
        return content, []

    kept, last = [], 0
    for start, end in sorted(spans):
        kept.append(content[last:start])
        last = end
    kept.append(content[last:])
    return "".join(kept).strip(), calls


def _parse_xml_tool_call(blob: str) -> tuple[str, dict] | None:
    """解析 `<function=名><parameter=键>值</parameter></function>`。不是这个形状就返回 None。"""
    m = _XML_FUNC_RE.search(blob)
    if m is None:
        return None
    name = m.group(1).strip()
    if not name:
        return None
    return name, {pm.group(1).strip(): _coerce_param(pm.group(2))
                  for pm in _XML_PARAM_RE.finditer(m.group(2))}


def _coerce_param(text: str) -> Any:
    """参数值统一是字符串，能当 JSON 读的就还原成原生类型。

    `<parameter=top_k>5</parameter>` 要变成整数 5，否则 pydantic 校验会失败；
    裸 `null` 保持字符串 —— 它更可能是文件名或正文，不是"无值"。
    """
    t = text.strip()
    if not t:
        return ""
    try:
        value = json.loads(t)
    except json.JSONDecodeError:
        return t
    return t if value is None else value


def _build_text_call(blob: str, index: int) -> ToolCall:
    """把一段文本负载解析成 ToolCall，支持 JSON 与 Qwen XML 两种写法。"""
    call = ToolCall(id=f"text_{index}_{hashlib.md5(blob.encode('utf-8')).hexdigest()[:8]}",
                    name="", arguments={}, raw_arguments=blob, source="text")

    xml = _parse_xml_tool_call(blob)
    if xml is not None:
        call.name, call.arguments = xml
        return call

    data = _parse_json(blob)
    name = data.get("name") or data.get("tool") or data.get("function")
    args = data.get("arguments", data.get("parameters", data.get("args", {})))
    if name:
        call.name = str(name)
        if isinstance(args, str):
            args = _parse_json(args)
        call.arguments = args if isinstance(args, dict) else {}
    else:
        call.parse_error = ("无法从工具调用里解析出工具名"
                            "（支持 JSON 与 <function=工具名> 两种写法）")
    return call


def extract_tool_calls(message) -> tuple[str, list[ToolCall], str]:
    """统一两条工具调用来源 → （可见正文, 工具调用, 思考文本）。"""
    content = message.content or ""
    thinking = getattr(message, "reasoning_content", None) or ""
    calls: list[ToolCall] = []

    for tc in (getattr(message, "tool_calls", None) or []):
        raw = getattr(tc.function, "arguments", "") or ""
        args = _parse_json(raw) if raw else {}
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args,
                              raw_arguments=raw, source="api"))

    content, text_calls = _extract_text_tool_calls(content)

    # 去重：同一工具同一入参只保留一条（优先保留 API 结构化那条）
    seen = {(c.name, json.dumps(c.arguments, sort_keys=True, ensure_ascii=False))
            for c in calls}
    for c in text_calls:
        key = (c.name, json.dumps(c.arguments, sort_keys=True, ensure_ascii=False))
        if key not in seen:
            seen.add(key)
            calls.append(c)

    return content, calls, thinking


def chat_with_tools(
    messages: list,
    tools: list | None = None,
    *,
    tool_choice: Any = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> LLMReply:
    """带工具的一次对话调用。tools=None 时纯文本对话（不传 tools 字段）。"""
    kwargs: dict = dict(
        model=settings.CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or settings.MAX_OUTPUT_TOKENS,
        extra_body=_extra_body(),
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice or "auto"

    resp = _client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    print(msg)
    content, calls, thinking = extract_tool_calls(msg)
    usage = getattr(resp, "usage", None)
    return LLMReply(
        content=content,
        tool_calls=calls,
        thinking=thinking,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        finish_reason=getattr(resp.choices[0], "finish_reason", None),
        raw=msg,
    )


def chat(messages: list, temperature: float = 0.85, max_tokens: int | None = None) -> str:
    """纯文本对话（不带工具）。"""
    return chat_with_tools(messages, None, temperature=temperature,
                           max_tokens=max_tokens).content


def chat_json(messages: list, temperature: float = 0.2, max_tokens: int = 256) -> dict:
    """强制 JSON 输出的对话；宽松解析（容忍 markdown 围栏/前后缀）。解析失败返回空结构。"""
    kwargs = dict(
        model=settings.CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=_extra_body(),
    )
    try:
        resp = _client.chat.completions.create(
            **kwargs, response_format={"type": "json_object"})
        text = resp.choices[0].message.content or ""
    except Exception:
        resp = _client.chat.completions.create(**kwargs)  # 不支持 response_format 时兜底
        text = resp.choices[0].message.content or ""
    return _parse_json(text)


def _parse_json(text: str) -> dict:
    """宽松 JSON 解析：剥离围栏、截取最外层对象；失败返回 `{"memories": []}`。"""
    if isinstance(text, dict):
        return text
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"memories": []}
    except json.JSONDecodeError:
        m = re.search(r'"(?:memories|topics)"\s*:\s*(\[.*?\])\s*}?\s*$', text, re.S)
        if m:
            try:
                return {"memories": json.loads(m.group(1))}
            except json.JSONDecodeError:
                pass
        return {"memories": []}


def embed(text: str) -> list:
    """单条嵌入。返回 `EMBED_DIM` 维向量。要嵌一批时用 `embed_many()`。"""
    return embed_many([text])[0]


def embed_many(texts: list[str]) -> list[list]:
    """批量嵌入。走 OpenAI 兼容的 `/embeddings`，按 `EMBED_BATCH_SIZE` 切块。

    要嵌一批时应统一走这里。按输入顺序返回：接口按 `index` 字段给出，
    不能假定数组顺序。
    """
    items = [t for t in (texts or [])]
    if not items:
        return []
    out: list[list] = []
    size = max(1, settings.EMBED_BATCH_SIZE)
    for i in range(0, len(items), size):
        chunk = items[i:i + size]
        r = requests.post(
            f"{settings.EMBED_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {settings.EMBED_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": settings.EMBED_MODEL, "input": chunk},
            timeout=settings.EMBED_TIMEOUT,
        )
        if r.status_code != 200:
            # 带上响应体：raise_for_status() 只给一句 "400 Bad Request"
            raise RuntimeError(
                f"嵌入接口返回 {r.status_code}：{r.text[:300]}\n"
                f"（模型 {settings.EMBED_MODEL!r} · 端点 {settings.EMBED_BASE_URL}）")
        data = r.json().get("data") or []
        if len(data) != len(chunk):
            raise RuntimeError(
                f"嵌入返回条数对不上：发了 {len(chunk)} 条，回来 {len(data)} 条")
        ordered = sorted(data, key=lambda d: d.get("index", 0))
        out.extend(d["embedding"] for d in ordered)
    return out
