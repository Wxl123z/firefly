"""上下文层：token 计量、prompt 组装、时间渲染、尾部通知。

prompt 四段式：system（恒定）+ history（追加式）+ tail（易变）+ 本轮输入。
易变内容一律进尾部，绝不进 system 头部（保证前缀缓存命中）；历史消息带稳定
时间前缀，由写入时的 ts 确定性渲染、永不变化；DB 存原文，prompt 只渲染该看见的部分。
"""
import json
import re
from datetime import datetime
from pathlib import Path

from config import DATA_DIR, get_settings

settings = get_settings()

#: 走了这么多步还没有计划，就在尾部提醒一次（见 build_notice）。
PLAN_NUDGE_AFTER = 4

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
# 启发式系数（Qwen BPE 经验值）：中文约 0.75 token/字，其余约 0.28 token/字符
_CJK_RATIO = 0.75
_OTHER_RATIO = 0.28
_PER_MESSAGE_OVERHEAD = 4      # chat template 每条消息的固定开销
_SYSTEM_OVERHEAD = 8

_CALIBRATION_FILE = DATA_DIR / "token_calibration.json"

_tokenizer = None
_tokenizer_tried = False


def _load_tokenizer():
    """尝试加载本地 Qwen tokenizer；失败返回 None（退化为估算）。"""
    global _tokenizer, _tokenizer_tried
    if _tokenizer_tried:
        return _tokenizer
    _tokenizer_tried = True
    if not settings.TOKENIZER_PATH:
        return None
    try:
        from tokenizers import Tokenizer  # 延迟导入：没装也不影响运行
        path = Path(settings.TOKENIZER_PATH)
        tokenizer_file = path / "tokenizer.json" if path.is_dir() else path
        _tokenizer = Tokenizer.from_file(str(tokenizer_file))
    except Exception:
        _tokenizer = None
    return _tokenizer


class TokenCounter:
    """token 计数 + 在线校准：估算值 × ratio ≈ 真实值，ratio 持久化。"""

    def __init__(self) -> None:
        self.ratio = 1.0
        self.samples = 0
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(_CALIBRATION_FILE.read_text(encoding="utf-8"))
            self.ratio = float(data.get("ratio", 1.0))
            self.samples = int(data.get("samples", 0))
        except Exception:
            pass

    def _save(self) -> None:
        try:
            _CALIBRATION_FILE.write_text(
                json.dumps({"ratio": self.ratio, "samples": self.samples}),
                encoding="utf-8",
            )
        except Exception:
            pass

    def raw(self, text: str) -> int:
        """单段文本的 token 数（未校准）。"""
        tk = _load_tokenizer()
        if tk is not None:
            return len(tk.encode(text or "").ids)
        text = text or ""
        cjk = len(_CJK_RE.findall(text))
        other = len(text) - cjk
        return int(cjk * _CJK_RATIO + other * _OTHER_RATIO) + 1

    def count(self, text: str) -> int:
        """单段文本的 token 数（已校准）。"""
        if _load_tokenizer() is not None:
            return self.raw(text)
        return int(self.raw(text) * self.ratio) + 1

    def count_messages(self, messages: list[dict]) -> int:
        """整条消息列表的 token 数（含 chat template 固定开销）。"""
        total = _SYSTEM_OVERHEAD
        for m in messages:
            total += _PER_MESSAGE_OVERHEAD
            total += self.count(m.get("content") or "")
            for tc in (m.get("tool_calls") or []):
                total += self.count(json.dumps(tc, ensure_ascii=False))
        return total

    def calibrate(self, estimated: int, real: int) -> None:
        """用 API 返回的真实 prompt_tokens 校准估算比例。"""
        if estimated <= 0 or not real or real <= 0:
            return
        observed = real / estimated
        if not (0.2 <= observed <= 5.0):     # 明显异常（例如服务端截断）直接丢弃
            return
        alpha = 0.3
        self.ratio = self.ratio * (1 - alpha) + observed * alpha
        self.samples += 1
        self._save()

    def describe(self) -> str:
        tk = _load_tokenizer()
        src = "tokenizer" if tk is not None else f"estimate(ratio={self.ratio:.3f}, n={self.samples})"
        return src


counter = TokenCounter()


# ---------- 渲染 ----------

def render_time_prefix(ts: datetime) -> str:
    """历史消息的稳定时间前缀。由 ts 确定性生成 → 一旦写入永不变化。"""
    return f"[{ts:%m-%d %H:%M}]"


def render_history(history: list[dict]) -> list[dict]:
    """把 DB 历史渲染成 prompt 消息（追加时间前缀）。"""
    out = []
    for turn in history:
        content = f"{render_time_prefix(turn['ts'])} {turn['content']}"
        out.append({"role": turn["role"], "content": content})
    return out


def now_line(ts: datetime | None = None) -> str:
    ts = ts or datetime.now()
    weekday = "一二三四五六日"[ts.weekday()]
    return f"现在：{ts:%Y-%m-%d %H:%M} 周{weekday}"


# ---------- 尾部通知 ----------

def build_notice(
    *,
    ctx_tokens: int = 0,
    ctx_ratio: float = 0.0,
    packs: list[tuple[str, str]] | None = None,
    loaded_tools: list[str] | None = None,
    last_compress_turn: int = 0,
    turn_count: int = 0,
    forced_compress: bool = False,
    ts: datetime | None = None,
    shell_hint: str = "",
    plan_text: str = "",
    steps: int = 0,
) -> str:
    """组装尾部易变区文本（不落库，每轮重算）。

    这里**不放记忆**（记忆由模型自己调 `retrieve_memory` 取）；`shell_hint` /
    `plan_text` / `steps` 是动态内容，只能走尾部，放前缀会炸缓存。
    """
    lines = [f"【系统通知】（以下是系统补充信息，不是{settings.USER_TERM}说的话）",
             f"- {now_line(ts)}"]

    if ctx_ratio >= settings.CTX_WARN_RATIO:
        pct = f"{ctx_ratio * 100:.0f}%"
        lines.append(
            f"- 上下文：{ctx_tokens:,} / {settings.CTX_WINDOW:,} token（{pct}）。"
            "如需清理早期对话可调用 compress_context；不清理也能继续。"
        )
    # ---- 压缩：分清"谁干的" ----
    # ⚠️ 两条**不能合成一条**：强制压缩是模型没有选择权的状态变更，自愿压缩是它
    # 自己调 `compress_context` 的结果。
    if forced_compress:
        lines.append("- ⚠️ 系统刚刚强制压缩了上下文（**不是你调用的 `compress_context`**）："
                     "早期对话已整理进摘要并移出，最近的对话没动，可以直接继续。")
    elif last_compress_turn:
        delta = turn_count - last_compress_turn
        if delta <= settings.COMPRESS_COOLDOWN_TURNS + 2:
            # delta <= 0 是"本轮内刚压过"（强制路径会走到这，自愿路径压完也会）
            lines.append(f"- 上次压缩：{'就是刚才' if delta <= 0 else f'{delta} 轮前'}")

    if shell_hint:
        lines.append(f"- 执行环境：{shell_hint}")

    if plan_text:
        lines.append("")
        lines.append(plan_text)
    elif steps >= PLAN_NUDGE_AFTER:
        # 提醒"你已经走了 N 步、还没有计划"：这个可观察状态它自己看不见，而提示词里
        # "任务超过 3 步就先计划"那条对它无效（那要求它预测任务长度）。
        lines.append("")
        lines.append(
            f"【提醒】本回合你已经走了 {steps} 步，还没有列计划。"
            f"如果这件事不止一个步骤，用 `todo_write` 把剩下的列出来 —— "
            f"免得做着做着忘了原本要干什么。已经只差最后一步就直接做完。")

    if packs:
        desc = " · ".join(f"{name}({brief})" for name, brief in packs)
        lines.append(f"- 可用工具包：{desc}")
    if loaded_tools:
        # 尾部说明只是"提示"，真正让工具可调用的是 tools= 字段里的注册。
        lines.append(f"- 已为你加载的扩展工具：{', '.join(loaded_tools)}"
                     "（已在你的可用工具列表中，可直接调用）")

    return "\n".join(lines)


SUMMARY_HEADER = (
    "【早期对话摘要】\n"
    "（较早的对话已被压缩成下面的摘要，原文不在上下文里了。"
    "它可能失真或过时——引用时用「我大概记得」的口吻，细节不确定就说不确定。）"
)


def build_user_message(user_input: str, notice: str = "") -> str:
    """尾部通知 + 原文，合成一条 user 消息。

    分隔线之后的才是真正要回应的话；通知块不落库。并入 user 而不是新增 system
    消息，是因为云端 Qwen 拒绝中途出现的 system。
    """
    if not notice:
        return user_input
    return f"{notice}\n\n---\n{user_input}"


def assemble_messages(
    system_prompt: str,
    history: list[dict],
    user_input: str,
    *,
    summary: str = "",
    notice: str = "",
    scratch: list[dict] | None = None,
) -> list[dict]:
    """组装发给模型的完整 messages。

    四段式（摘要属于**前缀**，与 system 一起被缓存）：
    [system] [摘要] [history] | [尾部通知 + 本轮输入] [本轮临时消息]
    └── 缓存命中 ──┘            └── 每轮重算 ──┘        └ 回合结束即丢 ┘
    摘要放前缀是因为它只在压缩时变化，而压缩本来就把 history 整个改写、缓存全炸。
    """
    messages = [{"role": "system", "content": system_prompt}]
    if summary:
        messages.append({"role": "user",
                         "content": f"{SUMMARY_HEADER}\n{summary}"})
    messages.extend(render_history(history))
    messages.append({"role": "user", "content": build_user_message(user_input, notice)})
    messages.extend(scratch or [])
    return messages


# ---------- 历史裁剪（压缩用） ----------

def split_history_by_tokens(history: list[dict], keep_tokens: int) -> tuple[list[dict], list[dict]]:
    """按 token 预算把历史切成（较早部分, 保留部分）。

    保留部分从尾部往前累加直到超过预算，并**对齐到 user 消息边界**以免拆散问答对。
    """
    if not history:
        return [], []

    total = 0
    cut = len(history)
    for i in range(len(history) - 1, -1, -1):
        total += counter.count(history[i]["content"]) + _PER_MESSAGE_OVERHEAD
        if total > keep_tokens:
            break
        cut = i

    # 对齐：保留段的起点必须是 user 消息（拆散问答对会让模型看到孤立的回答）
    while 0 < cut < len(history) and history[cut]["role"] != "user":
        cut -= 1
    cut = max(cut, 0)
    if cut == 0:
        return [], history
    return history[:cut], history[cut:]


def mask_tool_result(text: str, max_tokens: int | None = None) -> str:
    """观察掩码：超长工具结果截断中段（保留首尾，模型通常只需开头与结尾）。"""
    limit = max_tokens or settings.TOOL_RESULT_MAX_TOKENS
    if counter.count(text) <= limit:
        return text
    # 用字符比例粗切后重试，避免对超长文本反复计数
    keep_chars = max(int(len(text) * limit / max(counter.count(text), 1)), 200)
    head = text[: int(keep_chars * 0.6)]
    tail = text[-int(keep_chars * 0.3):]
    return f"{head}\n\n…（内容过长，已省略中间 {len(text) - len(head) - len(tail)} 字）…\n\n{tail}"
