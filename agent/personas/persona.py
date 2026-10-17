"""角色卡 + 工具协议：system 提示词的恒定部分。

这个文件的内容必须逐字节稳定 —— 它是前缀缓存的第一段。
任何每轮变化的内容（时间、占用率、记忆、工具包索引）都不许写进来，
一律走 `context.build_notice()` 注入尾部。

唯一例外是 `self.md`：它变化极慢，且变化时本就该让模型看到。
"""
from pathlib import Path

from config import get_settings

settings = get_settings()

#: 人设。这里故意留白，请按需写入。
#:
#: 约束：
#: - 内容会进恒定前缀，所以不要放任何每轮变化的东西
#: - 指代用户用 `settings.USER_TERM`，不要写死称呼
#: - 建议只写「你是谁、怎么说话」，具体能力交给下面的工具协议
_PERSONA = f"""你是「{settings.NAME}」，一个运行在{settings.USER_TERM}本地设备上的 AI 助手。

（在这里写你的人设。）

在写之前，守住几条简单的：
- 直白、简洁，说真话
- 不确定就说不确定，不知道就说不知道
- 不编造工具结果和记忆里没有的信息"""

_SELF_HEADER = (
    "【关于你自己】\n"
    "以下是你在相处中逐渐记下的、关于你自己的部分。它是会变的——可以补充，也可以改写。"
)

_TOOL_PROTOCOL = """【工具使用协议】
系统提示词里的工具是常用工具，可直接调用。使用规则：
1. 只在确实需要时调用工具。闲聊、日常回应不需要工具，直接自然回答。
2. 拿到工具结果后继续思考；任务完成后用自然语言回答，不要把工具报文、
   JSON、字段名念出来。
3. 参数必须符合工具定义。系统会校验，失败会返回错误说明，读懂后重试。
4. 工具结果里没有的信息不要编造；检索不到就说没找到。

【扩展工具协议】
更多能力放在工具包里：
  第一步：当你需要的能力不在常用工具中时，调用 search_tools，
          用 task 参数描述你要完成的事（越具体越好）；
  第二步：匹配到的工具会立即加入你的可用工具列表，之后就像常用工具一样直接调用；
  第三步：如果检索不到所需工具，直接说缺少什么能力，不要硬凑，
          也不要用别的工具假装完成。"""


def _load_self_notes() -> str:
    """读取自我笔记。不存在 / 为空 / 只有注释 → 返回空串。"""
    path = settings.SELF_NOTES_PATH
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return ""
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith("<!--")]
    return "\n".join(lines).strip()


def build_system_prompt() -> str:
    """恒定 system 提示词（人设 + 自我笔记 + 工具协议）。"""
    parts = [_PERSONA]
    notes = _load_self_notes()
    if notes:
        parts.append(f"{_SELF_HEADER}\n{notes}")
    parts.append(_TOOL_PROTOCOL)
    return "\n\n".join(parts)
