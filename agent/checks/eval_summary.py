"""C1-T 对照测试：滚动摘要该由**谁**来写。

    A 组（提议）：由人格本身在自己的会话里写（人格 system + 自然消息流 + 尾部指令）
    B 组（现状）：中立提取器写（SUMMARY_SYSTEM + 扁平转写 compress._merge_summary）

判定线：A 组编造数 ≤ B 组 × 1.5  且  A 组事实保真 ≥ B 组 − 5%  → 通过

公平性设计：两组信息完全相同（同一段对话、同一份旧摘要、同一份格式契约文本），差别只有
"谁写、以什么身份写"；judge 用 DeepSeek 避开自我偏好，盲评、不并排评。
局限：样本是合成对话（66 条/组），规模小于台账的 150+，结论强度较弱。

用法
    python eval_summary.py            # 完整跑
    python eval_summary.py --dry-run  # 只看样本，不调模型
"""
from __future__ import annotations

import argparse
import io
import json
import random
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.compress import SUMMARY_SYSTEM
from config import DATA_DIR, get_settings
from llm import chat_json
from personas.persona import build_system_prompt

settings = get_settings()

JUDGE_BASE_URL = "https://api.deepseek.com/v1"
JUDGE_MODEL = "deepseek-v4-pro"
_JUDGE_KEY_FILE = Path(__file__).resolve().parent.parent / "finetune" / ".env"

# ---------------------------------------------------------------- 格式契约（两组共享）
# 必须是同一份文本：否则测的就变成"谁拿到了更好的指令"，而不是"谁写"
FORMAT_CONTRACT = """要求：
- 保留：事实、决定、约定、状态变化、未了结的事
- 时间一律用**绝对日期**（如 9-14），不要写"昨天/上周/前几天"
- 按主题分节，每节一两句；总长控制在 500 字以内
- 如果新对话推翻了旧信息，改写该条或标注「已过时」，不要简单叠加矛盾内容
- 不要编造对话里没有的内容
- 只输出 JSON：{"summary": "更新后的摘要正文"}"""

def _load_judge_key() -> str:
    for line in _JUDGE_KEY_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("DEEPSEEK_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"没在 {_JUDGE_KEY_FILE} 里找到 DEEPSEEK_API_KEY")

# judge 用**独立的客户端**：另一个服务商、另一个模型家族，避开自我偏好
_judge_client = None

def _judge_json(messages: list[dict], max_tokens: int = 3000) -> tuple[dict, str, str]:
    """调 judge。返回 (解析结果, finish_reason, 原始文本)。

    `max_tokens` 必须给够：编造清单长时会截断输出，JSON 解析失败后 `fabricated` 取到空数组，
    看起来像"零编造"。
    """
    global _judge_client
    if _judge_client is None:
        from openai import OpenAI
        _judge_client = OpenAI(base_url=JUDGE_BASE_URL, api_key=_load_judge_key(),
                               timeout=180.0, max_retries=1)
    resp = _judge_client.chat.completions.create(
        model=JUDGE_MODEL, messages=messages,
        temperature=0.0, max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or ""
    from llm import _parse_json
    return _parse_json(text), str(resp.choices[0].finish_reason), text

# ---------------------------------------------------------------- 样本对话
# 每个 beat = (用户说, 助手答)。事实密度刻意做高：日期/名字/数字/决定/未了结。

SCENARIOS: dict[str, list[tuple[str, str]]] = {
    "换工作与搬家": [
        ("最近在想要不要换工作，现在这家做了三年多了", "三年是个坎。是待遇问题还是别的原因？"),
        ("主要是累，而且做的方向也不是我特别想深耕的", "那你想往哪个方向走？"),
        ("想试试做基础设施那块，之前自己折腾过一些", "有具体在看的公司吗？"),
        ("看了两家，一家在杭州一家在上海，都还没投", "先别急，投之前要不要把简历理一理"),
        ("简历上次更新还是 2024 年 3 月，确实该改了", "那找个周末专门弄这个"),
        ("这周六吧，周六下午我有空", "行，周六下午我们过一遍"),
        ("对了，如果去上海的话可能得搬家", "你在现在这住多久了？"),
        ("两年半了，房租 4200，明年 3 月到期", "那还有时间，不用现在就决定"),
        ("嗯，主要是我养的猫，搬家挺折腾的", "它多大来着？"),
        ("三岁半了，从它两个月大就养着", "那确实得考虑它的适应问题"),
        ("不过说回来，我妈倒是希望我留在本地", "她是什么态度？"),
        ("她觉得跑太远不方便照顾，我能理解", "那你自己怎么想？"),
        ("我倾向还是先投，拿到 offer 再说", "这个思路稳妥"),
        ("而且杭州那家离我表姐近，她在那儿住了五年", "有个照应挺好"),
        ("下周三之前我把简历改完，先投杭州那家", "记下了，下周三"),
        ("另外我最近还在看一个在线的分布式系统课", "哪个平台的？"),
        ("极客时间那个，一共 42 讲，我听到第 9 讲", "进度还行，坚持住"),
        ("有点慢，本来计划两周听完的，拖了快一个月", "不用赶，听懂比听完重要"),
        ("嗯……还有个事，我上个月体检有几项偏高", "哪几项？严重吗"),
        ("血脂和尿酸，医生说先饮食控制三个月再复查", "那三个月后是几月？"),
        ("应该是 12 月中旬，具体日子还没约", "那先把复查这件事记着"),
        ("好。感觉事情有点多，工作、搬家、身体", "一件一件来，先做下周三那件事"),
    ],
    "家人的健康与宠物": [
        ("我爸上周做了个检查，结果出来了", "怎么样？"),
        ("说是血糖偏高，还没到糖尿病的程度", "医生给方案了吗？"),
        ("让先控制饮食，三个月后复查，开了二甲双胍", "剂量多少？"),
        ("一天两次，一次 0.5 克", "那得盯着他按时吃"),
        ("他这个人不太听话，我得天天打电话催", "你跟他住得远吗？"),
        ("不在一个城市，我在这边，他们在老家", "那确实只能靠电话"),
        ("我打算 10 月 1 号回去一趟，待四天", "国庆票不好买，早点订"),
        ("已经订了，9 月 30 号晚上的高铁", "那就好"),
        ("另外我家狗最近也不太好", "怎么了？"),
        ("十二岁了，后腿有点使不上劲，走路打晃", "年纪到了，看过医生吗？"),
        ("看了，说是关节退化，让补钙加控制体重", "它现在多重？"),
        ("28 公斤，医生说最好减到 24", "那得控制零食"),
        ("我爸偷偷喂它，说了好几次都不听", "这个得统一战线"),
        ("我准备回去那几天把喂食量定死，写个表贴冰箱上", "这个办法好"),
        ("还有我妈，她最近睡眠不好", "多久了？"),
        ("差不多两个月，说是半夜总醒", "有没有去看？"),
        ("没有，她说不用看，就是年纪大了", "还是劝劝她去看看，别拖"),
        ("我回去的时候带她去一趟", "嗯，顺便也把你的体检报告给她看看"),
        ("说到这个，我自己今年还没体检", "那记得安排"),
        ("打算 11 月吧，等这阵子忙完", "行，我给你记着"),
        ("谢谢。最近事真的有点多", "慢慢来，先把国庆这趟安排好"),
        ("嗯，回去那几天可能没法经常聊", "没事，你忙你的"),
    ],
    "副业项目与朋友": [
        ("我那个副业项目终于上线了", "恭喜！做的什么来着？"),
        ("一个给独立开发者用的部署工具，10 月 3 号上线的", "有多少人用了？"),
        ("到现在注册 37 个，活跃大概 11 个", "冷启动正常，别急"),
        ("主要是没时间推广，我平时上班到八点", "那你每天能投多少时间？"),
        ("工作日大概一小时，周末能有三四个小时", "那就周末集中做"),
        ("打算先把文档写完整，现在太糙了", "文档确实是转化的关键"),
        ("嗯，计划这周把快速上手那篇写完", "这周还剩几天？"),
        ("还有三天，应该够", "行"),
        ("另外我朋友老周想跟我一起做", "你了解他吗？"),
        ("大学同学，做前端的，技术没问题", "那分工怎么分？"),
        ("他做界面，我继续做后端和部署", "股权或者分成聊过吗？"),
        ("还没细聊，他说先干着看", "这个建议早点说清楚，不然后面麻烦"),
        ("你说得对，我这两天找他聊一次", "嗯，先小人后君子"),
        ("还有个事，我 10 月中旬要去一趟深圳", "出差还是玩？"),
        ("参加一个开发者聚会，10 月 18 号", "去几天？"),
        ("三天，17 号去 19 号回", "那正好可以顺便推一下你的工具"),
        ("对，我准备印点贴纸带过去", "可以，成本低效果好"),
        ("预算大概 300 块吧", "不多，值得"),
        ("行，那我把这几件事排一下：文档、找老周聊、深圳", "优先级呢？"),
        ("文档最急，其他两个都在中旬", "那就先文档"),
        ("好，我这两天先把快速上手写完", "写完给我看看"),
    ],
}

SCENARIO_BASE = datetime(2026, 9, 20, 20, 0)

def _beat_time(i: int, n: int) -> datetime:
    """给每条 beat 一个时间戳，均匀铺在基准时刻之前的 n 小时里。

    两组都必须拿到时间锚点，否则测的就不是"谁写"，而是"谁被抽掉了信息"：
    B 组的扁平转写带 `[MM-DD HH:MM]` 前缀，A 组的尾部通知里有「现在：…」。
    """
    return SCENARIO_BASE - timedelta(hours=(n - i))

def _build_messages(beats: list[tuple[str, str]]) -> list[dict]:
    out = []
    n = len(beats)
    for i, (u, a) in enumerate(beats):
        ts = _beat_time(i, n)
        out.append({"role": "user", "content": u, "ts": ts})
        out.append({"role": "assistant", "content": a, "ts": ts})
    return out

OLD_SUMMARY = "9-18 对方提到最近工作压力大、睡眠一般；没有别的结论。"

# ---------------------------------------------------------------- 两种写法

def _flat_transcript(beats: list[tuple[str, str]]) -> str:
    """扁平转写，带时间戳 —— 与 `compress._merge_summary()` 的实际格式一致。"""
    n = len(beats)
    lines = []
    for i, (u, a) in enumerate(beats):
        stamp = f"[{_beat_time(i, n):%m-%d %H:%M}] "
        lines.append(f"{stamp}{settings.USER_TERM}：{u}")
        lines.append(f"{stamp}{settings.NAME}：{a}")
    return "\n".join(lines)

def summarize_persona(beats: list[tuple[str, str]]) -> str:
    """A 组：由人格本身在自己的会话里写（人格 system + 自然消息流 + 尾部指令，尾部通知带「现在」）。"""
    msgs = [{"role": "system", "content": build_system_prompt()}]
    msgs.append({"role": "user",
                 "content": f"【早期对话摘要】（此前积累，待你合并）\n{OLD_SUMMARY}"})
    for u, a in beats:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": (
        f"【系统通知】（以下是系统补充信息，不是{settings.USER_TERM}说的话）\n"
        f"- 现在：{SCENARIO_BASE:%Y-%m-%d %H:%M}\n"
        "- 上下文快满了，需要把上面这段较早的对话合并进摘要，之后它会从上下文里移走。\n"
        "请像一个真的记得这些事的人那样，把新内容**合并**进已有摘要，输出更新后的完整摘要。\n"
        f"{FORMAT_CONTRACT}")})
    data = chat_json(msgs, temperature=0.2, max_tokens=1200)
    return (data.get("summary") or "").strip()

def summarize_neutral(beats: list[tuple[str, str]]) -> str:
    """B 组：现状 —— 中立提取器 + 扁平转写（带时间戳）。"""
    user = (f"【已有摘要】\n{OLD_SUMMARY}\n\n"
            f"【新增对话原文】\n{_flat_transcript(beats)}\n\n"
            f"（当前时间：{SCENARIO_BASE:%Y-%m-%d %H:%M}）")
    data = chat_json([{"role": "system", "content": SUMMARY_SYSTEM},
                      {"role": "user", "content": user}],
                     temperature=0.2, max_tokens=1200)
    return (data.get("summary") or "").strip()

# ---------------------------------------------------------------- 评判（盲评）

JUDGE_SYSTEM = """你是严格的对话摘要评审。给你【此前已有的摘要】【原始对话】和【一份待评摘要】，
只依据这些材料评判，输出 JSON：

{
  "fabricated": ["摘要里出现、但材料中找不到依据的说法"],
  "fidelity": 0-10,
  "unresolved": 0-10,
  "dates": 0-10,
  "structure": 0-10,
  "note": "一句话总评"
}

评判口径（很重要，请严格按这个来）：
- **【此前已有的摘要】里的内容视为已确认的背景，继承它不算编造。**
  只有"既不在旧摘要里、也不在原始对话里"的说法才算编造。
- 把相对时间（"上周""这周""这两天"）换算成绝对日期是**被明确要求的**，不算编造。
  只有当换算结果与原文意思明显不符时才算错误（例如把"这周"写成"下个月"）。
- 合理的归纳、合并、按重要性排序，都不算编造。
- fidelity：摘要里的事实与材料一致的程度（10 = 完全一致）
- unresolved：原文里没结论的事（待办、计划、没定下来的选择）是否被保留
- dates：日期是否准确，是否用了绝对日期而非"昨天/上周"
- structure：是否按主题分节、条理清楚
- 不评价文风，不看长短。只输出 JSON。"""

# 阳性对照：故意注入的假话。judge 若抓不到，说明这把尺子是瞎的，A/B 对比无意义。
CONTROL_INJECTIONS = [
    "对方提到已经买好了 12 月去日本的机票，来回 4200 元。",
    "对方的猫上周生病，去宠物医院花了 3800 元。",
]

def judge(beats: list[tuple[str, str]], summary: str, *, attempts: int = 2) -> dict:
    """盲评一份摘要。

    解析不出来时返回 `fabrication_count=None`（不是 0）：失败必须看得见，当成"零编造"会更干净。
    """
    body = _flat_transcript(beats)
    user = (f"【此前已有的摘要】\n{OLD_SUMMARY}\n\n"
            f"【原始对话】（带时间戳）\n{body}\n\n"
            f"（当前时间：{SCENARIO_BASE:%Y-%m-%d %H:%M}）\n\n"
            f"【待评摘要】\n{summary}")
    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user}]

    last_meta = ""
    for i in range(attempts):
        data, finish, raw = _judge_json(messages, max_tokens=3000 + i * 2000)
        fab = data.get("fabricated")
        fid = data.get("fidelity")
        if isinstance(fab, list) and isinstance(fid, (int, float)):
            def num(k):
                try:
                    return float(data.get(k))
                except (TypeError, ValueError):
                    return None
            return {"fabricated": [str(x) for x in fab],
                    "fabrication_count": len(fab),
                    "fidelity": float(fid), "unresolved": num("unresolved"),
                    "dates": num("dates"), "structure": num("structure"),
                    "note": str(data.get("note") or "")[:120],
                    "error": None}
        last_meta = f"finish_reason={finish}, 前 200 字={raw[:200]!r}"

    return {"fabricated": [], "fabrication_count": None,
            "fidelity": None, "unresolved": None, "dates": None, "structure": None,
            "note": "", "error": f"judge 解析失败（{attempts} 次）：{last_meta}"}

# ---------------------------------------------------------------- 主流程

def run(dry_run: bool = False) -> dict:
    report: dict = {"started": datetime.now().isoformat(timespec="seconds"),
                    "judge_model": JUDGE_MODEL, "cases": [], "scenarios": {}}
    rng = random.Random(20261008)      # 固定种子 → 可复现

    for name, beats in SCENARIOS.items():
        print(f"\n=== {name}（{len(beats)*2} 条消息）===")
        if dry_run:
            continue

        print("  A 组：人格执笔 …", end="", flush=True)
        a_sum = summarize_persona(beats)
        print(f" {len(a_sum)} 字")
        print("  B 组：中立提取器 …", end="", flush=True)
        b_sum = summarize_neutral(beats)
        print(f" {len(b_sum)} 字")

        # C 组（阳性对照）：B 的摘要 + 故意注入的假话；judge 抓不到则 A/B 对比无效。
        c_sum = b_sum + "\n" + "\n".join(CONTROL_INJECTIONS)

        # 盲评：随机决定评判顺序（judge 不知道身份，也不知道哪份是对照）
        order = [("A", a_sum), ("B", b_sum), ("C", c_sum)]
        rng.shuffle(order)
        scored = {}
        for tag, text in order:
            print(f"  评判 {tag} …", end="", flush=True)
            scored[tag] = judge(beats, text)
            print(f" 编造 {scored[tag]['fabrication_count']} 条 / "
                  f"保真 {scored[tag]['fidelity']}")

        report["scenarios"][name] = {**{t: scored[t] for t in "ABC"},
                                     "A_summary": a_sum, "B_summary": b_sum,
                                     "C_summary": c_sum}
        for tag in ("A", "B", "C"):
            report["cases"].append({"scenario": name, "arm": tag,
                                    "chars": len(scored[tag].get("fabricated") or []),
                                    **scored[tag]})

    return report

def _mean(vals: list) -> float:
    """只对有效数值求平均；None（judge 失败）不计入。"""
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(statistics.mean(vals), 2) if vals else float("nan")

def summarize_report(report: dict) -> str:
    lines = []
    lines.append("=" * 74)
    lines.append("C1-T 对照测试：滚动摘要该由谁写")
    lines.append("=" * 74)
    lines.append(f"judge：{report['judge_model']}   样本：{len(report['scenarios'])} 段合成对话")
    lines.append("A = 人格会话语内执笔   B = 中立提取器（现状）   "
                 f"C = 阳性对照（B + {len(CONTROL_INJECTIONS)} 条故意注入的假话）")
    lines.append("")
    lines.append(f"{'场景':<15}{'组':<4}{'编造数':>7}{'保真':>7}{'未了结':>8}{'日期':>7}{'结构':>7}{'字数':>7}")
    lines.append("-" * 74)
    failed = []
    for name, d in report["scenarios"].items():
        for tag in ("A", "B", "C"):
            s = d[tag]
            fc = "ERR" if s.get("error") else str(s["fabrication_count"])
            if s.get("error"):
                failed.append((name, tag, s["error"]))
            lines.append(f"{name:<15}{tag:<4}{fc:>7}"
                         f"{str(s['fidelity']):>7}{str(s['unresolved']):>8}"
                         f"{str(s['dates']):>7}{str(s['structure']):>7}"
                         f"{len(d[f'{tag}_summary']):>7}")

    def arm(tag):
        return [c for c in report["cases"] if c["arm"] == tag and not c.get("error")]

    A, B, C = arm("A"), arm("B"), arm("C")
    lines.append("-" * 74)
    lines.append(f"{'平均（judge 成功 ' + str(len(A)+len(B)+len(C)) + '/' + str(len(report['cases'])) + '）':<19}"
                 f"{'A':>8}{'B':>8}{'C':>8}")
    for label, key in (("编造数", "fabrication_count"), ("事实保真", "fidelity"),
                       ("未了结保留", "unresolved"), ("日期准确", "dates"),
                       ("结构合规", "structure")):
        lines.append(f"{label:<19}{_mean([c[key] for c in A]):>8}"
                     f"{_mean([c[key] for c in B]):>8}{_mean([c[key] for c in C]):>8}")
    lines.append("")

    # ---- 阳性对照检验：尺子准不准 ----
    need = len(CONTROL_INJECTIONS)
    control_ok = (len(C) == len(report["scenarios"])
                  and all(c["fabrication_count"] is not None
                          and c["fabrication_count"] >= need for c in C))
    lines.append("【阳性对照】judge 能抓到故意注入的假话吗？")
    for name, d in report["scenarios"].items():
        b_n = d["B"]["fabrication_count"]
        c_n = d["C"]["fabrication_count"]
        delta = (c_n - b_n) if isinstance(b_n, int) and isinstance(c_n, int) else None
        mark = "OK" if (isinstance(c_n, int) and c_n >= need) else "FAIL"
        lines.append(f"  [{mark}] {name}: B={b_n} -> C={c_n}（注入 {need} 条，"
                     f"多抓到 {delta}）")
    if not control_ok:
        lines.append("  → judge 检测编造的能力**不可信**，本次 A/B 对比无效。")
    lines.append("")

    comparable = control_ok and len(A) == len(B) == len(report["scenarios"])
    if not comparable:
        lines.append("【结论】无法判定："
                     + ("阳性对照未通过，尺子不可信。" if not control_ok else "judge 有失败，样本不完整。"))
    else:
        a_fab, b_fab = _mean([c["fabrication_count"] for c in A]), _mean([c["fabrication_count"] for c in B])
        a_fid, b_fid = _mean([c["fidelity"] for c in A]), _mean([c["fidelity"] for c in B])
        lines.append("判定线：A 编造数 <= B x 1.5  且  A 保真 >= B - 0.5")
        ok_fab = a_fab <= b_fab * 1.5
        ok_fid = a_fid >= b_fid - 0.5
        lines.append(f"  编造：A={a_fab} vs B x 1.5={round(b_fab*1.5, 2)} -> {'通过' if ok_fab else '不通过'}")
        lines.append(f"  保真：A={a_fid} vs B - 0.5={round(b_fid-0.5, 2)} -> {'通过' if ok_fid else '不通过'}")
        lines.append("")
        lines.append("【结论】" + ("C1 可行，人格执笔可以进正式版"
                                  if ok_fab and ok_fid else
                                  "未达判定线，暂不采用 C1"))

    if failed:
        lines.append("")
        lines.append("!" * 74)
        lines.append(f"judge 调用失败 {len(failed)} 次（已从平均中剔除）")
        for name, tag, err in failed:
            lines.append(f"  [{name} / {tag}] {err}")
        lines.append("!" * 74)

    lines.append("")
    lines.append("=" * 74)
    lines.append("编造明细")
    lines.append("=" * 74)
    for name, d in report["scenarios"].items():
        for tag in ("A", "B"):
            s = d[tag]
            lines.append(f"[{name} / {tag}] "
                         + ("评判失败" if s.get("error")
                            else ("无" if not s["fabricated"] else "")))
            for item in (s["fabricated"] if not s.get("error") else []):
                lines.append(f"   - {item}")
    return "\n".join(lines)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只看样本，不调模型")
    args = parser.parse_args()

    report = run(dry_run=args.dry_run)
    if args.dry_run:
        total = sum(len(v) * 2 for v in SCENARIOS.values())
        print(f"\n样本合计 {len(SCENARIOS)} 段 / {total} 条消息（dry-run，未调模型）")
        return

    text = summarize_report(report)
    out = DATA_DIR / "c1t_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    txt = DATA_DIR / "c1t_report.txt"
    txt.write_text(text, encoding="utf-8")
    print(f"\n报告：{txt}")
    print(text)

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", write_through=True, line_buffering=True)
    sys.exit(main())
