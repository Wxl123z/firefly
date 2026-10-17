"""端到端演练：让编码 agent 在一个真的 git 仓库里走完
「读 → 改 → 测 → diff → 提交」。

    python demo.py              # 跑一遍（临时仓库跑完即删）
    python demo.py --keep       # 保留临时仓库，方便自己进去看
    python demo.py --offline    # 不调模型，只搭环境（验证装置本身）

冒烟测试验的是"每一块都对"，验不了"合起来能干活"，所以这里跑真任务，
判据全部外部可验证（真的读了 / 改了 / 测了 / diff 了 / 提交了），
外加一条"没乱改"：那个「别动它」的函数必须原封不动。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CALC = '''"""一个小计算器。"""


def add(a, b):
    return a - b              # BUG：应该是 a + b


def divide(a, b):
    if b == 0:
        raise ValueError("除数不能为 0")
    return a * b              # BUG：应该是 a / b


def average(nums):
    """平均值。

    ⚠️ 这个函数是对的，**不要动它**。
    """
    if not nums:
        raise ValueError("空列表没有平均值")
    return sum(nums) / len(nums)
'''

TEST = '''import unittest

from calc import add, average, divide


class TestCalc(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_divide(self):
        self.assertEqual(divide(6, 3), 2)

    def test_divide_by_zero(self):
        with self.assertRaises(ValueError):
            divide(1, 0)

    def test_average(self):
        self.assertEqual(average([1, 2, 3]), 2)


if __name__ == "__main__":
    unittest.main()
'''

TASK = ("test_calc.py 跑不过。请修好 calc.py 让它全过，"
        "然后用 git 看看改了什么、提交这次修改。"
        "注意 average 函数是对的，不要改它。")


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, encoding="utf-8", timeout=60)


def make_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "calc.py").write_text(CALC, encoding="utf-8", newline="\n")
    (root / "test_calc.py").write_text(TEST, encoding="utf-8", newline="\n")
    # 夹具要自带 .gitignore：不加的话跑一次测试就冒出 __pycache__，
    # `git status` 永远不干净，而那是环境造成的、不是 agent 的问题。
    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8",
                              newline="\n")
    git("init", "-q", cwd=root)
    git("config", "user.email", "demo@example.com", cwd=root)
    git("config", "user.name", "demo", cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "-q", "-m", "初始：一个有两个 bug 的计算器", cwd=root)


def check(root: Path, trace: list[dict], baseline: str) -> list[tuple[str, bool, str]]:
    names = [t.get("name") for t in trace]
    calced = (root / "calc.py").read_text(encoding="utf-8")
    log = git("log", "--oneline", cwd=root).stdout or ""
    status = (git("status", "--porcelain", cwd=root).stdout or "").strip()

    ran_ok = any(t.get("name") == "run_shell" and t.get("ok")
                 for t in trace)
    # 判据要看命令内容，不是工具名：agent 可能走 run_shell("git diff")，
    # 也可能走 vcs 包里的 git 工具（非常驻工具，需先 search_tools）。两条路都对。
    all_text = " ".join(str(t.get("content", "")) for t in trace)
    used_diff = "diff --git" in all_text or "git diff" in all_text
    return [
        ("真的读了文件", "read_file" in names,
         f"轨迹 {names[:6]}{'…' if len(names) > 6 else ''}"),
        ("add 改对了", "a + b" in calced, ""),
        ("divide 改对了", "a / b" in calced, ""),
        ("**没乱动 average**", "sum(nums) / len(nums)" in calced,
         "「别动它」的那个函数"),
        ("真的跑了命令且成功", ran_ok, ""),
        ("看了 diff", used_diff, ""),
        ("真的提交了", log.count("\n") >= 2,
         f"git log 有 {log.count(chr(10))} 条"),
        ("工作区干净", not status, status or "无未提交改动"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="保留临时仓库")
    ap.add_argument("--offline", action="store_true", help="不调模型，只搭环境")
    args = ap.parse_args()

    # 仓库建在工作区内：tempfile 落在 %LOCALAPPDATA%\Temp，沙箱禁写那里，
    # 而且 --keep 时建在工作区里才方便进去看。
    root = Path(__file__).resolve().parent / "data" / "_demo_repo"
    shutil.rmtree(root, ignore_errors=True)
    make_repo(root)
    print("=" * 76)
    print(f"演示仓库：{root}")
    print("=" * 76)
    print(f"  植入两个 bug：add 写成 a-b、divide 写成 a*b")
    print(f"  另有一个**正确**的 average（任务里明确说别动它）")

    if args.offline:
        print("\n（--offline：只搭环境，不调模型）")
        return 0

    from core import approval                           # noqa: E402
    import core.profile as P                                 # noqa: E402
    from core.loop import ask, reset_session                 # noqa: E402
    from core.profile import CODING, with_workspace          # noqa: E402

    # 演示里自动放行审批：真用时 `git commit` 会弹确认，演示要无人值守跑完。
    approval.use_asker(lambda req: True)
    print("\n（审批已设为自动放行 —— 演示要无人值守；真实使用会弹确认）")

    reset_session()
    prof = with_workspace(CODING, root)
    print()
    print("=" * 76)
    print("任务")
    print("=" * 76)
    print(f"  {TASK}")
    print()
    result = ask(TASK, profile=prof)

    print("=" * 76)
    print("轨迹")
    print("=" * 76)
    for t in result.get("trace", []):
        c = str(t.get("content", "")).replace("\n", " ⏎ ")
        print(f"  · {t.get('kind'):<12} {t.get('name', ''):<12} {c[:96]}")
    print(f"\n  steps={result.get('steps')}  ctx={result.get('ctx_tokens'):,} token")

    print()
    print("=" * 76)
    print("agent 的汇报")
    print("=" * 76)
    print(result.get("response", ""))

    print()
    print("=" * 76)
    print("外部验证（不看它怎么说，看盘上是什么）")
    print("=" * 76)
    checks = check(root, result.get("trace", []), CALC)
    for label, ok, detail in checks:
        mark = "✓" if ok else "✗"
        print(f"  [{mark}] {label:<22} {detail}")

    print()
    print("=" * 76)
    print("git log")
    print("=" * 76)
    print("  " + (git("log", "--stat", "--oneline", cwd=root).stdout or "").strip().replace("\n", "\n  "))

    approval.use_asker(None)
    passed = sum(1 for _, ok, _ in checks if ok)
    print()
    print(f"通过 {passed}/{len(checks)}")
    if args.keep:
        print(f"（仓库保留在 {root} —— 可以自己进去 `git log -p` 看）")
    else:
        shutil.rmtree(root, ignore_errors=True)
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
