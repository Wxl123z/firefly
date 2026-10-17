"""端到端验收。

    python smoke_test.py            # 全部检查（含需要 API 的端到端）
    python smoke_test.py --offline  # 跳过需要模型 API 的检查

输出同时写入 data/smoke_report.txt（UTF-8）。
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import readset
from config import BASE_DIR, DATA_DIR, get_settings

settings = get_settings()

#: 项目根 = agent/ 的上一级。执行工具的测试需要在**真的 git 仓库**里跑。
ROOT_DIR = Path(BASE_DIR).parent.resolve()

_LINES: list[str] = []
_PASS = 0
_FAIL = 0

def log(msg: str = "") -> None:
    """打印并同时收进报告缓冲。"""
    print(msg)
    _LINES.append(msg)

def check(name: str, fn) -> bool:
    """跑一个检查项并计时。fn 返回详情字符串（可为 None）；
    AssertionError 记简明原因，其它异常额外附 traceback。"""
    global _PASS, _FAIL
    started = time.time()
    try:
        detail = fn()
        _PASS += 1
        log(f"[PASS] {name} ({time.time() - started:.1f}s)" + (f" — {detail}" if detail else ""))
        return True
    except AssertionError as exc:
        _FAIL += 1
        log(f"[FAIL] {name} ({time.time() - started:.1f}s) — {exc}")
    except Exception as exc:
        _FAIL += 1
        log(f"[FAIL] {name} ({time.time() - started:.1f}s) — {type(exc).__name__}: {exc}")
        _LINES.append(traceback.format_exc())
    return False

# ---------------------------------------------------------------- 检查项

def t_registry():
    import tools
    from core.profile import CODING, COMPANION

    # 陪伴：6 个常驻。**顺序即 tools= 字段的渲染顺序**，改顺序 = 缓存纪元失效。
    assert tools.core_names(COMPANION) == ("retrieve_memory", "search_tools",
                                           "compress_context", "summarize_today",
                                           "web_search", "write_file"), \
        tools.core_names(COMPANION)
    schemas = tools.core_schemas(COMPANION)
    assert len(schemas) == 6
    for s in schemas:
        assert s["type"] == "function" and s["function"]["parameters"]["type"] == "object"

    # 编码：常驻的是另一批；哪些工具常驻由 Profile 决定。
    assert "read_file" in tools.core_names(CODING)
    assert "retrieve_memory" not in tools.core_names(CODING)

    # 同一个工具的"可检索性"随 Profile 变，有两条排除理由：它是本 Profile 的常驻工具，
    # 或它的包不在白名单里。
    comp_searchable = {s.name for s in tools.searchable_specs(COMPANION)}
    code_searchable = {s.name for s in tools.searchable_specs(CODING)}
    assert "read_file" in comp_searchable          # 陪伴不常驻它 → 可检索
    assert "read_file" not in code_searchable      # 理由一：编码常驻它
    assert "retrieve_memory" not in code_searchable  # 理由二：memory 包不在编码白名单
    assert "retrieve_memory" not in comp_searchable  # 理由一：陪伴常驻它
    # 判别式：web_search 在陪伴里因"常驻"被排除，在编码里因"包允许且不常驻"而可见，
    # 一次覆盖两条排除理由的不同分支。
    assert "web_search" not in comp_searchable
    assert "web_search" in code_searchable

    # 工具包索引跟着变：fs 包的工具在编码 Profile 里全部常驻 → 索引里不该再出现
    assert "fs" in [p for p, _ in tools.pack_index(COMPANION)]
    assert "fs" not in [p for p, _ in tools.pack_index(CODING)]
    assert "memory" not in [p for p, _ in tools.pack_index(CODING)], \
        "编码 Profile 不该看得见记忆包 —— 隔离是双向的"

    return (f"陪伴常驻 {len(tools.core_names(COMPANION))} 个 / "
            f"编码常驻 {len(tools.core_names(CODING))} 个 / 全部 {len(tools.REGISTRY)} 个")

def t_profile():
    """Profile 机制本身。"""
    import core.profile as P
    from tools import core_schemas

    assert P.active().name == "companion", "没激活时应当默认陪伴 Profile"
    assert len(P.BY_NAME) == 2 and set(P.BY_NAME) == {"companion", "coding"}

    # 会话归属：陪伴是 NULL（老数据的判据），编码是 "coding"
    assert P.COMPANION.conversation_source is None
    assert P.CODING.conversation_source == "coding"

    # 隔离级别：编码入库但不进索引、不进记忆
    assert P.CODING.index_messages is False
    assert P.CODING.memory_eligible is False
    assert P.COMPANION.index_messages is True and P.COMPANION.memory_eligible is True

    # 沙箱根不同，且编码的根必须是**已存在**的目录（不自动创建）
    from tools.sandbox import workspace
    ws_comp = workspace(P.COMPANION)
    ws_code = workspace(P.CODING)
    assert ws_comp != ws_code, (ws_comp, ws_code)
    assert ws_code.is_dir(), ws_code

    # activate / deactivate 必须成对还原，否则会污染同进程里的其它调用
    tok = P.activate(P.CODING)
    try:
        assert P.active().name == "coding"
        assert len(core_schemas()) == len(P.CODING.core_tools)
    finally:
        P.deactivate(tok)
    assert P.active().name == "companion"

    # --workspace 覆盖：必须是已存在的目录，且不就地改单例。
    # 探测目录用 DATA_DIR 下的可写目录，不走系统临时目录。
    from config import DATA_DIR
    probe_dir = DATA_DIR / "_smoke_ws_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    try:
        derived = P.with_workspace(P.CODING, probe_dir)
        from pathlib import Path as _P
        assert derived.workspace_roots == (_P(probe_dir).resolve(),)
        assert P.CODING.workspace_roots != derived.workspace_roots, "派生不能就地改单例"
        assert P.CODING.label == "coding", "派生不能污染单例的 label"
    finally:
        probe_dir.rmdir()

    try:
        P.with_workspace(P.CODING, str(DATA_DIR / "_definitely_not_here_zzz"))
        raise AssertionError("不存在的 --workspace 应当报错")
    except NotADirectoryError:
        pass

    return "两个 Profile 的工具集/工具包/沙箱/会话归属互相隔离，激活可还原"

def t_param_validation():
    from tools import execute_tool
    out = execute_tool("retrieve_memory", {})                 # 缺必填
    assert not out.ok and "query" in out.content, out.content
    out2 = execute_tool("no_such_tool", {"a": 1})
    assert not out2.ok and "search_tools" in out2.content, out2.content
    return "缺参/未知工具均回灌可读错误"

def t_sandbox():
    from tools import execute_tool
    from tools.sandbox import workspace
    out = execute_tool("write_file", {"path": "notes/hello.md", "content": "你好，世界"})
    assert out.ok, out.content
    out2 = execute_tool("read_file", {"path": "notes/hello.md"})
    assert out2.ok and "你好" in out2.content, out2.content
    out3 = execute_tool("write_file", {"path": "../../escape.txt", "content": "x"})
    assert not out3.ok and "越界" in out3.content, out3.content
    (workspace() / "notes" / "hello.md").unlink(missing_ok=True)
    return "写入/读取正常，越界被拒"

def t_file_tools():
    """文件工具：读的行号/分页、编辑的九种失败模式、行尾保真。

    逐条覆盖失败模式而不只测 happy path —— 编辑是破坏性操作，
    失败路径才是这个工具的主要难度。
    """
    from core import readset
    from tools import execute_tool
    from tools.sandbox import workspace

    ws = workspace()
    notes = ws / "notes"
    notes.mkdir(parents=True, exist_ok=True)
    readset.reset()

    def w(name, content, *, newline="\n"):
        raw = content.replace("\n", newline).encode("utf-8")
        (notes / name).write_bytes(raw)
        return notes / name

    def read(name, **kw):
        return execute_tool("read_file", {"path": f"notes/{name}", **kw})

    def edit(name, old, new, **kw):
        return execute_tool("edit_file", {"path": f"notes/{name}",
                                          "old_string": old, "new_string": new, **kw})

    steps = []

    # ---------- read_file：行号 + 分页 ----------
    body = "\n".join(f"第 {i} 行" for i in range(1, 51))
    w("r1.txt", body + "\n")
    out = read("r1.txt")
    assert out.ok, out.content
    assert "共 50 行" in out.content and "\t第 1 行" in out.content, out.content
    assert "第 1 行" in out.content.split("\n")[1], "行号应当和内容在同一行（cat -n 风格）"

    out = read("r1.txt", offset=41)
    assert "显示 41-50" in out.content, out.content
    assert "还有" not in out.content, "读到末尾了不该说还有"
    out = read("r1.txt", offset=1, limit=10)
    assert "显示 1-10" in out.content and "还有 40 行" in out.content, out.content
    steps.append("行号/分页")

    # ---------- 分页读一半 → 不算"读过" → 拒绝编辑 ----------
    # 必须用**从没被完整读过**的文件：完整读过的文件测不出这一条。
    w("r5.txt", body + "\n")
    out = read("r5.txt", offset=1, limit=10)
    assert "还有 40 行" in out.content, out.content
    out = edit("r5.txt", "第 3 行", "第 3 行（改）")
    assert not out.ok and "还没被读过" in out.content, out.content
    assert (notes / "r5.txt").read_text(encoding="utf-8").startswith("第 1 行"), "文件不该被动"
    steps.append("读一半不算读过")

    # ---------- read_file：长行截断 / 二进制 / 编码 ----------
    w("r2.txt", "x" * 3000 + "\n")
    out = read("r2.txt")
    assert "本行还有 1000 字符" in out.content, out.content[:200]
    (notes / "r3.bin").write_bytes(b"abc\x00def")
    out = read("r3.bin")
    assert not out.ok and "二进制" in out.content, out.content
    (notes / "r4.txt").write_bytes("正常".encode("utf-8") + b"\xff\xfe" + "尾巴".encode("utf-8"))
    out = read("r4.txt")
    assert out.ok and "可能不是 UTF-8" in out.content, out.content[:200]
    steps.append("长行/二进制/编码")

    # ---------- edit_file：前置拒绝 ----------
    w("e1.txt", "alpha\nbeta\ngamma\n")
    read("e1.txt")

    out = edit("e1.txt", "", "x")
    assert not out.ok and "不能为空" in out.content, out.content

    out = edit("e1.txt", "beta", "beta")
    assert not out.ok and "一模一样" in out.content, out.content

    out = edit("e1.txt", "beta", "BETA")
    assert out.ok and "已编辑" in out.content, out.content
    assert (notes / "e1.txt").read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    steps.append("空/无操作/正常替换")

    # ---------- 读后写：读完之后被外部改 ----------
    read("e1.txt")
    (notes / "e1.txt").write_text("alpha\nBETA\ngamma\n# 外部插了一行\n", encoding="utf-8")
    out = edit("e1.txt", "BETA", "beta")
    assert not out.ok and "被改动过了" in out.content, out.content
    steps.append("读后被改 → 拒绝")

    # ---------- 找不到 → 给最相似的一段 ----------
    read("e1.txt")
    out = edit("e1.txt", "alpha\nBETA\ngamma\n# 外部插了一行\n# 不存在的行", "x")
    assert not out.ok and "找不到" in out.content, out.content
    assert "最相似" in out.content or "原文是" in out.content, out.content[:400]
    steps.append("找不到 → 给线索")

    # ---------- 缩进不一致 → 说清原因（不做模糊替换） ----------
    w("e2.txt", "def f():\n    return 1\n")
    read("e2.txt")
    out = edit("e2.txt", "def f():\n  return 1", "def f():\n    return 2")
    assert not out.ok, "缩进不一致不该被静默替换"
    assert "缩进" in out.content and "制表符" in out.content, out.content[:400]
    assert (notes / "e2.txt").read_text(encoding="utf-8") == "def f():\n    return 1\n", \
        "文件不该被改动"
    steps.append("缩进不一致 → 拒绝并解释")

    # ---------- 出现多次 → 列行号，不猜 ----------
    w("e3.txt", "x = 1\ny = 2\nx = 1\nz = 3\nx = 1\n")
    read("e3.txt")
    out = edit("e3.txt", "x = 1", "x = 9")
    assert not out.ok and "出现了 3 次" in out.content, out.content
    assert "1, 3, 5" in out.content, out.content
    assert (notes / "e3.txt").read_text(encoding="utf-8").count("x = 1") == 3, "不该动文件"
    out = edit("e3.txt", "x = 1", "x = 9", replace_all=True)
    assert out.ok and (notes / "e3.txt").read_text(encoding="utf-8").count("x = 9") == 3
    steps.append("多次出现 → 列行号 / replace_all")

    # ---------- 行尾保真：CRLF 文件用 LF 的 old_string 也能改，且写回仍是 CRLF ----------
    w("crlf.txt", "one\ntwo\nthree\n", newline="\r\n")
    assert b"\r\n" in (notes / "crlf.txt").read_bytes()
    read("crlf.txt")
    out = edit("crlf.txt", "two", "TWO")          # 注意：old_string 用的是 LF
    assert out.ok, out.content
    raw = (notes / "crlf.txt").read_bytes()
    assert raw == b"one\r\nTWO\r\nthree\r\n", raw
    steps.append("CRLF 保真")

    # ---------- 编辑后语法坏 → 成功但警告，不回滚 ----------
    w("s1.py", "def f():\n    return 1\n")
    read("s1.py")
    out = edit("s1.py", "    return 1", "    return (1")
    assert out.ok, "语法坏不该阻断写入（模型可能是有意写半成品）"
    assert "语法检查没通过" in out.content, out.content
    assert "return (1" in (notes / "s1.py").read_text(encoding="utf-8")
    steps.append("语法坏 → 警告但不回滚")

    # ---------- write_file：create / overwrite / append ----------
    out = execute_tool("write_file", {"path": "notes/w1.txt", "content": "新的\n",
                                      "mode": "create"})
    assert out.ok and "已新建" in out.content, out.content
    out = execute_tool("write_file", {"path": "notes/w1.txt", "content": "又是新的\n",
                                      "mode": "create"})
    assert not out.ok and "已经存在" in out.content, out.content
    out = execute_tool("write_file", {"path": "notes/w1.txt", "content": "A\nB\nC\n"})
    assert out.ok and "1 行 → 3 行" in out.content, out.content
    out = execute_tool("write_file", {"path": "notes/w1.txt", "content": "D\n",
                                      "mode": "append"})
    assert out.ok and "已追加" in out.content, out.content
    assert (notes / "w1.txt").read_text(encoding="utf-8") == "A\nB\nC\nD\n"
    steps.append("write_file create/overwrite/append")

    for name in ("r1.txt", "r2.txt", "r3.bin", "r4.txt", "r5.txt", "e1.txt", "e2.txt",
                 "e3.txt", "crlf.txt", "s1.py", "w1.txt"):
        (notes / name).unlink(missing_ok=True)
    readset.reset()
    return "；".join(steps)

def t_code_tools():
    """代码检索：glob_files / grep_search / code_symbols。

    重点测**不该被搜到的东西**（.gitignore、硬编码忽略、二进制、超大文件），
    以及**失败时说的话**（坏正则、非 Python、语法错的文件）。
    """
    import shutil as _sh
    from tools import execute_tool
    from tools.sandbox import workspace

    ws = workspace()
    demo = ws / "_codedemo"
    _sh.rmtree(demo, ignore_errors=True)
    (demo / "sub").mkdir(parents=True)
    (demo / "build").mkdir()
    (demo / "node_modules").mkdir()

    (demo / ".gitignore").write_text("ignored.txt\n*.log\n", encoding="utf-8")
    (demo / "a.py").write_text(
        '"""模块说明。"""\n'
        'MARKER = "codeonly-3f7a"\n\n'
        'def alpha_fn(x: int) -> int:\n'
        '    """把 x 加一。"""\n'
        '    return x + 1\n\n'
        'class Alpha:\n'
        '    """一个类。"""\n\n'
        '    def method_one(self, y):\n'
        '        """方法一的说明。"""\n'
        '        return alpha_fn(y)\n', encoding="utf-8")
    (demo / "b.py").write_text("from a import alpha_fn\n\nVALUE = alpha_fn(1)\n",
                               encoding="utf-8")
    (demo / "sub" / "c.py").write_text("MARKER = 'codeonly-3f7a'\n", encoding="utf-8")
    (demo / "notes.md").write_text("# 笔记\n\n这里也有 codeonly-3f7a 这个标记\n",
                                   encoding="utf-8")
    (demo / "ignored.txt").write_text("codeonly-3f7a 在被忽略的文件里\n", encoding="utf-8")
    (demo / "build" / "gen.py").write_text("MARKER = 'codeonly-3f7a'\n", encoding="utf-8")
    (demo / "node_modules" / "x.js").write_text("// codeonly-3f7a\n", encoding="utf-8")
    (demo / "trace.log").write_text("codeonly-3f7a\n", encoding="utf-8")
    (demo / "blob.bin").write_bytes(b"\x00\x01codeonly-3f7a\x00")
    (demo / "broken.py").write_text("def f(:\n    pass\n", encoding="utf-8")

    def call(tool, **kw):
        return execute_tool(tool, kw)

    steps = []

    # ---------- glob_files ----------
    out = call("glob_files", pattern="**/*.py", path="_codedemo")
    assert out.ok, out.content
    assert "a.py" in out.content and "sub/c.py" in out.content, out.content
    assert "build/gen.py" not in out.content, "build/ 应当被硬编码忽略"
    assert "node_modules" not in out.content, "node_modules 应当被忽略"
    hits = [ln for ln in out.content.splitlines() if "  (" in ln]
    assert hits == sorted(hits, key=lambda s: s.split(", ")[-1].rstrip(")"), reverse=True) \
        or len(hits) < 2, "应当按修改时间倒序"
    out = call("glob_files", pattern="*.md", path="_codedemo")
    assert "notes.md" in out.content, out.content
    out = call("glob_files", pattern="**/*.nomatch", path="_codedemo")
    assert "没找到" in out.content, out.content
    steps.append("glob 模式/忽略/排序")

    # ---------- grep_search：该找到的找到，该忽略的忽略 ----------
    out = call("grep_search", pattern="codeonly-3f7a", path="_codedemo")
    assert out.ok, out.content
    for should in ("a.py", "sub/c.py", "notes.md"):
        assert should in out.content, f"{should} 本该被搜到：{out.content[:300]}"
    for shouldnt in ("ignored.txt", "trace.log", "build/gen.py", "node_modules",
                     "blob.bin"):
        assert shouldnt not in out.content, f"{shouldnt} 本该被忽略：{out.content[:300]}"
    assert "跳过 1 个二进制文件" in out.content, f"二进制跳过要明说：{out.content}"
    steps.append("命中/忽略/二进制计数")

    # ---------- 三种 output_mode ----------
    out = call("grep_search", pattern="MARKER", path="_codedemo", output_mode="files")
    assert "涉及" in out.content and ":" not in out.content.split("个文件")[0], out.content
    out = call("grep_search", pattern="MARKER", path="_codedemo", output_mode="count")
    assert "处" in out.content and "a.py" not in out.content, out.content
    steps.append("content/files/count")

    # ---------- ignore_case / context_lines / 坏正则 ----------
    out = call("grep_search", pattern="CODESONLY|codeonly", path="_codedemo")
    lc = out.content.count("处")
    out = call("grep_search", pattern="CODESONLY|codeonly", path="_codedemo",
               ignore_case=True)
    assert "没有命中" not in out.content, out.content
    out = call("grep_search", pattern="return x \\+ 1", path="_codedemo",
               context_lines=2)
    assert "def alpha_fn" in out.content, f"上下文行没带上：{out.content[:400]}"
    out = call("grep_search", pattern="([", path="_codedemo")
    assert not out.ok and "正则写错了" in out.content, out.content
    steps.append("大小写/上下文/坏正则")

    # ---------- .py 的命中要带"这行属于哪个定义" ----------
    out = call("grep_search", pattern="alpha_fn", path="_codedemo", glob="*.py")
    assert "«在 " in out.content, f"grep 结果少了归属标注：{out.content}"
    assert "«在 method_one 里»" in out.content, \
        f"调用点的归属应当是 method_one：{out.content}"
    out = call("grep_search", pattern="codeonly-3f7a", path="_codedemo/notes.md")
    assert "«在 " not in out.content, "非 .py 文件不该加归属标注"
    steps.append("命中归属标注")

    # ---------- code_symbols ----------
    out = call("code_symbols", action="outline", path="_codedemo/a.py")
    assert out.ok, out.content
    assert "class Alpha" in out.content and "def alpha_fn" in out.content, out.content
    assert "method_one" in out.content, "类方法也要列出来"
    assert "把 x 加一" in out.content, f"首行 docstring 该带上：{out.content}"
    assert "broken.py" not in out.content

    out = call("code_symbols", action="outline", path="_codedemo")
    assert "broken.py: 解析失败" in out.content, f"语法错的文件要报出来而不是崩：{out.content}"

    out = call("code_symbols", action="find", name="alpha_fn", path="_codedemo")
    assert "a.py" in out.content and "def alpha_fn" in out.content, out.content
    out = call("code_symbols", action="find", name="Alpha.method_one", path="_codedemo")
    assert "Alpha.method_one" in out.content, out.content
    out = call("code_symbols", action="find", name="根本没这个", path="_codedemo")
    assert "没找到" in out.content and "getattr" in out.content, \
        f"找不到时要说明局限：{out.content}"

    out = call("code_symbols", action="references", name="alpha_fn", path="_codedemo")
    assert "文本匹配" in out.content, f"引用搜索必须自报不准：{out.content[:300]}"
    assert "b.py" in out.content, out.content
    steps.append("outline/find/references/局限声明")

    # ---------- 非 Python ----------
    out = call("code_symbols", action="outline", path="_codedemo/notes.md")
    assert not out.ok and "没有 .py 文件" in out.content, out.content
    steps.append("非 Python 拒绝")

    _sh.rmtree(demo, ignore_errors=True)
    return "；".join(steps)

def t_exec_tools():
    """执行工具：必然挂死拦截、危险审批、超时杀进程树、输出截断。

    这里**不真的执行危险命令**（rm -rf 之类）。审批层的验证靠替换 asker：
    拒绝路径走一遍、放行路径只跑无害的等价物。
    """
    import shutil as _sh
    import sys as _sys
    import time as _time

    from core import approval
    from tools import execute_tool
    from tools.sandbox import workspace

    ws = workspace()
    demo = ws / "_execdemo"
    _sh.rmtree(demo, ignore_errors=True)
    demo.mkdir(parents=True)
    py = _sys.executable

    (demo / "hang.py").write_text(
        "import subprocess, sys, time\n"
        "# 子进程 6 秒后写标记文件；父进程赖着不走。\n"
        "# 只杀父进程的话子进程会活下来把标记写出来。\n"
        "subprocess.Popen([sys.executable, '-c',\n"
        "    \"import time;time.sleep(6);open('marker.txt','w').write('alive')\"])\n"
        "time.sleep(300)\n", encoding="utf-8")
    (demo / "noisy.py").write_text(
        "for i in range(3000):\n"
        "    print(f'第 {i} 行填充填充填充填充填充填充填充填充填充填充')\n"
        "print('★最后一行是关键★')\n", encoding="utf-8")
    (demo / "badenc.py").write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'good ' + b'\\xff\\xfe\\xff' + b' tail\\n')\n"
        "sys.stdout.buffer.flush()\n", encoding="utf-8")
    (demo / "sleeper.txt").write_text("x", encoding="utf-8")

    def sh(cmd, **kw):
        return execute_tool("run_shell", {"command": cmd, "cwd": "_execdemo", **kw})

    steps = []

    # ---------- ① 必然挂死：直接拒，而且**不该问人** ----------
    asked = []
    approval.use_asker(lambda req: (asked.append(req.summary), False)[1])
    try:
        for cmd, kw in [("vim notes.txt", "编辑器"), ("less big.log", "分页器"),
                        ("gdb ./a.out", "调试器"), ("tail -f app.log", "follow"),
                        ("python", "裸解释器"), ("nohup ./run.sh &", "脱离管控"),
                        ("python -m venv .venv", "建虚拟环境")]:
            out = sh(cmd)
            assert not out.ok and "不能执行" in out.content, f"{cmd} → {out.content[:200]}"
        assert not asked, f"必然挂死的命令不该走审批，却问了 {len(asked)} 次：{asked}"
    finally:
        approval.use_asker(None)
    steps.append("挂死命令直接拒且不问人")

    # ---------- ② 危险：问；拒绝时回灌可读的话，不抛异常 ----------
    seen = []
    approval.use_asker(lambda req: (seen.append(req.key), False)[1])
    try:
        out = sh("rm -rf build")
        assert not out.ok and "用户拒绝" in out.content, out.content
        assert seen and seen[-1] == "rm-recursive", seen
        out = sh("git push --force origin main")
        assert not out.ok and "用户拒绝" in out.content, out.content
        assert seen[-1] == "git-force-push", seen
        out = sh("git reset --hard HEAD~3")
        assert not out.ok and seen[-1] == "git-destroy", seen
    finally:
        approval.use_asker(None)
    steps.append("危险命令走审批、拒绝可读")

    # ---------- ③ 放行时真的执行（用无害等价物） ----------
    approval.use_asker(lambda req: True)
    try:
        out = sh(f'"{py}" -c "print(1+1)"')
        assert out.ok and "2" in out.content, out.content
        assert "退出码 0" in out.content, out.content
    finally:
        approval.use_asker(None)
    steps.append("放行后执行 + 报退出码")

    # ---------- ③b PowerShell 的两处修补 ----------
    # ① 语句开头的引号串在 PowerShell 里是"字符串表达式"不是命令名，工具替模型补上 `&`。
    out = sh(f'"{py}" -c "print(2+2)"')
    assert out.ok and "4" in out.content, \
        f"带引号的命令名没被修好（PowerShell 需要 `&`）：{out.content[:300]}"
    # ② PowerShell 会把原生命令的退出码归一化成 1，工具补一段收尾保住真实值。
    out = sh(f'"{py}" -c "import sys;sys.exit(7)"')
    assert "退出码 7" in out.content, \
        f"真实退出码被 PowerShell 吞了：{out.content[:200]}"
    # ③ 失败的 cmdlet 也不能报成成功：`$LASTEXITCODE` 只对原生命令有效，
    #    cmdlet 失败时它是空的 → `exit $null` = 退 0。
    out = sh('Set-Content -Path "Z:\\nope\\x.txt" -Value hi')
    assert not out.ok and "退出码 1" in out.content, \
        f"失败的 cmdlet 被报成了成功：{out.content[:200]}"
    out = sh("Get-ChildItem . > $null")
    assert out.ok, f"成功的 cmdlet 不该被判成失败：{out.content[:200]}"
    steps.append("自动补 `&` 与退出码收尾")

    # ---------- ④ 只读白名单：ALL_WRITES 档下不该问 ----------
    import core.profile as P
    from dataclasses import replace as _replace
    comp_strict = _replace(P.COMPANION, approval="all_writes")
    tok = P.activate(comp_strict)
    asked2 = []
    approval.use_asker(lambda req: (asked2.append(req.key), True)[1])
    try:
        out = sh(f'"{py}" --version')       # 不在白名单里 → 该问
        assert asked2, "ALL_WRITES 下非只读命令该问"
        asked2.clear()
        out = sh("echo hello")
        assert out.ok and not asked2, f"echo 是只读，不该问：{asked2}"
    finally:
        approval.use_asker(None)
        P.deactivate(tok)
    steps.append("ALL_WRITES 只读白名单")

    # ---------- ⑤ 超时：杀**整棵进程树** ----------
    marker = demo / "marker.txt"
    marker.unlink(missing_ok=True)
    t0 = _time.time()
    out = sh(f'"{py}" hang.py', timeout_s=3)
    elapsed = _time.time() - t0
    assert "超时" in out.content and "杀掉整棵进程树" in out.content, out.content[:300]
    assert elapsed < 25, f"超时处理太慢：{elapsed:.1f}s"
    _time.sleep(8)                    # 够那个孙进程跑完它的 6 秒
    assert not marker.exists(), \
        "孙进程活下来了（写出来了 marker.txt）→ 没杀干净进程树，它会占着端口"
    steps.append("超时杀整棵进程树")

    # ---------- ⑥ 头尾截断：尾巴必须在 ----------
    out = sh(f'"{py}" noisy.py')
    assert out.ok, out.content[:200]
    assert "★最后一行是关键★" in out.content, "尾部被丢了 —— 报错和失败摘要都在尾部"
    assert "省略" in out.content, "该明确说省略了多少"
    assert len(out.content) < 8000, f"回灌太长了：{len(out.content)}"
    steps.append("头尾截断保留尾部")

    # ---------- ⑦ stdout / stderr 分开；编码异常要明说 ----------
    out = sh(f'"{py}" -c "import sys;print(\'out行\');print(\'err行\',file=sys.stderr)"')
    assert "--- stdout ---" in out.content and "out行" in out.content, out.content
    assert "--- stderr ---" in out.content and "err行" in out.content, out.content
    out = sh(f'"{py}" badenc.py')
    assert "不可信" in out.content and "U+FFFD" in out.content, \
        f"坏编码必须明说而不是静默交给模型：{out.content[:300]}"
    steps.append("stdout/stderr 分开 + 编码告警")

    # ---------- ⑧ 失败命令的退出码 ----------
    out = sh(f'"{py}" -c "import sys;sys.exit(3)"')
    assert not out.ok and "退出码 3" in out.content, out.content
    steps.append("非零退出码")

    # ---------- ⑨ git：只读不问、写要问、参数走数组 ----------
    # 必须激活**编码** Profile：陪伴 Profile 的沙箱是 data/workspace，不含项目根。
    seen3 = []
    approval.use_asker(lambda req: (seen3.append(req.key), False)[1])
    tok_git = P.activate(P.with_workspace(P.CODING, ROOT_DIR))
    try:
        out = execute_tool("git", {"subcommand": "status", "cwd": "."})
        assert not seen3, f"git status 是只读，不该问：{seen3}"
        assert "退出码" in out.content, out.content
        out = execute_tool("git", {"subcommand": "commit", "args": ["-m", "x"], "cwd": "."})
        assert not out.ok and seen3 == ["git:commit"], (out.content[:200], seen3)
    finally:
        P.deactivate(tok_git)
        approval.use_asker(None)
    steps.append("git 只读直通/写走审批")

    # ---------- ⑩ agent/ 自我修改保护 + 覆盖前先读 ----------
    coding = P.with_workspace(P.CODING, ROOT_DIR)
    tok2 = P.activate(coding)
    try:
        out = execute_tool("write_file", {"path": "agent/_probe.py", "content": "x=1\n",
                                          "mode": "create"})
        assert not out.ok and "受保护" in out.content, \
            f"编码 Profile 不该能写 agent/：{out.content[:200]}"
        out = execute_tool("edit_file", {"path": "agent/core/loop.py", "old_string": "a",
                                         "new_string": "b"})
        assert not out.ok and "受保护" in out.content, out.content[:200]
        # 读过之后仍然受保护，且理由必须还是"受保护"而不是"先去读一次"。
        execute_tool("read_file", {"path": "agent/config.py"})
        out = execute_tool("edit_file", {"path": "agent/config.py",
                                         "old_string": "CTX_WINDOW", "new_string": "X"})
        assert not out.ok and "受保护" in out.content, \
            f"读过之后也该拒绝写，且理由要说'受保护'：{out.content[:200]}"
        # 保护只挡写，不挡读
        out = execute_tool("read_file", {"path": "agent/config.py"})
        assert out.ok, "agent/ 应当只读而不是不可读"
    finally:
        P.deactivate(tok2)
    steps.append("agent/ 写保护（读不受影响）")

    # 覆盖已存在文件前必须先读
    readset.reset()
    from tools import execute_tool as _et
    out = _et("write_file", {"path": "_execdemo/sleeper.txt", "content": "覆盖",
                             "mode": "overwrite"})
    assert not out.ok and "还没被读过" in out.content, out.content[:250]
    _et("read_file", {"path": "_execdemo/sleeper.txt"})
    out = _et("write_file", {"path": "_execdemo/sleeper.txt", "content": "覆盖\n"})
    assert out.ok, out.content
    steps.append("覆盖前必须先读")

    _sh.rmtree(demo, ignore_errors=True)
    readset.reset()
    return "；".join(steps)

def t_plan_tools():
    """任务编排：todo_write 的校验、渲染，以及不收敛护栏。"""
    import core.plan as plan_store
    from core.context import build_notice
    from core.loop import _STALL_LIMIT, _made_progress, get_session
    from tools import execute_tool

    cid = get_session().conversation_id or 0
    plan_store.clear(cid)
    steps = []

    def todo(items, **kw):
        return execute_tool("todo_write", {"items": items, **kw})

    # ---------- 正常写入 + 渲染 ----------
    out = todo([{"content": "读代码", "status": "completed"},
                {"content": "改函数", "status": "in_progress"},
                {"content": "跑测试", "status": "pending"}])
    assert out.ok, out.content
    assert "3 项" in out.content and "1 完成" in out.content, out.content
    assert "✅ 读代码" in out.content and "🔄 改函数" in out.content, out.content
    saved = plan_store.load(cid)
    assert len(saved) == 3 and saved[1]["status"] == "in_progress", saved
    steps.append("写入与渲染")

    # ---------- 核心约束：同时最多一个 in_progress ----------
    out = todo([{"content": "A", "status": "in_progress"},
                {"content": "B", "status": "in_progress"}])
    assert not out.ok and "最多" in out.content and "一个" in out.content, out.content
    assert len(plan_store.load(cid)) == 3, "被拒绝的调用不该改动已存的计划"
    steps.append("拒绝多个 in_progress")

    # ---------- 中文状态名要能救回来（提示词是中文的） ----------
    out = todo([{"content": "甲", "status": "已完成"},
                {"content": "乙", "status": "进行中"},
                {"content": "丙", "status": "待办"}])
    assert out.ok, out.content
    got = {x["content"]: x["status"] for x in plan_store.load(cid)}
    assert got == {"甲": "completed", "乙": "in_progress", "丙": "pending"}, got
    steps.append("中文状态别名")

    # ---------- 坏输入要可读地拒绝 ----------
    out = todo([])
    assert not out.ok and "空" in out.content, out.content
    out = todo(["直接给字符串"])
    assert not out.ok and "不是对象" in out.content, out.content
    out = todo([{"status": "pending"}])
    assert not out.ok and "content" in out.content, out.content
    out = todo([{"content": "x", "status": "瞎写"}])
    assert not out.ok and "只能是" in out.content, out.content
    steps.append("坏输入可读拒绝")

    # ---------- 全量替换语义 ----------
    todo([{"content": "只剩这一条", "status": "pending"}])
    assert [x["content"] for x in plan_store.load(cid)] == ["只剩这一条"]
    steps.append("全量替换")

    # ---------- 渲染进尾部通知 ----------
    text = plan_store.render(plan_store.load(cid))
    notice = build_notice(ctx_tokens=100, ctx_ratio=0.1, plan_text=text)
    assert "【当前任务】" in notice and "只剩这一条" in notice, notice
    assert build_notice(ctx_tokens=100, ctx_ratio=0.1, plan_text="") == \
        build_notice(ctx_tokens=100, ctx_ratio=0.1), "空计划不该占任何 token"
    steps.append("渲染进尾部")

    # ---------- 数组参数被字符串化也要能接住 ----------
    # pydantic 的 lax 模式救不了 '["x"]' → list，必须在归一化层还原。
    out = execute_tool("todo_write",
                       {"items": '[{"content": "从字符串来", "status": "pending"}]'})
    assert out.ok and "从字符串来" in out.content, \
        f"被字符串化的数组没还原：{out.content[:200]}"
    steps.append("字符串化数组还原")

    # ---------- 不收敛护栏的判据 ----------
    from core import readset
    readset.reset()
    assert not _made_progress({}, [("run_shell", True, {"command": "ls"})], 0), \
        "只读命令不该算进展，否则反复 ls 就能让护栏永不触发"
    assert _made_progress({}, [("run_shell", True, {"command": "rm -rf x"})], 0), \
        "有副作用的命令该算进展"
    assert _made_progress({}, [("write_file", True, {})], 0), "写文件算进展"
    assert not _made_progress({}, [("read_file", True, {})], 0), \
        "重读同一个文件不算进展"
    steps.append(f"不收敛判据（阈值 {_STALL_LIMIT} 步）")

    # ---------- 陈旧计划不能泄漏到下一件事 ----------
    # 编码会话是持久化的，上一轮已完成的计划会留到下一轮，让模型以为任务已经做完。
    from core.loop import _drop_finished_plan
    plan_store.save(cid, [{"content": "上一件事", "status": "completed"}])
    assert _drop_finished_plan(cid) == [], "已完成的旧计划必须被丢掉"
    assert plan_store.load(cid) == [], "而且要真的删掉，不只是这次不返回"
    plan_store.save(cid, [{"content": "上一件事", "status": "in_progress"}])
    assert len(_drop_finished_plan(cid)) == 1, "没做完的计划该保留（那确实还没做完）"
    steps.append("陈旧计划不跨任务")

    plan_store.clear(cid)
    readset.reset()
    return "；".join(steps)

def t_conversation_identity():
    """会话身份按 `(source, 工作区)` 分流。

    `--workspace A` 换到 `--workspace B` 时，A 的历史不能被装进 B 的会话。
    """
    import subprocess
    import sys as _sys

    import core.profile as P
    from core.loop import Session

    steps = []

    # ---------- ① Profile 层的身份 ----------
    assert P.COMPANION.conversation_workspace() is None, \
        "陪伴不该按工作区区分 —— 它那条主对话有几千条历史，切了会被劈成两条"
    assert P.COMPANION.conversation_key() == (None, None), P.COMPANION.conversation_key()

    # 两个工作区都从当前仓库推出来，不写死绝对路径 —— 否则换台机器克隆就挂。
    repo_root = Path(BASE_DIR).parent.resolve()
    other_root = repo_root.parent / "test_space"

    a = P.with_workspace(P.CODING, str(other_root))
    b = P.with_workspace(P.CODING, str(repo_root))
    assert a.conversation_key() != b.conversation_key(), \
        "不同工作区必须得到不同身份，否则还是串台"
    assert b.conversation_key() == P.CODING.conversation_key(), \
        "同一个工作区必须得到同一个身份"
    # 路径要先规范化，否则 <root> 和 <root>\..\<name> 会被当成两个项目
    c = P.with_workspace(P.CODING, str(repo_root / ".." / repo_root.name))
    assert c.conversation_key() == b.conversation_key(), \
        f"路径没规范化：{c.conversation_key()} != {b.conversation_key()}"
    assert a.conversation_title() and "test_space" in a.conversation_title()
    steps.append("身份 = (source, 规范化工作区)")

    # ---------- ② Session 守卫要比完整身份 ----------
    # 只比 profile.name 的话，换工作区会被判成"可复用"→ 跳过 preload
    s = Session(profile_name="coding", conversation_key=P.CODING.conversation_key(),
                ready=True)
    assert s.usable_for(P.CODING), "同工作区应当可复用"
    assert not s.usable_for(a), "换了工作区就不该复用！"
    assert not s.usable_for(P.COMPANION), "换了 Profile 也不该复用"
    steps.append("Session 守卫比完整身份")

    # ---------- ③ DB 层：轻量迁移 + 匹配（子进程 + 临时库，不碰真库）----------
    probe = DATA_DIR / "_smoke_identity.db"
    probe.unlink(missing_ok=True)
    import sqlite3
    # 造一个**老库**：conversation 表没有 workspace 列
    conn = sqlite3.connect(str(probe))
    conn.execute("CREATE TABLE conversation ("
                 "id INTEGER PRIMARY KEY, title VARCHAR(256), "
                 "other_person_id INTEGER, started_at DATETIME, "
                 "ended_at DATETIME, source VARCHAR(64))")
    conn.execute("INSERT INTO conversation (id, title, source) VALUES (1, '老的编码会话', 'coding')")
    conn.execute("INSERT INTO conversation (id, title, source) VALUES (2, '老的陪伴会话', NULL)")
    conn.commit()
    conn.close()

    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT_DIR / 'agent')!r})\n"
        "from store import db\n"
        "db.init_db()\n"
        "print('WS=', db.get_or_create_conversation(source='coding', workspace='X:/p1'))\n"
        "print('WS=', db.get_or_create_conversation(source='coding', workspace='X:/p2'))\n"
        "print('WS=', db.get_or_create_conversation(source='coding', workspace='X:/p1'))\n"
        "print('NOWS=', db.get_or_create_conversation(source='coding'))\n"
        "print('COMP=', db.get_or_create_conversation(source=None))\n"
    )
    import os
    env = dict(os.environ, DB_URL=f"sqlite:///{probe.as_posix()}",
               PYTHONIOENCODING="utf-8")
    r = subprocess.run([_sys.executable, "-X", "utf8", "-c", code],
                       capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode == 0, f"子进程失败：{r.stderr[-800:]}"
    got = [ln.split("=", 1)[1].strip() for ln in r.stdout.splitlines() if "=" in ln]
    ws1, ws2, ws1_again, no_ws, comp = got
    assert ws1 != ws2, f"不同工作区拿到了同一个会话：{ws1}"
    assert ws1 == ws1_again, f"同一工作区没拿到同一个会话：{ws1} vs {ws1_again}"
    assert no_ws == "1", f"workspace=None 该退回老行为、命中老会话 id=1，得到 {no_ws}"
    assert comp == "2", f"陪伴必须还是老会话 id=2，得到 {comp}"

    # 迁移是**加列**，老数据必须完好
    conn = sqlite3.connect(str(probe))
    cols = [x[1] for x in conn.execute("PRAGMA table_info(conversation)")]
    assert "workspace" in cols, f"workspace 列没加上：{cols}"
    rows = list(conn.execute("SELECT id, source, workspace FROM conversation ORDER BY id"))
    assert rows[0] == (1, "coding", None), f"老会话被动过了：{rows[0]}"
    assert rows[1] == (2, None, None), f"老的陪伴会话被动过了：{rows[1]}"
    conn.close()
    probe.unlink(missing_ok=True)
    steps.append("加列迁移不丢数据 + 老会话不被认领")

    return "；".join(steps)

def t_wiki():
    """LLM Wiki 构建管线：幂等、可重建、增量。

    全程用**注入的假 LLM**，不碰 API，所以结论可复现。
    """
    import hashlib
    import json as _json
    import shutil as _sh

    from wiki import pages as WP
    from wiki.build import build, resolve_topics

    root = DATA_DIR / "_smoke_wiki"
    _sh.rmtree(root, ignore_errors=True)
    steps = []

    def fake_llm(prompt: str) -> str:
        if "groups" in prompt:
            return _json.dumps({"groups": [
                ["对话概览", "对话概况", "对话状态", "对话记忆"],
                ["工作", "工作变动"],
            ]}, ensure_ascii=False)
        return "（假综述。）"

    def no_embed(_text):
        """**测试里不碰真嵌入后端**：否则测试会依赖网络、变慢，而且不可复现。
        归并会退到 `fake_llm` 那一档，正好把它也验了。"""
        raise ConnectionError("测试里不调真嵌入")

    # ---------- ① front-matter 往返稳定 ----------
    p = WP.Page(path="concepts/工作.md", title="工作", type="concept",
                aliases=["工作变动"], updated="2026-09-15",
                evidence=[WP.Evidence("memory:13", "2026-09-15", "乙"),
                          WP.Evidence("memory:6", "2026-09-15", "甲"),
                          WP.Evidence("memory:6", "2026-09-15", "甲（重复）")])
    once = p.render()
    twice = WP.parse(once, p.path).render()
    assert once == twice, "渲染 → 解析 → 渲染 不稳定，幂等从根上就不成立"
    # 锚点按**数值**排，不是字符串序（memory:6 要在 memory:13 前面）
    assert once.index("memory:6") < once.index("memory:13"), \
        f"锚点排序是字符串序：{once}"
    # 证据去重
    assert once.count("[memory:6]") == 1, "同一条证据重复写了"
    # **往返必须连"日期带备注"也能过**：备注后缀不能撑破 `- <日期> · [锚点]` 的格式，
    # 否则 `parse()` 读不出锚点。这里用一条带中文备注的正文钉住格式。
    note_p = WP.Page(path="concepts/宠物.md", title="宠物", type="concept",
                     updated="2026-09-15",
                     evidence=[WP.Evidence("memory:16", "2026-09-15",
                                           "（日期取的是入库时间）主人可能养了猫。")])
    rt = WP.parse(note_p.render(), note_p.path)
    assert len(rt.evidence) == 1, \
        f"带备注的证据行读不回来（格式被撑破了）：{note_p.render()}"
    assert rt.evidence[0].date == "2026-09-15", rt.evidence[0]
    assert rt.render() == note_p.render(), "带备注的页往返不稳定"
    steps.append("front-matter 往返稳定 + 锚点数值序 + 证据去重")

    # ---------- ② 主题归并的三档 ----------
    empty = {"pages": {}}
    m = resolve_topics(["工作"], empty, embed=None, ask_llm=None)
    assert m.method == "none" and "各自成页" in m.note, \
        f"两档都不可用时必须**说出来**，不能静默产出重复页：{m}"
    m = resolve_topics(["工作", "工作变动"], empty, embed=None, ask_llm=fake_llm)
    assert m.method == "llm" and m.mapping["工作变动"] == "工作", m.mapping
    idx = {"pages": {"concepts/工作.md": {"title": "工作", "aliases": ["工作变动"]}}}
    m = resolve_topics(["工作变动"], idx, embed=None, ask_llm=None)
    assert m.method == "exact" and m.mapping["工作变动"] == "工作", \
        f"别名必须能精确命中，否则每次都要重跑归并：{m}"
    steps.append("归并三档（exact/llm/none）")

    # ---------- ③ 构建 → 幂等 → 重建 ----------
    def snap():
        return {f.relative_to(root).as_posix():
                hashlib.sha1(f.read_bytes()).hexdigest()
                for f in sorted(root.rglob("*")) if f.is_file()}

    r1 = build(root=root, rebuild=True, ask_llm=fake_llm, embed=no_embed)
    s1 = snap()
    # 先证明构建不是空的，否则后面的幂等断言比的是两个空壳。
    assert r1.new_memories > 0, f"构建是空的，后面的幂等断言没有意义：{r1.summary()}"
    concepts = [k for k in s1 if k.startswith("wiki/concepts/")]
    assert len(concepts) >= 3, f"概念页太少（{len(concepts)}），构建没跑对"

    r2 = build(root=root, ask_llm=fake_llm, embed=no_embed)
    s2 = snap()
    diff = [k for k in set(s1) | set(s2) if s1.get(k) != s2.get(k)]
    assert not diff, f"同一水位线重跑不是逐字节相同：{diff}"
    assert r2.new_memories == 0, "第二次应当没有新记忆"

    r3 = build(root=root, rebuild=True, ask_llm=fake_llm, embed=no_embed)
    s3 = snap()
    diff3 = [k for k in set(s1) | set(s3) if s1.get(k) != s3.get(k)]
    assert not diff3, f"--rebuild 没能重现首次构建（LLM 缓存没生效？）：{diff3}"
    steps.append(f"幂等 + 重建逐字节相同（{len(s1)} 个文件）")

    # ---------- ④ --rebuild 真的要能重置水位线 ----------
    # `save_wiki_state` 有单调性保护，会把重置挡掉，那样"重建"没从 0 跑。
    assert r3.new_memories == r1.new_memories, \
        f"--rebuild 没重置水位线：首次 {r1.new_memories} 条，重建 {r3.new_memories} 条"
    steps.append("--rebuild 能重置水位线")

    # ---------- ⑤ log 去重 ----------
    log = (root / "wiki" / "log.md").read_text(encoding="utf-8")
    assert log.count("- build ·") == 1, \
        f"跑了 3 次构建，log 里却有 {log.count('- build ·')} 行"

    # ---------- ⑥ 增量构建**不能覆盖已有页的证据** ----------
    # `_get_page` 必须按路径取页；按标题取会永远返回 None，于是每次都新建空 Page，
    # 加了新记忆就把旧证据整页覆盖掉。幂等测试走的是"没有新记忆"那条路，测不出它。
    from store.db import get_wiki_state, save_wiki_state
    from wiki.build import _state_key

    def ev_counts():
        return {rel: len(WP.parse((root / rel).read_text(encoding="utf-8"), rel).evidence)
                for rel in concepts}

    before = ev_counts()
    assert sum(before.values()) >= len(concepts), f"起始证据太少，这条测不了：{before}"

    # 把水位线退回去一格，让最后一条记忆被重新处理（模拟"又来了一条新记忆"）
    skey = _state_key(root)
    st = get_wiki_state(skey)
    save_wiki_state(skey, memory_watermark=max(st["memory_watermark"] - 1, 0),
                    message_watermark=st["message_watermark"],
                    page_count=st["page_count"], force=True)
    build(root=root, ask_llm=fake_llm, embed=no_embed)

    after = ev_counts()
    lost = {k: (before[k], after.get(k, 0)) for k in before if after.get(k, 0) < before[k]}
    assert not lost, (
        f"增量构建把已有页的证据弄丢了（页: 之前→之后）：{lost}\n"
        f"最常见的原因是 `_get_page` 拿到的是标题而不是页面路径。")
    assert set(after) == set(before), \
        f"增量构建后页面集合变了：多了 {set(after) - set(before)}，少了 {set(before) - set(after)}"
    steps.append(f"增量构建不丢证据（{sum(before.values())} 条锚点全部保住）")

    # ---------- ⑦ 每个概念页都要挂锚点 ----------
    for rel in concepts:
        page = WP.parse((root / rel).read_text(encoding="utf-8"), rel)
        assert page.evidence, f"{rel} 没有任何证据锚点"
        assert page.sources, f"{rel} 的 front-matter sources 是空的（lint 第 8 项查它）"
        for ev in page.evidence:
            assert ev.ref.startswith("memory:") or ev.ref.startswith("rollup:")
            assert ev.date and ev.date.lower() != "none", \
                f"{rel} 有一条证据的日期是脏值：{ev.render()}"
    steps.append("概念页都挂了锚点且日期干净")

    # ---------- ⑧ 契约是渲染出来的，不是手写的 ----------
    wiki_md = (root / "WIKI.md").read_text(encoding="utf-8")
    assert "自动生成" in wiki_md and "只写 `wiki/`" in wiki_md, "WIKI.md 内容不对"
    assert (root / "scripts" / "lint.py").is_file(), "scripts/ 薄包装没生成"

    _sh.rmtree(root, ignore_errors=True)
    return "；".join(steps)

def t_wiki_lint():
    """lint 九项 —— **必须能失败**。

    一个从不报错的检查器会让一份坏掉的 Wiki 看起来是好的。
    两头都验：**干净的要报 0**（阴性对照），**注入 3 个坏页必须报出 3 个**。
    """
    import shutil as _sh
    from wiki import pages as WP
    from wiki.lint import lint

    root = DATA_DIR / "_smoke_wiki_lint"
    _sh.rmtree(root, ignore_errors=True)
    wiki = root / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "sources").mkdir(parents=True)
    (root / "outputs").mkdir(parents=True)
    steps = []

    def write(rel: str, title: str, *, aliases=(), summary="一句话综述。",
              evidence=(), related=(), ptype="concept"):
        page = WP.Page(path=rel, title=title, type=ptype, aliases=list(aliases),
                       summary=summary, updated="2026-01-01",
                       evidence=[WP.Evidence(r, "2026-01-01", t) for r, t in evidence],
                       related=list(related))
        page.sources = sorted({e.ref for e in page.evidence})
        WP.write_text(wiki / rel, page.render())
        return rel

    def set_index(paths):
        """（测试内部用）把一批路径登记进索引。"""
        WP.save_index(wiki, {"pages": {
            p: {"title": p.split("/")[-1][:-3], "type": "concept",
                "aliases": [], "sources": [], "updated": "2026-01-01"}
            for p in paths}})

    def overview(n):
        WP.write_text(wiki / "overview.md",
                      f"---\ntitle: 总览\ntype: meta\naliases: []\nsources: []\n"
                      f"updated: 2026-01-01\nconfidence: stated\n---\n\n"
                      f"# 总览\n\n## Health Dashboard\n\n- 页面：concept {n}（共 {n}）\n")

    # ---------- 阴性对照：干净的三页必须报 0 ----------
    a = write("concepts/甲.md", "甲", evidence=[("memory:1", "证据一")],
              related=["concepts/乙.md", "sources/2026-01-01.md"])
    b = write("concepts/乙.md", "乙", evidence=[("memory:2", "证据二")],
              related=["concepts/甲.md"])
    s = write("sources/2026-01-01.md", "2026-01-01", ptype="source",
              evidence=[("memory:1", "证据一")], related=["concepts/甲.md"])
    set_index([a, b, s])
    overview(3)
    clean = lint(wiki, db_check=False)
    assert clean.ok, (
        f"干净的 Wiki 就报了 {len(clean.issues)} 项 —— 检查器在乱报，"
        f"那它报出什么都不能信：\n" +
        "\n".join(f"  [{i.check}] {i.page}: {i.detail}" for i in clean.issues))
    steps.append("阴性对照：干净时报 0")

    # ---------- 注入 3 个坏页，必须**恰好**报出 3 个 ----------
    # ① 死链：丙 链到一个不存在的页
    write("concepts/丙.md", "丙", evidence=[("memory:3", "证据三")],
          related=["concepts/根本不存在的页.md", "concepts/甲.md"])
    # ③ 别名冲突：甲 和 丁 抢同一个 alias（页面文件里也要带上）
    write("concepts/甲.md", "甲", aliases=["共同别名"],
          evidence=[("memory:1", "证据一")],
          related=["concepts/乙.md", "sources/2026-01-01.md", "concepts/丙.md",
                   "concepts/丁.md"])
    write("concepts/丁.md", "丁", aliases=["共同别名"],
          evidence=[("memory:4", "证据四")], related=["concepts/乙.md"])
    # 乙 也链上它们，免得丙/丁 变成孤儿（那会多报两项，干扰"恰好 3 个"）
    write("concepts/乙.md", "乙", evidence=[("memory:2", "证据二")],
          related=["concepts/甲.md", "concepts/丙.md", "concepts/丁.md"])
    # ② 幽灵条目：index 里登记一个**没有文件**的页
    # 索引只写一次（不能被后续 save_index 覆盖）；丁 和 overview 计数也要跟上，
    # 否则会多报未登记/计数不符，干扰"恰好 3 个"。
    WP.save_index(wiki, {"pages": {
        a: {"title": "甲", "type": "concept", "aliases": ["共同别名"],
            "sources": [], "updated": "2026-01-01"},
        b: {"title": "乙", "type": "concept", "aliases": [],
            "sources": [], "updated": "2026-01-01"},
        s: {"title": "2026-01-01", "type": "source", "aliases": [],
            "sources": [], "updated": "2026-01-01"},
        "concepts/丙.md": {"title": "丙", "type": "concept", "aliases": [],
                           "sources": [], "updated": "2026-01-01"},
        "concepts/丁.md": {"title": "丁", "type": "concept", "aliases": [],
                           "sources": [], "updated": "2026-01-01"},
        "concepts/幽灵.md": {"title": "幽灵", "type": "concept", "aliases": [],
                             "sources": [], "updated": "2026-01-01"},
    }})
    overview(6)          # 5 个真实文件 + 1 个幽灵条目

    bad = lint(wiki, db_check=False)
    checks = {i.check for i in bad.issues}
    detail = "\n".join(f"  [{i.check}] {i.page}: {i.detail}" for i in bad.issues)
    assert "死链" in checks, f"死链没报出来：\n{detail}"
    assert "幽灵条目" in checks, f"幽灵条目没报出来：\n{detail}"
    assert "别名冲突" in checks, f"别名冲突没报出来：\n{detail}"
    assert len(bad.issues) == 3, (
        f"注入了 3 个坏页，却报了 {len(bad.issues)} 项 —— "
        f"多报和少报一样坏（前者是噪音，后者是漏检）：\n{detail}")
    steps.append("注入 3 个坏页恰好报出 3 个")

    # ---------- 其余几项也要各自能报 ----------
    write("concepts/空.md", "空", summary="", evidence=[], related=["concepts/乙.md"])
    empty = lint(wiki, db_check=False)
    assert any(i.check == "空页" and "空.md" in i.page for i in empty.issues), \
        "空页没报出来"
    # 未登记（文件在、index 里没有）
    assert any(i.check == "未登记" and "空.md" in i.page for i in empty.issues), \
        "未登记没报出来"
    steps.append("空页 / 未登记 各自能报")

    # ---------- 报告要落盘 ----------
    from wiki.lint import write_report
    p = write_report(wiki, empty)
    assert p.is_file() and "lint 报告" in p.read_text(encoding="utf-8")
    assert (root / "outputs" / "lint.json").is_file(), "机器可读的结果没写"
    steps.append("报告落盘")

    _sh.rmtree(root, ignore_errors=True)
    return "；".join(steps)

def t_wiki_access():
    """Wiki 的读写边界：**对话中只读**。

    两条都要验：
    - `read_file` / `wiki_search` 能读（不用为它新造工具）
    - `write_file` 写不进去（能写的话 Wiki 就成了第二个记忆出口，与 `Memory` 表双写）

    另外：**Wiki 根挂进沙箱时不能影响会话身份** —— `conversation_workspace()`
    取的是 `workspace_roots[0]`，根列表顺序错了会让陪伴主对话被劈成两条。
    """
    import shutil as _sh
    from pathlib import Path

    import core.profile as P
    from tools import execute_tool
    from wiki import pages as WP
    from wiki.search import search

    steps = []

    # ---------- ① 会话身份没被新沙箱根影响 ----------
    assert P.COMPANION.conversation_key() == (None, None), \
        f"陪伴的会话身份被沙箱根影响了：{P.COMPANION.conversation_key()}"
    roots = [str(r) for r in P.COMPANION.workspace_roots]
    assert len(roots) >= 2 and roots[0].endswith("workspace"), \
        f"Wiki 根必须**追加在最后**，否则会话身份会跟着变：{roots}"
    assert P.COMPANION.protected_paths, "Wiki 是只读的，protected_paths 不能空"
    steps.append("会话身份不受新沙箱根影响")

    # ---------- ② 检索：能按内容找到页 ----------
    def broken_embed(_text):
        raise ConnectionError("本组用关键词路径，故意让嵌入失败")

    root = DATA_DIR / "_smoke_wiki_access"
    _sh.rmtree(root, ignore_errors=True)
    wiki = root / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    page = WP.Page(path="concepts/换工作.md", title="换工作", type="concept",
                   summary="对方在考虑换工作，主要原因是通勤太远。",
                   updated="2026-01-01",
                   evidence=[WP.Evidence("memory:1", "2026-01-01",
                                         "对方提到在考虑换工作，原因是通勤太远")])
    page.sources = ["memory:1"]
    WP.write_text(wiki / page.path, page.render())

    hits, method = search("换工作", wiki, embed=broken_embed)
    assert hits and hits[0].title == "换工作", f"没搜到：{hits}"
    assert method == "keyword", f"传了坏的 embed 就该走关键词：{method}"
    hits2, _ = search("通勤", wiki, embed=broken_embed)
    assert hits2, "正文里的词也该能搜到"
    hits3, _ = search("完全不相干的词xyz", wiki, embed=broken_embed)
    assert not hits3, f"不相干的词不该有命中：{hits3}"
    steps.append("关键词检索（含空结果）")

    # ---------- ③ 嵌入挂掉时，退化必须写在结果里 ----------
    empty_hits, m2 = search("", wiki, embed=None)
    assert m2 == "empty" and not empty_hits
    # 假 embed：让它抛异常，走退化路径
    def broken(_text):
        raise ConnectionError("模拟嵌入后端没起")
    hits4, m4 = search("换工作", wiki, embed=broken)
    assert m4 == "keyword" and hits4, \
        "嵌入挂了要走关键词，而不是返回空（静默返回空 = 模型以为知识库里没有）"
    steps.append("嵌入不可用时退化到关键词")

    # ---------- ③b 嵌入可用时走语义路径，并且**能召回没有词面重合的页** ----------
    # 「最近在读什么」与页面用词几乎不重合，关键词路径很难排到它，语义路径可以。
    try:
        from llm import embed as real_embed
        _probe = real_embed("探测嵌入后端是否可用")
        real_ok = bool(_probe)
    except Exception:
        real_ok = False
    if real_ok:
        h_sem, m_sem = search("最近在读什么", wiki, embed=real_embed)
        assert m_sem == "embedding", f"嵌入可用时应当走语义路径：{m_sem}"
        assert h_sem, "语义检索没有命中"
        steps.append("嵌入可用时走语义路径")
    else:
        steps.append("嵌入后端不可用，语义路径本次未验（退化路径已验）")

    _sh.rmtree(root, ignore_errors=True)

    # ---------- ④ 真 Wiki 的读写边界 ----------
    tok = P.activate(P.COMPANION)
    try:
        out = execute_tool("read_file", {"path": "WIKI.md", "limit": 3})
        assert out.ok, f"Wiki 的契约文件应当读得到：{out.content[:120]}"
        for path in ("wiki/index.md", "raw/days/x.md", "WIKI.md", "scripts/lint.py"):
            out = execute_tool("write_file",
                               {"path": path, "content": "x", "mode": "overwrite"})
            assert not out.ok and "受保护" in out.content, \
                f"{path} 应当不可写（§7.8 对话中 Wiki 只读）：{out.content[:120]}"
        out = execute_tool("wiki_search", {"query": "换工作"})
        assert out.ok and "知识库" in out.content, out.content[:200]
    finally:
        P.deactivate(tok)
    steps.append("Wiki 可读不可写")

    from core import readset
    readset.reset()
    return "；".join(steps)

def t_tool_search():
    from core.toolsearch import index_available, search_tools
    from tools import active_schemas
    mode = "向量" if index_available() else "关键词（嵌入后端不可用，已退化）"
    names = search_tools("帮我看看这个文件里写了什么")
    assert "read_file" in names or "list_dir" in names, names
    got = [s["function"]["name"] for s in active_schemas(names)]
    assert "read_file" in got, got
    assert "retrieve_memory" in got, got          # 核心工具仍在
    return f"[{mode}] 命中 {names} → tools 字段共 {len(got)} 个"

def t_two_stage_schemas():
    """回归：扩展工具必须真的进 tools= 字段。

    显式传 COMPANION，不依赖"当前激活的是谁"。
    """
    from core.profile import COMPANION
    from tools import active_schemas, core_schemas
    assert len(core_schemas(COMPANION)) == 6
    full = active_schemas(["read_file", "list_dir"], COMPANION)
    assert len(full) == 8, len(full)
    assert active_schemas([], COMPANION) == core_schemas(COMPANION)
    assert active_schemas(["不存在的工具"], COMPANION) == core_schemas(COMPANION)
    # 已经是常驻的不要再加一遍，否则 tools= 里会出现两个同名工具
    assert active_schemas(["retrieve_memory"], COMPANION) == core_schemas(COMPANION)
    return "核心 6 + 扩展 2 正确并入（重复的不重复加）"

def t_dependency_closure():
    """依赖闭包机制（当前 fs 包无依赖对，只验证机制本身可用）。"""
    from tools.registry import REGISTRY, ToolSpec, expand_closure
    probe = ToolSpec(name="_probe_a", description="", parameters={"type": "object"},
                     func=lambda: "", pack="probe", requires=("_probe_b",))
    probe_b = ToolSpec(name="_probe_b", description="", parameters={"type": "object"},
                       func=lambda: "", pack="probe", requires=("_probe_c",))
    probe_c = ToolSpec(name="_probe_c", description="", parameters={"type": "object"},
                       func=lambda: "", pack="probe")
    for s in (probe, probe_b, probe_c):
        REGISTRY[s.name] = s
    try:
        got = expand_closure(["_probe_a"])
        assert got == ["_probe_a", "_probe_b", "_probe_c"], got
        assert expand_closure(["read_file"]) == ["read_file"]
    finally:
        for s in (probe, probe_b, probe_c):
            REGISTRY.pop(s.name, None)
    return "A→B→C 传递闭包正确"

def t_prompt_layering():
    """核心断言：易变内容只出现在历史之后的尾部，前缀（system + 摘要 + 历史）恒定。"""
    from core.context import assemble_messages, build_notice
    from personas.persona import build_system_prompt
    base = datetime(2026, 9, 14, 16, 0)
    history = [
        {"id": 1, "role": "user", "content": "今天有点累", "ts": base},
        {"id": 2, "role": "assistant", "content": "那就早点休息呀", "ts": base},
    ]
    summary = "9-13 对方提到在考虑换工作。"
    notice_a = build_notice(ctx_tokens=1000, ctx_ratio=0.1, packs=[("fs", "文件")], ts=base)
    notice_b = build_notice(ctx_tokens=99000, ctx_ratio=0.75,
                            packs=[("fs", "文件"), ("code", "代码")],
                            ts=base + timedelta(days=3))
    a = assemble_messages(build_system_prompt(), history, "在吗",
                          summary=summary, notice=notice_a)
    b = assemble_messages(build_system_prompt(), history, "在吗",
                          summary=summary, notice=notice_b)

    # 前缀 = [system, 摘要, 历史 user, 历史 assistant] —— 两条构造必须逐条相同
    assert a[0] == b[0], "system 消息发生了变化（恒定段被污染）"
    assert a[:4] == b[:4], "前缀（system + 摘要 + 历史）不稳定"
    assert a[4] != b[4], "尾部通知应当不同（否则说明易变内容没进尾部）"

    # 摘要必须真的在 prompt 里
    assert summary in a[1]["content"], "摘要段没有注入"
    assert a[1]["role"] == "user", "摘要段应当是 user 角色（避免中途 system 的端点限制）"

    # 无摘要时不应出现空段
    no_sum = assemble_messages(build_system_prompt(), history, "在吗", notice=notice_a)
    assert len(no_sum) == len(a) - 1, "summary 为空时不应留下空消息"

    # 时间前缀稳定性：同一 ts 渲染两次必须一致
    from core.context import render_time_prefix
    assert render_time_prefix(base) == render_time_prefix(base)
    return f"前缀稳定 {len(a[:4])} 段（system+摘要+历史），分歧点在第 5 段（尾部）"

def t_history_split():
    from core.context import split_history_by_tokens
    base = datetime(2026, 9, 14, 16, 0)
    history = []
    for i in range(10):
        history.append({"id": i, "role": "user", "content": f"问题{i}" * 5, "ts": base})
        history.append({"id": i, "role": "assistant", "content": f"回答{i}" * 5, "ts": base})
    older, recent = split_history_by_tokens(history, keep_tokens=30)
    assert older and recent, (len(older), len(recent))
    assert recent[0]["role"] == "user", "保留段必须以 user 开头（问答对不能拆散）"
    assert len(older) + len(recent) == len(history)
    return f"较早 {len(older)} 条 / 保留 {len(recent)} 条"

def t_compress_guards():
    from core.compress import check_guards
    base = datetime(2026, 9, 14, 16, 0)
    history = []
    for i in range(40):
        history.append({"id": i, "role": "user", "content": "内容" * 40, "ts": base})
        history.append({"id": i, "role": "assistant", "content": "回复" * 40, "ts": base})
    ok, reason, older, recent = check_guards(history, turn_count=10,
                                             last_compress_turn=9, force=False,
                                             keep_recent_tokens=200)
    assert not ok and "冷却" in reason, reason
    ok2, reason2, _, _ = check_guards(history, turn_count=30,
                                      last_compress_turn=0, force=False,
                                      keep_recent_tokens=200)
    assert ok2, reason2
    ok3, reason3, _, _ = check_guards([], turn_count=30, last_compress_turn=0)
    assert not ok3 and "没有可压缩" in reason3, reason3
    ok4, reason4, _, _ = check_guards(history, turn_count=30, last_compress_turn=0,
                                      force=False, keep_recent_tokens=100000)
    assert not ok4 and "没有可压缩" in reason4, reason4
    return "冷却/收益/空历史三条护栏生效"


def t_forced_compress_notice():
    """强制压缩必须**告诉模型**，而且不能和自愿压缩混为一谈。

    判据分正反两面：**该说的要说**，**冷却期外不能出现**（否则每轮都提，
    模型会对一件早就过去的事持续保持警觉）。
    """
    from core.context import build_notice
    import re

    #: 判据必须锚在**句式**上：notice 里有日期（如 `2026-10-12` 含 `-1`），
    #: 不能在全串上做子串搜索。
    negative_age = re.compile(r"-\d+\s*轮前")

    def at(**kw):
        return build_notice(ctx_tokens=10_000, ctx_ratio=0.5, **kw)

    # ① 强制压缩后：必须有一条明确的、说明"不是你调用的"的通知
    n = at(forced_compress=True, last_compress_turn=8, turn_count=7)
    assert "强制压缩" in n, n
    assert "不是你调用" in n, f"没说明是谁干的，模型会以为是自己的决定：{n}"
    assert not negative_age.search(n), f"又出现负数了：{negative_age.search(n).group()}"

    # ② 自愿压缩（同一轮内）：只是复述，不能冒充强制
    v = at(forced_compress=False, last_compress_turn=8, turn_count=7)
    assert "强制" not in v, f"模型自己调的压缩被说成系统强制的：{v}"
    assert re.search(r"^- 上次压缩：\S+", v, re.M), \
        f"自愿压缩也该留一行「上次压缩：…」：{v}"

    # ③ 负数在任何组合下都不许出现
    for turn_count in (5, 6, 7, 8, 9, 10):
        line = at(forced_compress=False, last_compress_turn=8, turn_count=turn_count)
        assert not negative_age.search(line), \
            f"turn_count={turn_count} → {negative_age.search(line).group()}"

    # ④ 冷却期外：这一行**不该出现**（反例，防"每轮都提"）
    far = at(forced_compress=False, last_compress_turn=1, turn_count=30)
    assert "压缩" not in far, f"早就过去了还在提：{far}"

    # ⑤ 没压过：什么都没有
    never = at(forced_compress=False, last_compress_turn=0, turn_count=3)
    assert "压缩" not in never, never

    return "强制/自愿分得清 · 负数没了 · 冷却期外不出现"

def t_tool_call_parsing():
    from llm import _extract_text_tool_calls
    text = ('我先查一下文件。\n<tool_call>\n{"name": "read_file", '
            '"arguments": {"path": "notes/hello.md"}}\n</tool_call>')
    content, calls = _extract_text_tool_calls(text)
    assert len(calls) == 1 and calls[0].name == "read_file", calls
    assert calls[0].arguments == {"path": "notes/hello.md"}, calls[0].arguments
    assert "tool_call" not in content and "我先查一下文件" in content, content
    fp = calls[0].fingerprint()
    assert fp.startswith("read_file:")
    return "文本 tool_call 解析 + 指纹正常"

def t_e2e_chitchat():
    """端到端：闲聊不应调用工具。

    | 输入 | 期望 | 为什么 |
    |---|---|---|
    | 「下午好呀」 | **绝不能调工具** | 无任何可检索内容，调了就是浪费一步 |
    | 「在吗？今天心情有点复杂」 | 一般不调 | 带情绪，查记忆可以理解 → **容错重试** |

    「心情」那条是概率性的，实测约 0/5 调工具。
    """
    from core.loop import ask, reset_session

    # prompt 钩子必须被填上，否则 --show-prompt 会静默打印空
    def prompt_ok(result):
        prompt = result.get("prompt") or []
        assert prompt and prompt[0].get("role") == "system", \
            f"state['prompt'] 没填（--show-prompt 会静默失效）：{len(prompt)} 条"
        return len(prompt)

    # ---- ① 纯闲聊：绝不能调工具（干脆的断言，不重试）----
    reset_session()
    r1 = ask("下午好呀")
    assert r1.get("response"), "没有回复"
    assert not r1.get("trace"), \
        f"纯闲聊却调了工具 —— 这句话没有任何可检索的内容：{r1.get('trace')}"
    n_prompt = prompt_ok(r1)
    assert r1.get("ctx_tokens", 0) > 0

    # ---- ② 带情绪的问候：可以理解它去查一次，但不能次次都查 ----
    def one(i):
        reset_session()
        r = ask("在吗？今天心情有点复杂")
        assert r.get("response"), "没有回复"
        if r.get("trace"):
            return ""
        return (f"第 {i}/3 次没调工具 · 回复 {len(r['response'])} 字 · "
                f"ctx {r['ctx_tokens']} token · prompt {n_prompt} 条")

    msg, _ = _first_success(one, times=3)
    assert msg, ("3 次都调了工具。基线约 0/5，3 连调约等于没有 —— "
                 "更像是提示词或工具集变了，不是运气。")
    return f"纯闲聊 0 次调工具；带情绪问候 {msg}"

def _first_success(attempt, *, times: int):
    """重复调用 attempt(i)，返回第一个非空结果。

    专门用来包"断言模型会自己做出某个选择"的检查：那类检查是**概率性**的，
    按确定性断言就会误报。

    用它之前必须**先量出基线通过率**并写在调用处的文档里。
    目前两处：`t_e2e_tool_call` 67%（4/6）、`t_e2e_two_stage` 83%（5/6）。
    """
    for i in range(1, times + 1):
        got = attempt(i)
        if got:
            return got, i
    return None, times

def t_e2e_tool_call():
    """端到端：让模型回忆 → **必须**由模型自己调用 retrieve_memory。

    记忆只有这一条来路，所以断言是"必须"。但它**是概率性的：
    实测通过率 67%（4/6，2026-10-10）**，重试 4 次（p=0.67 时 4 连败约 1.2%）
    并要求至少成功 1 次。
    """
    from core.loop import ask, reset_session

    attempts = 4

    def one(i):
        reset_session()
        result = ask("你还记得我之前说过什么吗？帮我回忆一下")
        names = [t.get("name") for t in result.get("trace", [])]
        assert result.get("response"), "没有回复"
        if "retrieve_memory" not in names:
            return ""
        hits = [t for t in result.get("trace", [])
                if t.get("name") == "retrieve_memory"]
        return (f"第 {i}/{attempts} 次成功 · 轨迹 {names} · "
                f"记忆工具返回 {len(hits[0].get('content', ''))} 字")

    msg, _ = _first_success(one, times=attempts)
    assert msg, (f"{attempts} 次都没调 retrieve_memory。基线通过率约 67%，"
                 f"{attempts} 连败约 1.2% —— 更像是记忆链路真的断了，不是运气。")
    return msg

def t_e2e_two_stage():
    """端到端：非核心工具两阶段加载（search_tools → 直接调用 read_file）。

    光把定义放进工具结果，模型不会调用它。

    **这个检查天然是概率性的，实测通过率约 83%（5/6，2026-10-10）。**
    剩下那 ~17% 里模型不检索，直接用核心工具 `write_file` 编一份内容覆盖掉夹具。
    所以**重试 3 次、要求至少成功 1 次**：p=0.83 时 3 次全败约 0.5%。
    """
    from core.loop import ask, reset_session
    from tools import execute_tool
    from tools.sandbox import workspace

    marker = "牛奶、面包、猫罐头"
    attempts = 3

    def one(i):
        # 夹具要**自己控制初始状态**：先删掉再写，避免撞上读后写保护。
        (workspace() / "notes" / "reading.md").unlink(missing_ok=True)
        out = execute_tool("write_file", {"path": "notes/reading.md",
                                          "content": f"待办：{marker}"})
        assert out.ok, out.content

        reset_session()
        result = ask("帮我看看 notes/reading.md 里写了什么")
        names = [t.get("name") for t in result.get("trace", [])]

        # 不断言"回复里含夹具内容"：模型可能先用 write_file 覆盖夹具。
        # 这里只验证两阶段加载之后 read_file 确实被调用且返回了内容。
        reads = [t for t in result.get("trace", [])
                 if t.get("name") == "read_file" and t.get("kind") == "tool_result"]
        if "read_file" not in names or not reads or not reads[0].get("content"):
            return ""
        loaded = result.get("loaded_tools") or []
        assert "search_tools" in names, f"调到了 read_file 却没检索（轨迹 {names}）"
        return (f"第 {i}/{attempts} 次成功 · 轨迹 {names} · "
                f"已加载 {loaded} · 读到 {len(reads[0]['content'])} 字")

    msg, _ = _first_success(one, times=attempts)
    assert msg, (f"{attempts} 次都没走通两阶段加载。基线通过率约 83%，"
                 f"{attempts} 连败约 0.5% —— 更像是真的坏了，不是运气。")
    return msg

def t_e2e_forced_compress():
    """端到端：构造超预算历史，检验压缩链路（含 consolidate 固化）。"""
    from core.compress import compress
    from core.context import counter
    from store.db import (append_message, get_context_state, get_or_create_conversation,
                    load_history)
    from core.loop import get_session

    cid = get_or_create_conversation()
    base = datetime.now() - timedelta(days=2)
    filler = "我最近在考虑换工作，也在想要不要搬去别的城市住一段时间。"
    for i in range(40):
        append_message(cid, "owner", f"第{i}轮：{filler}{filler}", ts=base)
        append_message(cid, "assistant", f"第{i}轮：我记住啦，你要注意休息呀。{filler}", ts=base)

    state = get_context_state(cid)
    history = load_history(cid, after_id=state["summary_upto_id"])
    total = sum(counter.count(m["content"]) for m in history)
    assert total > settings.COMPRESS_KEEP_TOKENS, f"构造的历史太小（{total} token）"

    result = compress(cid, history, turn_count=99, keep_recent_tokens=500,
                      force=True, verbose=False)
    assert result.ok, result.reason
    assert result.removed_count > 0 and result.summary, result
    assert len(result.keep) < len(history), "压缩后历史没有变短"
    after = get_context_state(cid)
    assert after["summary_upto_id"] > state["summary_upto_id"], "水位线没有前移"
    return (f"压掉 {result.removed_count} 条 / 省 {result.removed_tokens} token / "
            f"固化 {result.consolidated.get('created', 0)} 条记忆 / "
            f"摘要 {len(result.summary)} 字")

def t_summary_survives_compression():
    """行为测试：压缩之后，模型应当仍然**看得见**被压缩的内容。

    必须是行为测试：只断言数据库状态（摘要非空、水位线前移）会在模型什么都
    没看到的情况下通过。这里问一句"我养了什么宠物"，答案只可能来自摘要。
    """
    from core.compress import compress
    from store.db import (append_message, get_context_state, get_or_create_conversation,
                    load_history)
    from core.loop import ask, reset_session

    marker_pet = "小橘"
    cid = get_or_create_conversation()
    base = datetime.now() - timedelta(days=3)
    filler = "今天聊了些有的没的，天气、吃饭、还有那部看了一半的剧。"
    # 先铺垫足够多的历史，再把关键事实夹在中间
    for i in range(30):
        append_message(cid, "owner", f"第{i}轮：{filler}{filler}", ts=base)
        append_message(cid, "assistant", f"第{i}轮：嗯嗯，我听着呢。{filler}", ts=base)
    append_message(cid, "owner", f"对了，我养了一只叫{marker_pet}的橘猫。", ts=base)
    append_message(cid, "assistant", "橘猫！记下了。", ts=base)
    for i in range(30, 40):
        append_message(cid, "owner", f"第{i}轮：{filler}{filler}", ts=base)
        append_message(cid, "assistant", f"第{i}轮：嗯嗯，我听着呢。{filler}", ts=base)

    state = get_context_state(cid)
    history = load_history(cid, after_id=state["summary_upto_id"])
    result = compress(cid, history, turn_count=99, keep_recent_tokens=300,
                      force=True, verbose=False)
    assert result.ok, result.reason
    # 关键：那只猫必须已经**不在**保留的历史里（否则这个测试没有意义）
    kept_text = " ".join(m["content"] for m in result.keep)
    assert marker_pet not in kept_text, "猫还在保留窗口里，测试构造无效"

    reset_session()                      # 模拟重启：只剩摘要 + 历史
    answer = ask("我养了什么宠物来着？")["response"]
    assert marker_pet in answer or "猫" in answer, \
        f"压缩后模型看不到摘要（回复里没有猫）：{answer[:120]}"
    return f"压缩 {result.removed_count} 条后，模型仍能答出「{marker_pet}」"

def t_compress_invalidates_readset():
    """压缩必须让「读过」的记录失效，否则守卫会替「盲改」背书。

    `readset` 的判据是"这个会话里读过"，想保证的是"模型**此刻看得见**这个文件"。
    压缩把消息移出上下文之后两者分开了，而 `edit_file` 的守卫还会放行。

    判据分正反两面：**该失效的要失效**（压缩后必须重读）、**不该失效的别误伤**、
    **拒绝理由要说对原因**（说"还没被读过"是错的 —— 它读过）。
    """
    from core import readset
    from core.compress import compress
    from core.context import counter
    from store.db import (append_message, get_context_state,
                          get_or_create_conversation, load_history)

    target = DATA_DIR / "_readset_target.py"
    target.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    readset.reset()

    try:
        # ---- 反面（防误伤）：没压缩过，读过就该能改 ----
        readset.mark_read(target)
        assert readset.stale_reason(target) is None, "没压缩过就拒了 —— 误伤"

        # ---- 构造一段够压的历史 ----
        cid = get_or_create_conversation()
        base = datetime.now() - timedelta(days=2)
        filler = "我最近在考虑换工作，也在想要不要搬去别的城市住一段时间。"
        for i in range(40):
            append_message(cid, "owner", f"第{i}轮：{filler}{filler}", ts=base)
            append_message(cid, "assistant", f"第{i}轮：我记住啦，你要注意休息呀。{filler}",
                           ts=base)
        state = get_context_state(cid)
        history = load_history(cid, after_id=state["summary_upto_id"])
        assert sum(counter.count(m["content"]) for m in history) > \
            settings.COMPRESS_KEEP_TOKENS, "构造的历史太小"
        assert readset.has_read(target), "压缩前读过的记录就该是有效的"

        result = compress(cid, history, turn_count=99, keep_recent_tokens=500,
                          force=True, verbose=False)
        assert result.ok and result.removed_count > 0, result.reason

        # ---- 正面：压缩之后必须失效，而且理由要说得对 ----
        assert not readset.has_read(target), \
            "压缩移走了消息，可 readset 还说「读过」—— 守卫会替盲改背书"
        reason = readset.stale_reason(target) or ""
        assert "压缩" in reason, \
            f"拒绝理由必须点明是压缩干的（说「还没被读过」是错的）：{reason}"

        # ---- 重读之后应当恢复 ----
        readset.mark_read(target)
        assert readset.stale_reason(target) is None, "重读之后还不让改"

        # ---- reset 必须把「作废记录」也清掉 ----
        readset.reset()
        after_reset = readset.stale_reason(target) or ""
        assert "压缩" not in after_reset, (
            "reset 只清了 _seen、忘了清 _forgotten —— 新会话里这个文件会被说成"
            f"「上次压缩移走了」，而它压根没被读过：{after_reset}")
        assert "还没被读过" in after_reset, after_reset

        return (f"压掉 {result.removed_count} 条后作废 · 理由说得对 · "
                f"重读恢复 · 不误伤 · reset 清干净")
    finally:
        target.unlink(missing_ok=True)
        readset.reset()


def t_offline_compress():
    """端到端：启动是**纯读取**，压缩发生在**退出**时。

    验证三件事：
      ① preload 不跑任何 LLM（首句应当秒回，且历史仍是长的）
      ② 退出流程（offline.run）会把历史压到预算内，水位线前移
      ③ 再次启动装载的历史变短了
    """
    from core.context import counter
    from store.db import (append_message, get_context_state, get_or_create_conversation,
                    load_history)
    from core.loop import ask, get_session, reset_session
    from offline.run import run as offline_run

    cid = get_or_create_conversation()
    base = datetime.now() - timedelta(days=1)
    filler = "明天想去超市买点东西，顺便把上次借的书还了，还要给猫买罐头。"
    for i in range(140):
        append_message(cid, "owner", f"第{i}轮：{filler}{filler}", ts=base)
        append_message(cid, "assistant", f"第{i}轮：好呀，我帮你记着。{filler}", ts=base)

    state = get_context_state(cid)
    raw = load_history(cid, after_id=state["summary_upto_id"])
    raw_tokens = sum(counter.count(m["content"]) for m in raw)
    budget = settings.OFFLINE_KEEP_TOKENS
    # 必须超过"闸门"（预算 + 最小收益），否则 offline 会按设计跳过压缩
    gate = budget + settings.COMPRESS_MIN_GAIN_TOKENS
    assert raw_tokens > gate, f"前置历史不够大（{raw_tokens} vs 闸门 {gate}）"

    # ① 启动：应当立刻可用，且历史**没有被压缩**（压缩在退出时做）
    reset_session()
    t0 = time.time()
    result = ask("我回来啦")
    first = time.time() - t0
    assert result.get("response"), "启动后没有回复"
    loaded_before = sum(counter.count(m["content"]) + 4 for m in get_session().history)
    assert loaded_before > budget, "启动阶段不该压缩（离线工作已移到退出时）"
    assert first < 90, f"首句耗时 {first:.0f}s，启动路径不该有 LLM 调用"

    # ② 退出：跑离线流程，历史应被压到预算内
    stats = offline_run(verbose=False)
    c = stats["compress"]
    # 必须先查 error：只断言 .get("skipped") 的话 {"error": ...} 会得到 None，
    # `not None` 为真，异常被当成通过。
    assert "error" not in c, f"退出压缩报错：{c['error']}"
    assert not c.get("skipped"), f"退出时没压缩：{c}"
    after = get_context_state(cid)
    assert after["summary_upto_id"] > state["summary_upto_id"], "水位线没有前移"

    # ③ 再启动：装载的历史变短
    reset_session()
    ask("嗯嗯")
    loaded_after = sum(counter.count(m["content"]) + 4 for m in get_session().history)
    assert loaded_after < loaded_before * 0.7, \
        f"再次启动装载 {loaded_after} token，没有变短（之前 {loaded_before}）"
    return (f"启动首句 {first:.0f}s / 装载 {loaded_before:,} token（未压缩）"
            f" → 退出压缩省 {stats['compress'].get('saved', 0):,} token"
            f" → 再启动 {loaded_after:,} token")

# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="跳过需要模型 API 的检查")
    parser.add_argument("--fresh", action="store_true",
                        help="端到端前清空当前会话（让测试可复现，不动长期记忆）")
    args = parser.parse_args()

    log(f"=== agent v19 smoke test · {datetime.now():%Y-%m-%d %H:%M:%S} ===")
    log(f"模型 {settings.CHAT_MODEL} @ {settings.CHAT_BASE_URL}")
    log(f"窗口 {settings.CTX_WINDOW:,} · 预警 {settings.CTX_WARN_RATIO:.0%} · "
        f"强制 {settings.CTX_FORCE_RATIO:.0%}")
    log("")

    log("--- 静态检查 ---")
    check("工具注册表与 schema（按 Profile）", t_registry)
    check("Profile 机制（工具集/沙箱/会话归属隔离）", t_profile)
    check("会话身份（按 source+工作区分流）", t_conversation_identity)
    check("参数校验与错误回灌", t_param_validation)
    check("文件沙箱", t_sandbox)
    check("文件工具（行号/分页/编辑九种失败模式/行尾保真）", t_file_tools)
    check("代码检索（glob/grep/符号 + 忽略规则）", t_code_tools)
    check("执行工具（挂死拦截/审批/杀进程树/截断）", t_exec_tools)
    check("任务编排（清单校验/渲染/不收敛护栏）", t_plan_tools)
    check("LLM Wiki（幂等/重建/归并/锚点）", t_wiki)
    check("Wiki lint（九项 + 注入三坏页必须报出三）", t_wiki_lint)
    check("Wiki 读写边界（可读不可写 / 检索退化）", t_wiki_access)
    check("工具检索（search_tools）", t_tool_search)
    check("扩展工具并入 tools= 字段", t_two_stage_schemas)
    check("依赖闭包机制", t_dependency_closure)
    check("prompt 分层（前缀稳定）", t_prompt_layering)
    check("历史按 token 裁剪并对齐问答对", t_history_split)
    check("压缩护栏", t_compress_guards)
    check("强制压缩通知（说清是谁干的 + 无负数）", t_forced_compress_notice)
    check("文本 tool_call 解析", t_tool_call_parsing)

    if args.offline:
        log("\n（--offline：跳过端到端检查）")
    else:
        log("\n--- 端到端检查（需要模型 API）---")
        if args.fresh:
            from store.db import clear_conversation, get_or_create_conversation
            from core.loop import reset_session
            clear_conversation(get_or_create_conversation())
            reset_session()
            log("（--fresh：已清空当前会话，长期记忆库保留）")
        check("闲聊不调用工具", t_e2e_chitchat)
        check("模型自主调用记忆工具", t_e2e_tool_call)
        check("模型自主检索扩展工具（search_tools → read_file）", t_e2e_two_stage)
        check("压缩链路（固化 + 摘要 + 水位线）", t_e2e_forced_compress)
        check("D1 摘要真的进了 prompt（行为）", t_summary_survives_compression)
        check("压缩让读后写保护失效（守卫不替盲改背书）",
              t_compress_invalidates_readset)
        check("C12 启动纯读取 + 退出压缩", t_offline_compress)

    log("")
    log(f"=== 通过 {_PASS} / 失败 {_FAIL} ===")

    report = DATA_DIR / "smoke_report.txt"
    report.write_text("\n".join(_LINES), encoding="utf-8")
    print(f"报告已写入 {report}")

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", write_through=True, line_buffering=True)
    main()
