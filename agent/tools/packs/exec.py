"""执行工具包（exec）：跑命令、跑 git。

必然挂死的命令（`vim` / `gdb` / `less` / `tail -f` / 裸解释器）直接拒绝，不走审批；
危险的（`rm -rf` / `git push --force`）走审批。

输出处理：**头 60% + 尾 40%**（错误信息、traceback、失败摘要全在尾部）；stdout /
stderr 分开截断再合并；退出码明确给出；子进程强制 UTF-8，替换字符明确报告。

超时必须杀掉整棵进程树，只杀父进程的话子进程会活下来占着端口。
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path

from config import get_settings
from tools.registry import ToolOutcome, tool
from tools.sandbox import relative, resolve

settings = get_settings()

# 单条命令的输出上限（内存护栏，不是显示截断），否则 `yes` 之类的命令能吃光内存。
OUTPUT_CAP_BYTES = 5 * 1024 * 1024
# 回灌给模型的字符上限
DISPLAY_CHARS = 6000
HEAD_RATIO = 0.6        # 头 60% / 尾 40%
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 900


# ---------------------------------------------------------------- 必然挂死（直接拒）

#: (正则, 说明)。命中直接拒，不过审批 —— 它们不是"危险"，是"回不来"。
BLOCKED: list[tuple[str, str]] = [
    (r"(?i)(^|[|&;]\s*)(vim|vi|nvim|emacs|nano|pico|joe|mcedit)\b",
     "交互式编辑器会让进程挂住，永远等不到结束"),
    (r"(?i)(^|[|&;]\s*)(less|more|man|info)\b",
     "分页器要按键才继续，会一直等"),
    (r"(?i)(^|[|&;]\s*)(gdb|lldb|pdb|ipdb|pudb)\b",
     "交互式调试器会进自己的提示符"),
    (r"(?i)(^|[|&;]\s*)(htop|top|atop|watch)\b",
     "全屏交互程序会一直刷屏不退出"),
    (r"(?i)tail\s+(-[a-zA-Z]*f|--follow)|Get-Content\b[^|]*-Wait",
     "会一直等着新内容，永远不结束"),
    (r"(?i)^\s*(python3?|py|node|irb|sqlite3|psql|mysql|pwsh|powershell|bash|sh|zsh|cmd)\s*$",
     "裸解释器会进 REPL，之后再也没有输出也永远不会结束"),
    (r"(?i)\b(nohup|disown)\b|&\s*$",
     "把进程丢到后台会脱离管控，我们再也拿不到它的输出和退出码"),
    (r"(?i)\bpython3?\s+-m\s+venv\b",
     "创建虚拟环境很慢，而且会生成一大堆文件"),
]

# ---------------------------------------------------------------- 危险（走审批）

#: (正则, 类别, 说明)。命中即需要审批，**不阻断**。
DANGEROUS: list[tuple[str, str, str]] = [
    (r"(?i)\brm\s+(-[a-zA-Z]*[rR][a-zA-Z]*[fF]?|-[a-zA-Z]*[fF][a-zA-Z]*[rR]?)\b",
     "rm-recursive", "递归/强制删除"),
    (r"(?i)\b(Remove-Item|ri)\b[^|;]*\s-(Recurse|Force|r|f)\b",
     "ps-remove", "PowerShell 递归/强制删除"),
    (r"(?i)\b(del|erase)\s+/[sq]|\brd\s+/s|\brmdir\s+/s",
     "win-del", "Windows 递归删除"),
    (r"(?i)\bformat\s+[a-z]:|\bmkfs\b|\bdd\s+if=",
     "disk", "格式化 / 写裸设备"),
    (r"(?i)\bgit\s+push\b[^|;]*(--force|-f)\b",
     "git-force-push", "强制推送会覆盖远端历史"),
    (r"(?i)\bgit\s+(reset\s+--hard|clean\s+-[a-zA-Z]*f)",
     "git-destroy", "会丢弃未提交的改动"),
    (r"(?i)\bgit\s+(checkout\s+--\s|restore\s+--source)",
     "git-discard", "会丢弃工作区改动"),
    (r"(?i)\b(shutdown|reboot|halt|Restart-Computer|Stop-Computer)\b",
     "power", "关机 / 重启"),
    (r"(?i)\b(curl|wget|iwr|Invoke-WebRequest)\b[^|;]*\|\s*(ba|z|da)?sh\b",
     "pipe-to-shell", "把网上下载的东西直接交给 shell 执行"),
    (r"(?i)\b(Stop-Process|taskkill)\b[^|;]*(-Force|/F)\b",
     "kill-force", "强制杀进程"),
    (r"(?i)>\s*(/dev/(sd|nvme)|[a-z]:\\(Windows|Program Files))",
     "overwrite-system", "往设备或系统目录写"),
    (r"(?i)\bchmod\s+-R\s+777\b|\bchown\s+-R\b",
     "perms", "递归改权限"),
    (r"(?i)\bgit\s+(commit|merge|rebase|cherry-pick|stash\s+(pop|drop)|tag\s+-d)\b",
     "git-write", "会改动仓库状态"),
]

# ---------------------------------------------------------------- 只读白名单（ALL_WRITES 档用）

READONLY_PREFIXES = (
    "ls", "dir", "cat", "type", "head", "wc", "pwd", "echo", "find", "rg", "grep",
    "git status", "git diff", "git log", "git show", "git branch", "git remote",
    "git rev-parse", "git blame", "python --version", "python3 --version",
    "which", "where", "whoami", "date", "node --version", "npm ls",
)


# ---------------------------------------------------------------- 底层执行

def _shell_argv() -> list[str]:
    """选一个 shell：优先 pwsh，退到 powershell，再退到 cmd；UTF-8 处理差别大，不能假设有哪个。"""
    if settings.SHELL_COMMAND:
        return shlex.split(settings.SHELL_COMMAND)
    if os.name == "nt":
        from shutil import which
        if which("pwsh"):
            return ["pwsh", "-NoProfile", "-NonInteractive", "-Command"]
        if which("powershell"):
            return ["powershell", "-NoProfile", "-NonInteractive", "-Command"]
        return ["cmd", "/c"]
    return ["/bin/bash", "-lc"]


def _needs_utf8_prologue(argv: list[str]) -> bool:
    """**恒为 False** —— 不做编码前导：给 PowerShell 命令拼编码设置会让以引号开头的命令
    被当成表达式报语法错误。编码交给 `_decode`。
    """
    return False


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"     # 子进程里的 python 也走 UTF-8
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _kill_tree(proc: subprocess.Popen, killer: "_TreeKiller | None" = None) -> None:
    """杀掉整棵进程树（只杀父进程的话，子进程会活下来占着端口）。
    ① Windows Job Object（`TerminateJobObject`），最可靠且不依赖外部命令 —— 受限环境
    里 `taskkill /F /T` 会 `Access denied`；② POSIX 进程组 `killpg`；③ 兜底 `proc.kill()`。
    """
    if proc.poll() is not None:
        return
    if killer is not None and killer.alive:
        killer.kill()
        return
    if os.name != "nt":
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (OSError, ProcessLookupError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


class _TreeKiller:
    """Windows Job Object 的薄包装：把子进程放进 job，杀 job = 杀整棵树。
    `KILL_ON_JOB_CLOSE` 意味着关句柄也会带走整棵树，所以 `close()` 必须在进程结束后调。
    """

    def __init__(self) -> None:
        self.handle: int | None = None
        self.alive = False
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes

            class _IO(ctypes.Structure):
                _fields_ = [("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                            ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong),
                            ("e", ctypes.c_ulonglong), ("f", ctypes.c_ulonglong)]

            class _Basic(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                            ("LimitFlags", wintypes.DWORD),
                            ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t),
                            ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_void_p),
                            ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class _Ext(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", _Basic),
                            ("IoInfo", _IO),
                            ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]

            k = ctypes.WinDLL("kernel32", use_last_error=True)
            h = k.CreateJobObjectW(None, None)
            if not h:
                return
            info = _Ext()
            info.BasicLimitInformation.LimitFlags = 0x2000   # KILL_ON_JOB_CLOSE
            if not k.SetInformationJobObject(h, 9, ctypes.byref(info),
                                             ctypes.sizeof(info)):
                k.CloseHandle(h)
                return
            self._k, self._ctypes, self.handle = k, ctypes, h
            self.alive = True
        except Exception:
            self.handle, self.alive = None, False

    def attach(self, proc: subprocess.Popen) -> None:
        if not self.alive:
            return
        try:
            if not self._k.AssignProcessToJobObject(self.handle, int(proc._handle)):
                # 进程可能已经在别的 job 里且不允许嵌套 —— 退回旧路径
                self.alive = False
        except Exception:
            self.alive = False

    def kill(self) -> None:
        if self.alive:
            try:
                self._k.TerminateJobObject(self.handle, 1)
            except Exception:
                pass

    def close(self) -> None:
        if self.handle:
            try:
                self._k.CloseHandle(self.handle)
            except Exception:
                pass
            self.handle, self.alive = None, False



def shell_hint() -> str:
    """一句话说清"你在写哪种 shell"，进尾部通知。
    PowerShell 5.1 不支持 `&&` / `||`；工具描述是恒定前缀、不能随环境变，只能放尾部通知。
    """
    argv = _shell_argv()
    name = Path(argv[0]).stem.lower() if argv else "shell"
    if name in ("powershell",):
        return ("Windows PowerShell 5.1 —— **不是 bash**："
                "不支持 `&&` / `||`（用 `;` 分隔），"
                "命令名带空格要用 `& \"路径\"`")
    if name == "pwsh":
        return ("PowerShell 7 —— 不是 bash："
                "`&&` / `||` 可用，但管道/重定向/变量语法是 PowerShell 的")
    if name == "cmd":
        return "Windows cmd —— 不是 bash：用 `&&` 可以，但没有 `$(...)`"
    if name in ("bash", "sh", "zsh"):
        return f"{name} —— 标准 POSIX shell"
    return f"{argv[0] if argv else 'shell'}"


def _is_powershell(argv: list[str]) -> bool:
    return bool(argv) and Path(argv[0]).stem.lower() in ("pwsh", "powershell")


def _fix_powershell(argv: list[str], command: str) -> str:
    """两处 PowerShell 特有的修补。
    ① 语句开头是引号串 → 补调用运算符 `&`，否则后面的 `-c` 会被当成语法错误。
    ② 末尾补退出码收尾（见 `_PS_TAIL`）：`$LASTEXITCODE` 只反映原生命令，cmdlet 失败
       时它是空的，`exit $null` = 退出 0，会把失败报成成功。
    """
    if not _is_powershell(argv):
        return command
    fixed = command
    stripped = fixed.lstrip()
    if stripped and stripped[0] in "\"'" and not stripped.startswith("&"):
        fixed = "& " + fixed
    fixed = fixed.rstrip().rstrip(";")
    if fixed and "$__ok" not in fixed:
        fixed = fixed + _PS_TAIL
    return fixed


#: PowerShell 命令的收尾：先看原生命令退出码，再看 cmdlet 成败。变量用 `$__` 前缀避免撞名。
_PS_TAIL = ("; $__ok = $?; $__code = $LASTEXITCODE; "
            "if ($__code -ne $null -and $__code -ne 0) { exit $__code } "
            "elseif (-not $__ok) { exit 1 } else { exit 0 }")


def _drain(stream, buf: bytearray, cap: int) -> None:
    """把管道读干，只留前 cap 字节。必须一直读到 EOF，提前停会让子进程阻塞在写管道上。"""
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            if len(buf) < cap:
                buf.extend(chunk[:cap - len(buf)])
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _decode(raw: bytes) -> tuple[str, str]:
    """bytes → 文本。返回 (文本, 编码告警)，不做静默乱码：UTF-8 干净就直接用；有替换字符
    则试 GBK，干净就用 GBK 并说明换了编码；都不行就用替换结果，并说明这些位置不可信。
    """
    text = raw.decode("utf-8", errors="replace")
    bad = text.count("\ufffd")
    if not bad:
        return text, ""
    try:
        gbk = raw.decode("gbk")
    except UnicodeDecodeError:
        gbk = ""
    if gbk and "\ufffd" not in gbk:
        return gbk, ("\n（注：这段输出不是 UTF-8，已按 GBK 解码 —— "
                     "内容应当是可读的，但如果你看到的和我说的对不上，以这个提醒为准。）")
    return text, (f"\n⚠️ 输出里有 {bad} 个字节既不是合法 UTF-8 也不是合法 GBK，"
                  f"已用 U+FFFD 占位。**这些位置的内容不可信**，不要照着推理。")



def _truncate(text: str) -> str:
    """头 60% + 尾 40%。**按 token 算，和外层 `mask_tool_result` 同一把尺子。**
    只截头是错的 —— 错误信息、traceback、失败摘要全在尾部。外层也是 token 口径
    （头 60% / 尾 30%），这里先截到界内，外层就不会把尾部留量改成 30%。
    """
    from core.context import counter
    limit = settings.TOOL_RESULT_MAX_TOKENS
    total = counter.count(text)
    if total <= limit:
        return text
    keep = max(int(len(text) * limit / max(total, 1)), 400)
    head, tail = int(keep * HEAD_RATIO), keep - int(keep * HEAD_RATIO)
    omitted = len(text) - head - tail
    return (text[:head]
            + f"\n\n……（中间省略 {omitted:,} 字符）……\n\n"
            + text[-tail:])


def _run(argv: list[str], command: str, *, cwd: Path, timeout: int,
         env: dict[str, str]) -> dict:
    """跑一条命令，返回原始结果。不判断、不审批 —— 那是调用方的事。"""
    full = list(argv)
    command = _fix_powershell(argv, command)
    if _needs_utf8_prologue(argv):
        command = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
                   "$OutputEncoding=[Text.Encoding]::UTF8;" + command)
    full.append(command)

    started = time.time()
    killer = _TreeKiller()          # Windows 上先建 job，attach 之后杀它=杀整棵树
    try:
        proc = subprocess.Popen(
            full, cwd=str(cwd), env=env,
            stdin=subprocess.DEVNULL,           # 禁止交互：需要输入的命令直接失败
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=(os.name != "nt"),
        )
    except (OSError, ValueError) as exc:
        killer.close()
        return {"error": f"启动失败：{type(exc).__name__}: {exc}"}
    killer.attach(proc)

    out_buf, err_buf = bytearray(), bytearray()
    threads = [threading.Thread(target=_drain, args=(proc.stdout, out_buf, OUTPUT_CAP_BYTES),
                                daemon=True),
               threading.Thread(target=_drain, args=(proc.stderr, err_buf, OUTPUT_CAP_BYTES),
                                daemon=True)]
    for t in threads:
        t.start()

    timed_out = False
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc, killer)
        try:
            code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            code = None
    for t in threads:
        t.join(timeout=5)
    killer.close()                  # 必须在进程结束之后 —— KILL_ON_JOB_CLOSE 会带走整棵树

    return {"code": code, "out": bytes(out_buf), "err": bytes(err_buf),
            "timed_out": timed_out, "elapsed": time.time() - started,
            "capped": len(out_buf) >= OUTPUT_CAP_BYTES or len(err_buf) >= OUTPUT_CAP_BYTES}


# ---------------------------------------------------------------- 判定

def _blocked(command: str) -> str | None:
    for pat, why in BLOCKED:
        if re.search(pat, command):
            return why
    return None


def _danger(command: str) -> tuple[str, str] | None:
    for pat, key, why in DANGEROUS:
        if re.search(pat, command):
            return key, why
    return None


def _is_readonly(command: str) -> bool:
    stripped = command.strip().lower()
    if any(ch in stripped for ch in ("|", ">", ";", "&&", "||", "$(", "`")):
        return False            # 有管道/重定向/串联就不是"只读"了
    return stripped.startswith(READONLY_PREFIXES)


def _risk_of(command: str) -> tuple[str, str, str]:
    """返回 (risk, key, reason)。工具自己判定风险 —— 它最清楚自己要干什么。"""
    hit = _danger(command)
    if hit:
        key, why = hit
        return "destructive", key, why
    if _is_readonly(command):
        return "safe", "", ""
    return "write", "shell", "执行 shell 命令"


# ---------------------------------------------------------------- run_shell

@tool(
    name="run_shell",
    description=("在工作目录里执行一条 shell 命令，返回输出和退出码。"
                 "跑测试/构建/git 之类都走这个。"
                 "**不能交互**：需要输入的命令会直接失败，别调 vim/gdb/less 这类程序。"),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的命令（单行）"},
            "cwd": {"type": "string", "description": "在哪个子目录下执行，默认工作区根"},
            "timeout_s": {"type": "integer",
                          "description": f"超时秒数，默认 {DEFAULT_TIMEOUT}，上限 {MAX_TIMEOUT}",
                          "default": DEFAULT_TIMEOUT},
        },
        "required": ["command"],
    },
    pack="exec",
)
def run_shell(command: str, cwd: str | None = None, timeout_s: int = DEFAULT_TIMEOUT
              ) -> ToolOutcome:
    from core import approval
    command = (command or "").strip()
    if not command:
        return ToolOutcome(content="错误：命令为空。", ok=False)

    # ---- ① 必然挂死：直接拒，不问 ----
    why = _blocked(command)
    if why:
        return ToolOutcome(
            content=(f"错误：这条命令不能执行 —— {why}。\n"
                     f"命令：{command}\n"
                     f"这里没有人和你交互：进程挂住之后整轮就卡死了。\n"
                     f"换一条不需要交互的写法（比如非交互模式的参数、把输入用管道喂进去）。"),
            ok=False)

    # ---- ② 危险：问一句 ----
    risk, key, reason = _risk_of(command)
    policy = _policy()
    verdict = approval.require(f"执行命令：{command}", risk=approval.Risk(risk),
                               tool="run_shell", key=key, detail=command,
                               reason=reason, policy=policy)
    if not verdict.allowed:
        return ToolOutcome(content=f"错误：{verdict.reason}", ok=False)

    # ---- ③ 执行 ----
    workdir = resolve(cwd) if cwd else resolve(".")
    if not workdir.is_dir():
        return ToolOutcome(content=f"错误：{relative(workdir)} 不是目录。", ok=False)
    timeout = max(1, min(int(timeout_s), MAX_TIMEOUT))

    try:
        result = _run(_shell_argv(), command, cwd=workdir, timeout=timeout,
                      env=_child_env())
    except Exception as exc:
        return ToolOutcome(content=f"错误：执行失败：{type(exc).__name__}: {exc}",
                           ok=False)
    if "error" in result:
        return ToolOutcome(content=f"错误：{result['error']}", ok=False)

    out_text, out_warn = _decode(result["out"])
    err_text, err_warn = _decode(result["err"])
    # stdout / stderr **分别截断再合并** —— 否则 stderr 会被 stdout 挤掉
    parts = []
    if out_text.strip():
        parts.append(f"--- stdout ---\n{_truncate(out_text.rstrip())}")
    if err_text.strip():
        parts.append(f"--- stderr ---\n{_truncate(err_text.rstrip())}")
    body = "\n".join(parts) or "（没有任何输出）"

    code = result["code"]
    if result["timed_out"]:
        head = (f"⏱️ 超时：命令跑了 {timeout} 秒仍未结束，**已杀掉整棵进程树**"
                f"（退出码 {code}）。")
    else:
        verdict_txt = "成功" if code == 0 else "失败"
        head = f"退出码 {code}（{verdict_txt}）· 耗时 {result['elapsed']:.1f}s"

    extra = []
    if result["capped"]:
        extra.append(f"⚠️ 输出超过 {OUTPUT_CAP_BYTES // 1024 // 1024}MB，**已被截断**，"
                     f"后面的内容没收到")
    if out_warn or err_warn:
        extra.append((out_warn + err_warn).strip())

    text = f"{head}\n\n{body}"
    if extra:
        text += "\n\n" + "\n".join(extra)
    return ToolOutcome(content=text, ok=(code == 0 and not result["timed_out"]))


def counts_as_progress(command: str) -> bool:
    """这条命令算不算"有进展"？给不收敛护栏（`loop._made_progress`）用。
    只读命令不算，否则模型可以靠反复 `ls` / `git status` 让护栏永不触发。
    """
    return _risk_of(command)[0] != "safe"


def _policy():
    """当前 Profile 的审批档。取不到就用默认。"""
    from core import approval
    try:
        from core.profile import active
        return approval.ApprovalPolicy(active().approval)
    except Exception:
        return approval.ApprovalPolicy.DESTRUCTIVE


# ---------------------------------------------------------------- git

GIT_READONLY = {"status", "diff", "log", "show", "branch", "remote", "rev-parse",
                "blame", "describe", "tag", "stash", "config", "ls-files",
                "shortlog", "whatchanged", "cat-file"}
GIT_WRITE = {"commit", "push", "pull", "merge", "rebase", "reset", "checkout",
             "switch", "restore", "clean", "cherry-pick", "revert", "stash",
             "tag", "init", "clone", "fetch", "add", "rm", "mv", "apply", "am"}


@tool(
    name="git",
    description=("执行 git 命令。**只读子命令**（status/diff/log/show/blame…）直接跑；"
                 "**会改状态的**（commit/reset/checkout/clean…）需要用户确认。"
                 "参数按数组给，不要拼成一整条字符串。"),
    parameters={
        "type": "object",
        "properties": {
            "subcommand": {"type": "string",
                           "description": "git 子命令，如 status / diff / log / commit"},
            "args": {"type": "array", "items": {"type": "string"},
                     "description": "子命令的参数，如 [\"--stat\"] 或 [\"-m\", \"提交说明\"]"},
            "cwd": {"type": "string", "description": "在哪个子目录下执行，默认工作区根"},
        },
        "required": ["subcommand"],
    },
    pack="vcs",
)
def git(subcommand: str, args: list[str] | None = None,
        cwd: str | None = None) -> ToolOutcome:
    from core import approval
    sub = (subcommand or "").strip().lower()
    if not sub or not re.fullmatch(r"[a-z][a-z0-9-]*", sub):
        return ToolOutcome(content=f"错误：子命令 `{subcommand}` 看着不像 git 子命令。"
                                   f"参数按数组给，不要拼成一整条字符串。", ok=False)
    argv = [str(a) for a in (args or [])]

    readonly = sub in GIT_READONLY and not _git_args_write(sub, argv)
    risk = "safe" if readonly else "destructive"
    why = "" if readonly else f"git {sub} 会改动仓库状态"
    verdict = approval.require(f"git {sub} {' '.join(argv)}".strip(),
                               risk=approval.Risk(risk), tool="git", key=f"git:{sub}",
                               detail=" ".join(["git", sub, *argv]), reason=why,
                               policy=_policy())
    if not verdict.allowed:
        return ToolOutcome(content=f"错误：{verdict.reason}", ok=False)

    workdir = resolve(cwd) if cwd else resolve(".")
    # 不用 shell：参数已经是一个数组了，再拼字符串就要处理引号/转义/注入。
    try:
        proc = subprocess.run(["git", sub, *argv], cwd=str(workdir),
                              stdin=subprocess.DEVNULL,
                              capture_output=True, timeout=DEFAULT_TIMEOUT,
                              env=_child_env())
    except FileNotFoundError:
        return ToolOutcome(content="错误：找不到 git 可执行文件。", ok=False)
    except subprocess.TimeoutExpired:
        return ToolOutcome(content=f"错误：git {sub} 超过 {DEFAULT_TIMEOUT} 秒没结束。",
                           ok=False)
    except OSError as exc:
        return ToolOutcome(content=f"错误：执行失败：{exc}", ok=False)

    out_text, w1 = _decode(proc.stdout)
    err_text, w2 = _decode(proc.stderr)
    limit = DISPLAY_CHARS
    if sub == "diff" and len(out_text) > limit:
        # diff 的**尾部是最新改的**，通常最相关 → 尾部多留一点
        head, tail = int(limit * 0.4), limit - int(limit * 0.4)
        body = (out_text[:head]
                + f"\n\n……（diff 中间省略 {len(out_text) - limit:,} 字符）……\n\n"
                + out_text[-tail:])
    else:
        body = _truncate(out_text.rstrip() or "(no stdout)")

    text = f"git {sub} {' '.join(argv)} → 退出码 {proc.returncode}\n\n{body}"
    if err_text.strip():
        text += f"\n\n--- stderr ---\n{_truncate(err_text.rstrip())}"
    if w1 or w2:
        text += (w1 + w2)
    return ToolOutcome(content=text, ok=(proc.returncode == 0))


def _git_args_write(sub: str, argv: list[str]) -> bool:
    """同名的只读/写子命令，看参数区分。主要是 `stash` / `tag` / `branch`。"""
    if sub in ("stash", "tag", "branch", "remote", "config"):
        return bool(argv) and argv[0] not in ("list", "-l", "--list", "-v", "--help")
    return False
