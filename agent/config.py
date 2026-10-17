"""agent 配置。

所有字段可用环境变量覆盖。.env 读取路径：agent/.env（本包自带）。
"""
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- 身份 ----------
    NAME: str = "助手"
    # 冷层（记忆提取 / 摘要 / 转写）指代用户的词，热层（人设）称「你」。
    # 只影响此后写入的内容。
    USER_TERM: str = "用户"
    # 自我笔记。存在且非空时注入 system 提示词；留空则 DATA_DIR/self.md
    SELF_NOTES_PATH: Path | None = None

    # ---------- 模型 ----------
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    CHAT_MODEL: str = "qwen2.5:7b"
    OLLAMA_NUM_CTX: int = 8192

    # 留空则自动推导为 {OLLAMA_BASE_URL}/v1
    CHAT_BASE_URL: str | None = None
    CHAT_API_KEY: str = "ollama"

    # ---------- 嵌入模型 ----------
    # 走 OpenAI 兼容的 /embeddings 接口；本地 Ollama 用 http://localhost:11434/v1
    EMBED_BASE_URL: str | None = None    # 留空 → 跟随 CHAT_BASE_URL
    EMBED_API_KEY: str | None = None     # 留空 → 跟随 CHAT_API_KEY
    EMBED_MODEL: str = "BAAI/bge-m3"
    # 批量嵌入的切块大小；要嵌一批时走 embed_many
    EMBED_BATCH_SIZE: int = 32
    EMBED_TIMEOUT: int = 60
    # 思考模型（Qwen3.5 等）开关；陪伴对话默认关闭
    ENABLE_THINKING: bool = False
    # 思考开关的传参方式：
    #   top_level            云端 OpenAI 兼容（硅基流动/DeepSeek 等），顶层 enable_thinking
    #   chat_template_kwargs 本地 vLLM 起 Qwen，chat_template_kwargs={"enable_thinking":...}
    #   none                 不传（本地 Ollama）
    THINKING_PARAM_STYLE: str = "top_level"
    # 单次回复的输出上限
    MAX_OUTPUT_TOKENS: int = 1024

    # ---------- 记忆 ----------
    MILVUS_MODE: str = "lite"           # lite=嵌入式 | standalone=http://localhost:19530
    MILVUS_URI: str | None = None       # 留空按 MILVUS_MODE 推导
    COLLECTION_NAME: str = "agent_memory"
    MESSAGE_INDEX_COLLECTION: str = "message_index"
    EMBED_DIM: int = 1024               # bge-m3 输出维度
    MEMORY_MIN_CONFIDENCE: float = 0.6      # 低于此置信度只进 SQLite（待审核）
    INFERENCE_SUPPORT_THRESHOLD: int = 2    # inference 独立佐证达此数 → 提升置信度

    # ---------- 结构化存储 ----------
    DB_PATH: Path | None = None         # 留空则 DATA_DIR/agent.db
    DB_URL: str | None = None           # 留空则由 DB_PATH 推导
    DB_ECHO: bool = False

    # ---------- 上下文计量与压缩 ----------
    CTX_WINDOW: int = 131072            # 模型上下文窗口；本地 27B 部署时按实际改
    CTX_WARN_RATIO: float = 0.60        # ≥ 此比例：尾部提醒，模型可选压缩
    CTX_FORCE_RATIO: float = 0.80       # ≥ 此比例：强制压缩（tool_choice 强制 + 系统兜底）
    COMPRESS_KEEP_TOKENS: int = 4000    # 压缩时逐字保留的最近 token
    COMPRESS_MIN_GAIN_TOKENS: int = 2000  # 可压缩量低于此值则不压（护栏）
    COMPRESS_COOLDOWN_TURNS: int = 3    # 距上次压缩不足此轮数则拒绝（护栏，仅会话内）

    # ---------- Agent 循环护栏 ----------
    MAX_REACT_STEPS: int = 8            # 单回合工具调用步数上限
    # 单次装载的消息条数硬上限。触顶即丢弃更早的消息且不摘要 → "摘要空洞"
    MAX_HISTORY_MESSAGES: int = 400
    TOOL_RESULT_MAX_TOKENS: int = 3000  # 观察掩码：单条工具结果截断上限
    TOOL_SEARCH_TOP_K: int = 6          # 工具检索返回条数
    # 已加载工具集合保持粘性的上限。集合一变 tools= 渲染即变，前缀缓存全失效。
    # 集合只由模型调用 search_tools 引起变化。
    MAX_LOADED_TOOLS: int = 12

    # ---------- tokenizer ----------
    # 本地 Qwen tokenizer 目录；留空则退化为字符估算，并用 API 的 usage 在线校准
    TOKENIZER_PATH: str | None = None

    # ---------- 文件工具沙箱 ----------
    AGENT_WORKSPACE: Path | None = None  # 留空则 DATA_DIR/workspace（陪伴 Profile 的根）
    # 编码 Profile 的默认工作区。留空则取项目根（BASE_DIR 的上一级 = 本仓库）。
    CODING_WORKSPACE: Path | None = None
    WIKI_DIR: Path | None = None         # 留空则 DATA_DIR/wiki

    # ---------- LLM Wiki ----------
    # 主题归并阈值：精确匹配优先，没匹配上的用它做嵌入相似度归并。
    # 调高会漏并同义主题，调低会误并相关但不同的主题。
    WIKI_MERGE_SIMILARITY: float = 0.72
    # 证据数达到这个量才值得花一次 LLM 调用重写「综述」
    WIKI_SUMMARY_MIN_EVIDENCE: int = 2
    # synthesis 页的门槛：证据数 ≥ N **且**跨 ≥ M 天
    WIKI_SYNTHESIS_MIN_EVIDENCE: int = 5
    WIKI_SYNTHESIS_MIN_DAYS: int = 2
    # 构建时是否调用 LLM 写「综述」。关掉也能构建，综述留占位，且不碰 API。
    WIKI_USE_LLM: bool = True
    # 退出整理时是否重建 Wiki。慢（要跑 LLM），但失败无代价：纯派生物。
    WIKI_BUILD_ON_EXIT: bool = True

    # ---------- 编码 Profile ----------
    # 编码任务的回合长度与单次输出长度都远超陪伴
    CODING_MAX_STEPS: int = 40
    CODING_MAX_OUTPUT_TOKENS: int = 8192
    # 审批档：auto（全自动）/ destructive（只问危险操作，默认）/ all_writes（写操作都问）
    APPROVAL_POLICY: str = "destructive"
    # 默认不让编码 agent 改自己（agent/ 目录只读）
    ALLOW_SELF_EDIT: bool = False

    # ---------- 执行工具 ----------
    # 留空则自动探测：pwsh → powershell → cmd（POSIX 上 /bin/bash -lc）。
    # 值会被 shlex.split，带空格的路径要加引号。
    SHELL_COMMAND: str | None = None

    # ---------- 离线任务 ----------
    # 全部在退出时执行（offline.run()）：压缩 → 记忆固化 → 每日总结。
    # 启动阶段（preload）是纯读取，零 LLM 调用、零后台线程。
    OFFLINE_ON_EXIT: bool = True
    OFFLINE_ROLLUP_DAYS: int = 3        # 退出时回溯几天的每日总结
    # 退出压缩的目标：让下次启动逐字装载的 token 不超过这个数。
    # 同时也是"要不要压缩"的闸门（低于此值 + COMPRESS_MIN_GAIN_TOKENS 就直接跳过）
    OFFLINE_KEEP_TOKENS: int = 8000

    # ---------- 联网搜索（阶段 B） ----------
    SEARCH_PROVIDER: str = "duckduckgo"   # duckduckgo | tavily | searxng
    SEARCH_API_KEY: str | None = None
    SEARCH_BASE_URL: str | None = None

    @model_validator(mode="after")
    def _fill_derived(self) -> "Settings":
        """填充可被环境变量覆盖的派生字段。"""
        if self.CHAT_BASE_URL is None:
            self.CHAT_BASE_URL = f"{self.OLLAMA_BASE_URL}/v1"

        # 嵌入默认跟随聊天端点
        if self.EMBED_BASE_URL is None:
            self.EMBED_BASE_URL = self.CHAT_BASE_URL
        if self.EMBED_API_KEY is None:
            self.EMBED_API_KEY = self.CHAT_API_KEY

        if self.MILVUS_URI is None:
            self.MILVUS_URI = (
                str(DATA_DIR / "milvus_lite.db")
                if self.MILVUS_MODE == "lite"
                else "http://localhost:19530"
            )

        if self.DB_PATH is None:
            self.DB_PATH = DATA_DIR / "agent.db"

        if self.DB_URL is None:
            self.DB_URL = f"sqlite:///{self.DB_PATH.as_posix()}"

        if self.AGENT_WORKSPACE is None:
            self.AGENT_WORKSPACE = DATA_DIR / "workspace"
        self.AGENT_WORKSPACE.mkdir(parents=True, exist_ok=True)

        if self.CODING_WORKSPACE is None:
            # 项目根 = agent/ 的上一级（本仓库）
            self.CODING_WORKSPACE = BASE_DIR.parent
        self.CODING_WORKSPACE = self.CODING_WORKSPACE.resolve()

        if self.WIKI_DIR is None:
            self.WIKI_DIR = DATA_DIR / "wiki"

        if self.SELF_NOTES_PATH is None:
            self.SELF_NOTES_PATH = DATA_DIR / "self.md"

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
