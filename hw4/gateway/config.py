import os
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_DIR = Path(os.getenv("PROMPT_DIR", BASE_DIR / "prompts"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

# MOCK_LLM=1：所有 provider 都换成本地 MockProvider，无需 API Key 就能跑通全部功能（做 demo / 测试用）
MOCK_LLM = os.getenv("MOCK_LLM", "0") == "1"

# MCP 风格工具服务的地址（mcp_server/server.py）
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:9000")


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str  # OpenAI 兼容接口的 base url（不含 /chat/completions）
    api_key: str


# 三家都兼容 OpenAI 协议，所以同一个 Provider 类就能适配。
# 想加新的 provider：在这里多加一项即可。
PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        "openai",
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        os.getenv("OPENAI_API_KEY", ""),
    ),
    "deepseek": ProviderConfig(
        "deepseek",
        os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        os.getenv("DEEPSEEK_API_KEY", ""),
    ),
    "qwen": ProviderConfig(
        "qwen",
        os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        os.getenv("DASHSCOPE_API_KEY", ""),
    ),
}


@dataclass(frozen=True)
class ModelRoute:
    provider: str
    model: str


# cheap / strong 两档模型，各自绑定到一个 provider（默认：便宜的走 DeepSeek，强的走 OpenAI）
TIERS: dict[str, ModelRoute] = {
    "cheap": ModelRoute(os.getenv("CHEAP_PROVIDER", "deepseek"), os.getenv("CHEAP_MODEL", "deepseek-chat")),
    "strong": ModelRoute(os.getenv("STRONG_PROVIDER", "openai"), os.getenv("STRONG_MODEL", "gpt-4o")),
}

# RAG embedding：hash（本地哈希向量，零依赖，默认）| api（调用 provider 的 /embeddings）
EMBED_MODE = os.getenv("EMBED_MODE", "hash")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
