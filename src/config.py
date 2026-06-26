from pathlib import Path

# aws model id strings
_NOVA_PRO = "bedrock/us.amazon.nova-pro-v1:0"
_TITAN = "bedrock/amazon.titan-embed-text-v2:0"
_COHERE = "cohere-bedrock/embed-v4" # custom litellm adapter for cohere
_OPUS = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"
_QWEN = "ollama/qwen2.5:3b"
_GEMMA = "ollama/gemma3:1b"
_HAIKU = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
_DSFLASH = "openrouter/deepseek/deepseek-v4-flash"
_NEMOTRON_EMBED = "openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free"
_QWEN_EMBED = "openrouter/qwen/qwen3-embedding-8b"
_GLM52 = "openrouter/z-ai/glm-5.2"
_DSPRO = "openrouter/deepseek/deepseek-v4-pro"

# models
EMBED_MODEL = _NEMOTRON_EMBED
CLAIM_MODEL = _DSFLASH
ANSWER_MODEL = _DSFLASH
TRIPLE_MODEL = _DSFLASH
QA_GEN_MODEL = _DSFLASH

# paths
ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT/"datasets"
SRC = ROOT/"src"
FIGURES_DIR = ROOT/"figures"
CACHE_DIR = ROOT/"preprocess_cache"
LOG_DIR = ROOT/"logs"

# rag
RETRIEVAL_LIMIT = 3

# chunking
CHUNK_TOKEN_TARGET = 256
BATCH_MAX_TOKENS = 5000

# deduplication
DEDUP_SIM_THRESHOLD = 0.75  # cosine cutoff for embedding-based duplicate clustering

# embedding
EMBED_INIT_DELAY = 5
MAX_EMBED_ATTEMPTS = 5
EMBED_DIMENSION = 2048

# concurrency
MAX_CONCURRENT_EMBED = 3
MAX_CONCURRENT_PROCESS = 10 # controls num concurrent papers processed in preprocess.py

# cohere stuff
COHERE_COMPATIBLE_FORMATS = {".png", ".jpeg", ".jpg", ".webp", ".gif"}
COHERE_TRANSFORMABLE_FORMATS = {".pdf", ".tif", ".tiff"}
COHERE_MAX_W = 1536
COHERE_MAX_H = 2048
COHERE_BATCH_MAX = 96