from pathlib import Path

# aws model id strings
_NOVA_PRO = "bedrock/us.amazon.nova-pro-v1:0"
_TITAN = "bedrock/amazon.titan-embed-text-v2:0"
_COHERE = "cohere-bedrock/embed-v4" # custom litellm adapter for cohere
_OPUS = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"
_QWEN = "ollama/qwen2.5:3b"
_GEMMA = "ollama/gemma3:1b"
_HAIKU = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
_DSFLASH = "deepseek/deepseek-v4-flash"
_NEMOTRON_EMBED = "openrouter/nvidia/llama-nemotron-embed-vl-1b-v2:free"
_QWEN_EMBED = "qwen/qwen3-embedding-8b"

# models
EMBED_MODEL = _NEMOTRON_EMBED
CLAIM_MODEL = _DSFLASH
ANSWER_MODEL = None
TRIPLE_MODEL = _DSFLASH
QA_GEN_MODEL = _DSFLASH

# paths
ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT/"datasets"
SRC = ROOT/"src"
FIGURES_DIR = ROOT/"figures"
CACHE_DIR = ROOT/"preprocess_cache"
LOG_DIR = ROOT/"logs"

# chunking
MIN_CHUNK_TOKENS = 50
BATCH_MAX_TOKENS = 5000

# embedding
EMBED_INIT_DELAY = 5
MAX_EMBED_ATTEMPTS = 5
EMBED_DIMENSION = 1536 # one of 256, 512, 1024, 1536

# concurrency
MAX_CONCURRENT_EMBED = 3
MAX_CONCURRENT_PROCESS = 10 # controls num concurrent papers processed in preprocess.py

# cohere stuff
COHERE_COMPATIBLE_FORMATS = {".png", ".jpeg", ".jpg", ".webp", ".gif"}
COHERE_TRANSFORMABLE_FORMATS = {".pdf", ".tif", ".tiff"}
COHERE_MAX_W = 1536
COHERE_MAX_H = 2048
COHERE_BATCH_MAX = 96