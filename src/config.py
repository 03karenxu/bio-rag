from pathlib import Path

# aws model id strings
_NOVA_PRO = "bedrock/us.amazon.nova-pro-v1:0"
_TITAN = "bedrock/amazon.titan-embed-text-v2:0"
_COHERE = "cohere-bedrock/embed-v4" # custom litellm adapter for cohere
_OPUS = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"

# models
EMBED_MODEL = _COHERE
ANSWER_MODEL = _OPUS

# paths
ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT/"dataset"
SRC = ROOT/"src"
FIGURES_DIR = ROOT/"figures"
CACHE_DIR = ROOT/"preprocess_cache"
LOG_DIR = ROOT/"logs"

# chunking
MIN_CHUNK_TOKENS = 30
BATCH_MAX_TOKENS = 10000

# embedding
EMBED_INIT_DELAY = 1
MAX_EMBED_ATTEMPTS = 5
EMBED_DIMENSION = 1024 # one of 256, 512, 1024, 1536

# concurrency
MAX_CONCURRENT_EMBED = 3
MAX_CONCURRENT_PROCESS = 10 # controls num concurrent papers processed in preprocess.py

# cohere stuff
COHERE_COMPATIBLE_FORMATS = {".png", ".jpeg", ".jpg", ".webp", ".gif"}
COHERE_TRANSFORMABLE_FORMATS = {".pdf", ".tif", ".tiff"}
COHERE_MAX_W = 1536
COHERE_MAX_H = 2048
COHERE_BATCH_MAX = 96