from pathlib import Path

_NOVA_PRO = "bedrock/us.amazon.nova-pro-v1:0"
_TITAN = "bedrock/amazon.titan-embed-text-v2:0"
_OPUS = "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"

# paths
ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT/"dataset"
SRC = ROOT/"src"
FIGURES_DIR = ROOT/"figures"
CACHE_DIR = ROOT/"preprocess_cache"

# chunking
MIN_CHUNK_SIZE = 5
MAX_CHUNK_SIZE = 512

# embedding retry
EMBED_INIT_DELAY = 1
MAX_EMBED_ATTEMPTS = 5

# concurrency
MAX_CONCURRENT_EMBED = 10

# models
EMBED_MODEL = _TITAN