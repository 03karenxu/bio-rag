import random
import logging
import asyncio
from litellm import aembedding
from config import EMBED_INIT_DELAY, EMBED_MODEL, MAX_EMBED_ATTEMPTS

logger = logging.getLogger(__name__)

async def embed_with_retry(to_embed: list[str]) -> list[list[float]]:
    retry_delay = EMBED_INIT_DELAY
    for attempt in range(1, MAX_EMBED_ATTEMPTS + 1):
        try:
            resp = await aembedding(model=EMBED_MODEL, input=to_embed)
            embeddings = [item["embedding"] for item in resp.data]
            return embeddings
        except Exception as e:
            if attempt == MAX_EMBED_ATTEMPTS:
                error_message = f"Max retry attempts reached. Skipping {len(to_embed)} embeddings: {e}"
                logger.error(error_message)
                raise RuntimeError(error_message)
            logger.error(f"Embed attempt {attempt} failed for {len(to_embed)} items. Retrying in {retry_delay}s: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay = retry_delay * 2 + random.uniform(0, 1)