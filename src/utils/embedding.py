import os
import json
import boto3
import random
import logging
import asyncio
import litellm
from typing import Callable
from functools import wraps
from botocore.config import Config
from litellm.types.utils import Embedding
from litellm import aembedding, CustomLLM, EmbeddingResponse, Usage
from config import EMBED_INIT_DELAY, EMBED_MODEL, MAX_EMBED_ATTEMPTS, SRC
from dotenv import load_dotenv
logger = logging.getLogger(__name__)

load_dotenv(SRC / ".env")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

## cohere-specific ##

class CohereBedrockAdapter(CustomLLM):
    '''
    custom litellm adapter for Cohere embed v4 that allows for interleaved input
    '''
    def __init__(self, model_id: str = "us.cohere.embed-v4:0", region: str = "us-west-2"):
        self.model_id = model_id
        self.bedrock = boto3.client(
            "bedrock-runtime", region_name=region,
            config=Config(read_timeout=30, connect_timeout=5),
        )

    def _build_body(self, input_: list, **kwargs) -> dict:
        body = {
            # mandatory param
            "input_type": kwargs.get("input_type", "search_document"),
        }
        
        # optional params
        for key in ("output_dimension", "max_tokens", "truncate", "embedding_types"):
            if key in kwargs:
                body[key] = kwargs[key]

        if not input_:
            raise ValueError("input is empty")

        contents_list = []
        for item in input_:
            if isinstance(item, dict) and "content" in item:
                contents_list.append(item)
            else:
                raise ValueError(f"Unexpected input format: {item}")

        body["inputs"] = contents_list

        return body

    def _invoke(self, body: dict) -> EmbeddingResponse:
        raw = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(raw["body"].read())
        vectors = result["embeddings"]["float"]

        return EmbeddingResponse(
            model=self.model_id,
            data=[Embedding(embedding=v, index=i, object="embedding") for i, v in enumerate(vectors)],
            usage=Usage(prompt_tokens=0, total_tokens=0),
        )

    def embedding(self, model: str, input: list, **kwargs) -> EmbeddingResponse:
        return self._invoke(self._build_body(input, **kwargs))

    async def aembedding(self, model: str, input: list, **kwargs) -> EmbeddingResponse:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._invoke(self._build_body(input, **kwargs))
        )

_adapter = CohereBedrockAdapter()

litellm.custom_provider_map = [
    {"provider": "cohere-bedrock", "custom_handler": _adapter}
]

## general embedding util ##

def with_retry(max_attempts: int, init_delay: float):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_delay = init_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise

                    logger.error(
                        f"Attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {retry_delay:.2f}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = retry_delay * 2 + random.uniform(0, 1)

        return wrapper
    return decorator

@with_retry(max_attempts=MAX_EMBED_ATTEMPTS, init_delay=EMBED_INIT_DELAY)
async def embed(input_: list[str] | str, **kwargs) -> list[list[float]]:
    logger.debug(f"Embedding {len(input_) if isinstance(input_, list) else 1} item(s)...")
    resp = await aembedding(
        model=EMBED_MODEL,
        input=input_,
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
        **kwargs
    )
    logger.debug(f"Received {len(input_)} embeddings")
    embeddings = [item["embedding"] for item in resp.data]
    if len(embeddings) == 1:
        return embeddings[0]
    return embeddings