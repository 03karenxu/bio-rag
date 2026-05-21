import random
import logging
import asyncio
import json
import boto3
import litellm
litellm.suppress_debug_info = True
from litellm.types.utils import Embedding
from litellm import aembedding, CustomLLM, EmbeddingResponse, Usage
from config import EMBED_INIT_DELAY, EMBED_MODEL, MAX_EMBED_ATTEMPTS, EMBED_DIMENSION

logger = logging.getLogger(__name__)

class CohereBedrockAdapter(CustomLLM):
    '''
    custom litellm adapter for Cohere embed v4 that allows for interleaved input
    '''
    def __init__(self, model_id: str = "us.cohere.embed-v4:0", region: str = "us-west-2"):
        self.model_id = model_id
        self.bedrock = boto3.client("bedrock-runtime", region_name=region)

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

async def embed_with_retry(input_: list[str | dict],
                           sem: asyncio.Semaphore | None = None,
                           output_dim: int = EMBED_DIMENSION) -> list[list[float]]:
    retry_delay = EMBED_INIT_DELAY
    for attempt in range(1, MAX_EMBED_ATTEMPTS + 1):
        try:
            async with (sem or asyncio.nullcontext()):
                logger.info(f"Embedding {len(input_)} items...")
                resp = await aembedding(model=EMBED_MODEL,
                                        input=input_,
                                        output_dimension=output_dim)
            logger.info(f"Received {len(input_)} embeddings")
            embeddings = [item["embedding"] for item in resp.data]
            return embeddings
        except Exception as e:
            if attempt == MAX_EMBED_ATTEMPTS:
                error_message = f"Max retry attempts reached. Skipping {len(input_)} embeddings: {input_}"
                logger.error(error_message)
                raise e
            logger.error(f"Embed attempt {attempt} failed for {len(input_)} items. Retrying in {retry_delay}s")
            await asyncio.sleep(retry_delay)
            retry_delay = retry_delay * 2 + random.uniform(0, 1)