import base64
import logging
import asyncio
from pathlib import Path

from utils.schemas import Paper, Chunk
from utils.embedding import embed_with_retry
from preprocess.img_processing import get_image_paths
from preprocess.paper_parser import PaperParser, MEDIA_MARKER
from config import BATCH_MAX_TOKENS, COHERE_BATCH_MAX
 
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------

async def embed_paper(paper: Paper, media_folder: Path | None, sem: asyncio.Semaphore) -> Paper:
    batches = _make_batches(paper)
    n = len(batches)

    async def _embed_batch(i: int, batch: list) -> list:
        to_embed = [_chunk_to_input(chunk, media_folder) for chunk in batch]
        async with sem:
            embeddings = await embed_with_retry(input_=to_embed)
        logger.info(f"Done batch {i+1}/{n} ({len(batch)} items) of {paper.title}")
        return embeddings

    results = await asyncio.gather(*[_embed_batch(i, batch) for i, batch in enumerate(batches)])
    for batch, embeddings in zip(batches, results):
        for chunk, embedding in zip(batch, embeddings):
            chunk.embedding = embedding
    return paper


async def process_paper(xml_path: Path,
                        out_path: Path,
                        media_folder: Path | None,
                        parser: PaperParser,
                        embed_sem: asyncio.Semaphore,
                        process_sem: asyncio.Semaphore) -> None:
    '''
    processes a single paper. reads the xml file and generates a preprocessed,
    embedded version at out_path.
    '''
    if out_path.exists():
        logger.info(f"Skipping {xml_path.stem}, already processed")
        return

    async with process_sem:
        paper = await asyncio.to_thread(parser.parse_paper, xml_path, media_folder)

    # paper = await embed_paper(paper, media_folder, embed_sem)

    with open(out_path, "w") as f:
        f.write(paper.model_dump_json())

# ------------------------------------------------------------------------------

def _chunk_to_input(chunk: Chunk, media_folder: Path | None) -> dict:
    '''
    converts a chunk to the format expected for Cohere embed v4
    '''
    if chunk.section:
        text = f"Section: {chunk.section} Content: {chunk.text}"
    else:
        text = chunk.text
        
    if not media_folder:
        return {"content": [{"type": "text", "text": text}]}
    
    content: list[dict] = []
    last = 0
    for m in MEDIA_MARKER.finditer(text):
        before = text[last:m.start()].strip()
        if before:
            content.append({"type": "text", "text": before})
        
        for path in get_image_paths(media_folder, m.group(1)):
            ext = path.suffix.lstrip(".").lower()
            if ext == "jpg":
                ext = "jpeg"
            else:
                logger.warning(f"NOT JPG: {path}")
            b64 = base64.b64encode(path.read_bytes()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{ext};base64,{b64}"},
            })

        last = m.end()

    after = text[last:].strip()
    if after:
        content.append({"type": "text", "text": after})

    logger.debug(content)
    
    return {"content": content}


def _make_batches(paper: Paper, max_tokens: int = BATCH_MAX_TOKENS) -> list[list[Chunk]]:
    '''
    takes a paper and batches its chunks. enforces cohere's max batch size of 
    96 items, and ensures that each batch has a token size <= max_tokens.
    '''
    batches, current, current_tokens = [], [], 0

    for chunk in (paper.abstract + paper.body):
        if current_tokens + chunk.n_tokens > max_tokens or len(current) >= COHERE_BATCH_MAX:
            if current:
                batches.append(current)
            current, current_tokens = [chunk], chunk.n_tokens
        else:
            current.append(chunk)
            current_tokens += chunk.n_tokens

    if current:
        batches.append(current)

    return batches