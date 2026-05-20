# preprocess.py
#
# May 6th, 2026
#
# processes the evaluation dataset into metadata + chunks + embeddings

import base64
import logging
import asyncio
import argparse
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

from utils.log import init_logging
from utils.embed import embed_with_retry
from utils.image_handling import get_image_paths
from utils.xml_parser import PaperParser, Paper, Chunk, MEDIA_MARKER
from config import DATASET_DIR, CACHE_DIR, MAX_CONCURRENT_EMBED, BATCH_MAX_TOKENS, LOG_DIR, MAX_CONCURRENT_PROCESS, COHERE_BATCH_MAX
 
logger = logging.getLogger(__name__)

def _chunk_to_input(chunk: Chunk, paper_dir: Path) -> dict:
    '''
    converts a chunk to the format expected for Cohere embed v4
    '''
    text = f"Section: {chunk.section} Content: {chunk.text}"
    content: list[dict] = []
    last = 0
    
    for m in MEDIA_MARKER.finditer(text):
        before = text[last:m.start()].strip()
        if before:
            content.append({"type": "text", "text": before})

        og_media_path: Path = paper_dir / m.group(1)
        converted_media_paths: list[Path] = get_image_paths(og_media_path)

        for path in converted_media_paths:
            ext = path.suffix.lstrip(".").lower()
            if ext == "jpg":
                ext = "jpeg"
            b64 = base64.b64encode(path.read_bytes()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{ext};base64,{b64}"},
            })

        last = m.end()

    after = text[last:].strip()
    if after:
        content.append({"type": "text", "text": after})

    return {"content": content}


def _make_batches(paper: Paper, max_tokens: int = BATCH_MAX_TOKENS) -> list[list[Chunk]]:
    '''
    takes a paper and batches its chunks. enforces cohere's max batch size of 
    96 items, and ensures that each batch has a token size of max_tokens.
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


async def _embed_paper(paper: Paper, paper_dir: Path, sem: asyncio.Semaphore) -> Paper:
    batches = _make_batches(paper)
    num_batches = len(batches)
    for i, batch in enumerate(batches):
        to_embed = [_chunk_to_input(chunk, paper_dir) for chunk in batch]
        embeddings = await embed_with_retry(input_=to_embed, sem=sem)
        logger.info(f"Done batch {i+1}/{num_batches} ({len(batch)} items) of {paper.title}")
        for j, chunk in enumerate(batch):
            chunk.embedding = embeddings[j]

    return paper


async def process_paper(paper_dir: Path, out_dir: Path, embed_sem: asyncio.Semaphore, process_sem: asyncio.Semaphore) -> None:
    '''
    processes a single paper. reads the paper's xml file and generates a preprocessed,
    embedded version in out_dir.
    '''
    if paper_dir.name == ".DS_Store":
        return
    out_path = out_dir / f"{paper_dir.name}.json"
    if out_path.exists():
        logger.info(f"Skipping {paper_dir.name}, already processed")
        return
    
    async with process_sem:
        xml_files = list(paper_dir.glob("*.xml"))
        if len(xml_files) > 1:
            logger.warning(f"Multiple XML files found in {paper_dir.name}, using first")
        elif len(xml_files) == 0:
            raise ValueError(f"No XML found in {paper_dir.name}")
        
        parser = PaperParser(xml=xml_files[0])
        paper = await asyncio.to_thread(parser.parse_paper)

    paper = await _embed_paper(paper, paper_dir, embed_sem)

    with open(out_path, "w") as f:
        f.write(paper.model_dump_json())


async def process_dataset(dataset: Path, out_dir: Path) -> None:
    '''
    generates preprocessed cache files for the evaluation dataset. reads
    from a specified folder and outputs cache files.
    '''

    process_sem = asyncio.Semaphore(MAX_CONCURRENT_PROCESS)
    embed_sem = asyncio.Semaphore(MAX_CONCURRENT_EMBED)

    paper_dirs = [d for d in dataset.iterdir() if d.name != ".DS_Store"]

    tasks = [process_paper(d, out_dir, embed_sem, process_sem) for d in paper_dirs]
    failed = []
    for task in tqdm_asyncio.as_completed(tasks, total=len(paper_dirs)):
        try:
            await task
        except Exception as e:
            logger.error(f"Task failed: {e}")
            failed.append(e)

    if failed:
        logger.error(f"{len(failed)}/{len(paper_dirs)} papers failed")
    else:
        logger.info(f"All {len(paper_dirs)} papers processed successfully")


if __name__ == "__main__":
    init_logging(LOG_DIR/"preprocess.log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="papers_test", help="The name of the folder to process")
    parser.add_argument("--out-dir", type=Path, default="cache_test", help="The name of output folder (within preprocess_cache)")
    args = parser.parse_args()

    in_dir = DATASET_DIR / args.dataset
    if not in_dir.exists():
        raise FileNotFoundError(f"{in_dir} does not exist")
    
    out_dir = CACHE_DIR / args.out_dir
    Path(out_dir).mkdir(exist_ok=True)

    logger.info(f"Processing {in_dir}, dumping to {out_dir}")
    asyncio.run(process_dataset(in_dir, out_dir))