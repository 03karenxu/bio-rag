# preprocess.py
#
# May 6th, 2026
#
# processes the evaluation dataset into metadata + chunks + embeddings

import shutil
import base64
import logging
import asyncio
import argparse
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

from utils.logging import init_logging
from utils.schemas import Paper, Chunk
from utils.embeddings import embed_with_retry
from utils.image_processing import get_image_paths
from utils.paper_parser import PaperParser, MEDIA_MARKER
from config import DATASET_DIR, CACHE_DIR, MAX_CONCURRENT_EMBED, BATCH_MAX_TOKENS, MAX_CONCURRENT_PROCESS, COHERE_BATCH_MAX
 
logger = logging.getLogger(__name__)

def _chunk_to_input(chunk: Chunk, media_folder: Path | None) -> dict:
    '''
    converts a chunk to the format expected for Cohere embed v4
    '''
    text = f"Section: {chunk.section} Content: {chunk.text}"
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


async def _embed_paper(paper: Paper, media_folder: Path | None, sem: asyncio.Semaphore) -> Paper:
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

    paper = await _embed_paper(paper, media_folder, embed_sem)

    with open(out_path, "w") as f:
        f.write(paper.model_dump_json())


async def process_dataset(dataset: Path, out_dir: Path) -> None:
    '''
    generates preprocessed cache files for the evaluation dataset. reads
    from a specified folder and outputs cache files. works with papers/,
    pmc_lit_reviews/, and pmc_ref_papers/ layouts.
    '''
    process_sem = asyncio.Semaphore(MAX_CONCURRENT_PROCESS)
    embed_sem = asyncio.Semaphore(MAX_CONCURRENT_EMBED)
    parser = PaperParser()

    items = []
    for parent_dir in dataset.iterdir():
        if not parent_dir.is_dir() or parent_dir.name == ".DS_Store":
            logger.warning(f"Skipping {parent_dir}, not a dir")
            continue

        xml_path: Path = next((parent_dir / "paper").glob("*.xml"), None)
        if xml_path is None:
            logger.warning(f"No XML found in {parent_dir}, skipping")
            continue

        out_path: Path = out_dir / f"{xml_path.stem}.json"
        media_folder: Path | None = parent_dir / "media" if (parent_dir / "media").exists() else None
        items.append((xml_path, out_path, media_folder))

    tasks = [process_paper(xml, out_path, media_folder, parser, embed_sem, process_sem) for xml, out_path, media_folder in items]
    failed = []
    for task in tqdm_asyncio.as_completed(tasks, total=len(items)):
        try:
            await task
        except Exception as e:
            logger.error(f"Task failed: {e}")
            failed.append(e)

    if failed:
        logger.error(f"{len(failed)}/{len(items)} papers failed")
    else:
        logger.info(f"All {len(items)} papers processed successfully")


if __name__ == "__main__":
    init_logging("preprocess.log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=f"papers_test", help="The name of the folder to process")
    parser.add_argument("--out-dir", type=Path, default=f"papers_test", help="The name of output folder (within preprocess_cache)")
    parser.add_argument("--overwrite", type=bool, default=False, help="Overwrite existing cache folder?")
    args = parser.parse_args()

    in_dir = DATASET_DIR / args.in_dir
    if not in_dir.exists():
        raise FileNotFoundError(f"{in_dir} does not exist")
    
    out_dir = CACHE_DIR / args.out_dir
    if args.overwrite:
        logger.warning(f"Overwriting existing cache in {out_dir}")
        shutil.rmtree(out_dir, ignore_errors=True)
    Path(out_dir).mkdir(exist_ok=True)

    logger.info(f"Processing {in_dir}, dumping to {out_dir}")
    asyncio.run(process_dataset(in_dir, out_dir))