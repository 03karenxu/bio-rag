import shutil
import asyncio
import logging
import argparse
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from utils.logging import init_logging
from preprocess.core import process_paper
from config import MAX_CONCURRENT_EMBED, MAX_CONCURRENT_PROCESS, DATASET_DIR, CACHE_DIR

logger = logging.getLogger(__name__)

async def process_dataset(dataset: Path, out_dir: Path) -> None:
    '''
    generates preprocessed cache files for every .xml file in the dataset
    '''
    process_sem = asyncio.Semaphore(MAX_CONCURRENT_PROCESS)
    embed_sem = asyncio.Semaphore(MAX_CONCURRENT_EMBED)

    items = []
    for parent_dir in dataset.iterdir():
        if not parent_dir.is_dir() or parent_dir.name == ".DS_Store":
            logger.warning(f"Skipping {parent_dir}, not a dir")
            continue

        xml_path = next((parent_dir / "paper").glob("*.xml"), None)
        if not xml_path:
            logger.warning(f"No XML file found in {parent_dir}, skipping")
            continue
        
        out_path: Path = out_dir / f"{xml_path.stem}.json"
        media_folder: Path | None = parent_dir / "media" if (parent_dir / "media").exists() else None
        items.append((xml_path, out_path, media_folder))

    tasks = [process_paper(xml, out_path, embed_sem, process_sem, media_folder) for xml, out_path, media_folder in items]
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
    init_logging("preprocess_biorxiv.log")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--in-dir", type=Path, default=f"papers_test", help="The name of the folder to process")
    arg_parser.add_argument("--out-dir", type=Path, default=f"papers_test", help="The name of output folder (within preprocess_cache)")
    arg_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = arg_parser.parse_args()

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