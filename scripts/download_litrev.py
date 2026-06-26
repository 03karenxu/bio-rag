# download_litrev.py
import shutil
import logging
import argparse

from config import DATASET_DIR
from utils.log import init_logging
from ingest.to_disk import save_to_disk
from ingest.fetch.pmc import PMCFetcher
from ingest.parse_xml.jats import parse_string
from utils.paper_schema import Paper, PubType

logger = logging.getLogger(__name__)

def journal_only_ref(parsed: Paper) -> bool:
    for ref in parsed.references:
        if ref.pub_type != PubType.JOURNAL:
            return False
    return True

if __name__ == "__main__":
    init_logging("download_litrev.log")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--n", type=int, default=10, help="Number of articles to download")
    arg_parser.add_argument("--out-dir", default="lit_reviews_test", help="Directory to save files")
    arg_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = arg_parser.parse_args()

    output_dir = DATASET_DIR / args.out_dir
    if args.overwrite and output_dir.exists():
        logger.warning(f"Overwriting folder {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving files to {output_dir}")

    pmc = PMCFetcher()

    saved_papers = 0
    batch_size = args.n * 10
    batch = pmc.fetch_by_count(n=batch_size, query='"systematic review"[ti]')
    for paper in batch:
        if saved_papers == args.n: break

        if (output_dir / paper.identifier).exists():
            logger.warning(f"{paper.identifier} already saved, skipping")
            continue

        # is it the expected format, and does it contain only journal references?
        try:
            parsed = parse_string(paper.main_file[1].decode('utf-8'))
            assert journal_only_ref(parsed), f"Contains non-journal references"
        except Exception as e:
            logger.warning(f"Skipping download for {paper.identifier}: {e}")
            continue
        
        # download paper
        paper_dir = output_dir / paper.identifier
        paper_dir.mkdir(parents=True, exist_ok=True)
        try:
            paper_path = save_to_disk(paper=paper, out_dir=paper_dir, wrap_main_file=True)
            saved_papers += 1
        except Exception as e:
            logger.warning(f"Could not download {paper.identifier}: {e}")

    if saved_papers < args.n:
        logger.warning(f"Saved only {saved_papers} from batch of {batch_size}")
    else:
        logger.info(f"Downloaded {args.n} paper(s)")