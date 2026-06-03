import logging
import argparse
from pathlib import Path
from config import DATASET_DIR
from download.core import download_paper
from utils.logging import init_logging
from download.fetchers.biorxiv import BiorxivFetcher

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    init_logging("download_biorxiv.log")
    parser = argparse.ArgumentParser(description="Download preprints from bioRxiv S3 bucket")
    parser.add_argument("--n", type=int, default=10, help="Number of preprints to download")
    parser.add_argument("--out-dir", type=Path, default="papers", help="Output directory (within dataset dir)")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of parallel s3 fetches")
    parser.add_argument("--s3-folder", required=False, help="The folder to pull from in the s3 bucket")
    args = parser.parse_args()

    output_dir = DATASET_DIR / args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    biorxiv = BiorxivFetcher(args.s3_folder)

    failed: list[str] = []
    for paper in biorxiv.fetch_by_count(n=args.n):
        paper_dir = output_dir / paper.identifier.replace("/", "_")
        paper_dir.mkdir(parents=True, exist_ok=True)
        try:
            download_paper(paper=paper, out_dir=paper_dir, wrap_main_file=True)
        except Exception as e:
            logger.warning(f"Could not download paper {paper.identifier}, skipping: {e}")
            failed.append(paper.identifier)
            continue

    logger.info(f"Finished downloaded papers ({len(failed)} failed)")

    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")