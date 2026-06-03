# download_litrev.py
import shutil
import logging
import argparse
from tqdm import tqdm

from config import DATASET_DIR
from utils.schemas import Paper
from utils.logging import init_logging
from download.core import download_paper
from download.fetchers.pmc import PMCFetcher
from preprocess.paper_parser import PaperParser
from download.fetchers.scihub import ScihubFetcher

logger = logging.getLogger(__name__)

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
    parser = PaperParser()
    scihub = ScihubFetcher()

    total_saved = total_skipped = total_failed = 0

    for paper in pmc.fetch_by_count(n=args.n, query='"systematic review"[ti]'):
        paper_dir = output_dir / paper.identifier
        paper_dir.mkdir(parents=True, exist_ok=True)

        # download literature review paper
        download_paper(paper=paper, out_dir=paper_dir, wrap_main_file=True)

        # download referenced papers
        paper_fn, xml_bytes = paper.main_file
        parsed: Paper = parser.parse_paper(xml=xml_bytes.decode())

        refs_dir = paper_dir / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)

        for ref in tqdm(parsed.references, desc=f"{paper_fn} refs"):
            if ref.pub_type != "journal":
                logger.warning(f"Skipping non-journal ref '{ref.title}' in {paper_fn}")
                total_skipped += 1
                continue

            if not ref.pmcid and not ref.doi:
                logger.warning(f"No pmcid or doi for '{ref.title}' in {paper_fn}, skipping")
                total_skipped += 1
                continue

            try:
                ref_paper = pmc.fetch_by_pmcid(ref.pmcid) if ref.pmcid else scihub.fetch_by_doi(ref.doi)
                download_paper(paper=ref_paper, out_dir=refs_dir)
                total_saved += 1
            except Exception as e:
                logger.error(f"Could not download ref '{ref.title}' in {paper_fn}: {e}")
                total_failed += 1

    logger.info(f"Done. Total: {total_saved} saved, {total_skipped} skipped, {total_failed} failed")