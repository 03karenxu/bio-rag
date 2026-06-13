import logging
import argparse
from tqdm import tqdm

from config import DATASET_DIR
from utils.logging import init_logging
from download.core import download_paper
from download.fetchers.pmc import PMCFetcher
from download.fetchers.elsevier import SDFetcher
from preprocess.xml_parsers.jats import parse_file

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    init_logging("download_refs.log")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--in-dir", default="lit_reviews_test", help="Directory to read from")
    arg_parser.add_argument("--dry-run", action="store_true", help="Test code")
    args = arg_parser.parse_args()

    PMC     = PMCFetcher()
    SCIDIR  = SDFetcher()

    in_dir  = DATASET_DIR / args.in_dir
    if not in_dir.exists(): raise FileNotFoundError(f"{in_dir} does not exist.")
    
    for content_dir in in_dir.iterdir():
        ref_dir = content_dir / "refs"
        if ref_dir.exists():
            logger.warning(f"References already exist in {content_dir.name}, skipping")
            continue
        elif not args.dry_run:
            ref_dir.mkdir()
        
        # find paper file, get references
        xml_file = next((content_dir / "paper").glob("*.xml"), None)
        assert xml_file, f"No .xml file found in {content_dir / "paper"}"
        parsed = parse_file(xml_file)

        total           = 0
        total_skipped   = 0
        total_saved     = 0
        for ref in tqdm(parsed.references, desc=f"Downloading {content_dir.name} refs", total=len(parsed.references)):
            total += 1
            pmcid = ref.pmcid
            doi = ref.doi

            # attempt to fetch paper
            try:
                assert pmcid, "No pmcid"
                if not args.dry_run:
                    fetched = PMC.fetch_by_pmcid(pmcid=pmcid)
            except Exception:
                try:
                    assert doi, "No doi or pmcid"
                    if not args.dry_run:
                        fetched = SCIDIR.fetch_by_doi(doi=doi)
                except Exception as e:
                    logger.warning(f"Could not fetch ref {ref.id} in {content_dir.name}: {e}")
                    total_skipped += 1
                    continue
            
            if args.dry_run:continue

            # attempt to download paper
            try:
                out_path = download_paper(fetched, out_dir=ref_dir)
                logger.info(f"New file created: {out_path}")
                total_saved += 1
            except Exception as e:
                logger.warning(f"Failed to download {fetched.identifier}")
                total_skipped += 1
    
        logger.info(f"Done downloading references for {content_dir.name}: {total_saved}/{total} saved, {total_skipped}/{total} skipped.")

                


