# download_refs.py

import json
import time
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

from utils.logging import init_logging
from utils.paper_parser import PaperParser
from config import CACHE_DIR, DATASET_DIR
from utils.schemas import Reference, Paper
from dataset_preprocessing.download_pmc import fetch_full_text
from dataset_preprocessing.download_biorxiv import build_s3_index, fetch_xmls_by_dois

logger = logging.getLogger(__name__)


def download_by_pmcid(pmcid: str, output_dir: Path, request_count: list[int]) -> None:
    xml_path = output_dir / f"PMC{pmcid}.xml"
    if xml_path.exists():
        return

    xml_text = fetch_full_text(pmcid)
    request_count[0] += 1
    if request_count[0] % 3 == 0:
        time.sleep(1)

    if "<article" not in xml_text and "<pmc-articleset" not in xml_text:
        raise ValueError(f"No article content for PMC{pmcid}")

    xml_path.write_text(xml_text)


if __name__ == "__main__":
    init_logging("download_refs.log")

    parser = argparse.ArgumentParser(description="Download referenced papers from PMC lit reviews")
    parser.add_argument("--input-dir", default="pmc_lit_reviews_test", help="Cache subdir containing lit review JSONs")
    parser.add_argument("--output-dir", default="pmc_ref_papers_test", help="Dataset subdir to save reference XMLs")
    args = parser.parse_args()

    input_dir = CACHE_DIR / args.input_dir
    output_dir = DATASET_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} lit review files in {input_dir}")

    total_saved = total_failed = total_skipped = 0
    request_count = [0]

    # --- Pass 1: PMC refs (download immediately) and collect biorxiv DOIs ---
    # doi_to_dest maps doi -> (output_path, paper_output_dir) for later saving
    doi_to_dest: dict[str, tuple[Path, Path]] = {}

    for json_path in tqdm(json_files, desc="PMC refs"):
        paper = Paper.model_validate(json.loads(json_path.read_text()))
        parent_pmcid = json_path.stem
        paper_output_dir = output_dir / parent_pmcid
        paper_output_dir.mkdir(exist_ok=True)

        for ref in paper.references:
            if ref.pub_type != "journal":
                total_skipped += 1
                continue

            if ref.pmcid:
                try:
                    download_by_pmcid(ref.pmcid, paper_output_dir, request_count)
                    total_saved += 1
                except Exception as e:
                    logger.error(f"PMC error on '{ref.title}' in {json_path.name}: {e}")
                    total_failed += 1
            elif ref.doi:
                safe_name = ref.doi.replace("/", "_")
                xml_path = paper_output_dir / f"{safe_name}.xml"
                if xml_path.exists():
                    total_skipped += 1
                else:
                    doi_to_dest[ref.doi] = (xml_path, paper_output_dir)
            else:
                logger.warning(f"No pmcid or doi for '{ref.title}' in {json_path.name}")
                total_skipped += 1

    # --- Pass 2: bioRxiv refs (batch S3 fetch) ---
    if doi_to_dest:
        logger.info(f"Fetching {len(doi_to_dest)} bioRxiv refs from S3...")
        s3_index = build_s3_index()
        xml_map = fetch_xmls_by_dois(list(doi_to_dest.keys()), s3_index)

        for doi, xml_text in xml_map.items():
            xml_path, _ = doi_to_dest[doi]
            xml_path.write_text(xml_text)
            total_saved += 1

        missing = set(doi_to_dest) - set(xml_map)
        for doi in missing:
            logger.warning(f"Not found on bioRxiv: {doi}")
            total_failed += 1

    logger.info(f"Done. Total: {total_saved} saved, {total_skipped} skipped, {total_failed} failed")
