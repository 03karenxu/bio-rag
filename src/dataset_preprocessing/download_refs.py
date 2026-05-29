# download_refs.py

import json
import time
import shutil
import logging
import argparse
import requests
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup

from utils.logging import init_logging
from config import CACHE_DIR, DATASET_DIR
from utils.schemas import Paper
from dataset_preprocessing.download_pmc import fetch_full_text

logger = logging.getLogger(__name__)

def download_by_pmcid(pmcid: str, output_dir: Path, request_count: list[int]) -> None:
    xml_path = output_dir / f"{pmcid}.xml"
    if xml_path.exists():
        return
    xml_text = fetch_full_text(pmcid, "pmc")
    request_count[0] += 1
    if request_count[0] % 3 == 0:
        time.sleep(1)

    if "<article" not in xml_text and "<pmc-articleset" not in xml_text:
        raise ValueError(f"No article content for PMC{pmcid}: {xml_text}")

    xml_path.write_text(xml_text)

SCIHUB_BASE = "https://sci-hub.box"
_last_scihub_fetch: float = 0.0

def download_by_doi(doi: str, output_dir: Path) -> None:
    safe_name = doi.replace("/", "_")
    pdf_path = output_dir / f"{safe_name}.pdf"
    if pdf_path.exists():
        return

    global _last_scihub_fetch
    elapsed = time.time() - _last_scihub_fetch
    if elapsed < 3.0:
        time.sleep(3.0 - elapsed)

    # get html page
    url = f"{SCIHUB_BASE}/{doi}"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"})
    _last_scihub_fetch = time.time()
    r.raise_for_status()

    # get full pdf
    soup = BeautifulSoup(r.text, 'html.parser')
    pdf_url_elem = soup.find(attrs={"name": "citation_pdf_url"})
    if not pdf_url_elem:
        raise ValueError(f"Could not find pdf url at {url}")
    pdf_url = pdf_url_elem.get("content")

    r = requests.get(f"{SCIHUB_BASE}{pdf_url}", timeout=30, headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"})
    _last_scihub_fetch = time.time()
    r.raise_for_status()

    pdf_path.write_bytes(r.content)


if __name__ == "__main__":
    init_logging("download_refs.log")

    parser = argparse.ArgumentParser(description="Download referenced papers from PMC lit reviews")
    parser.add_argument("--in-dir", default="lit_reviews_test", help="Cache subdir containing lit review JSONs")
    parser.add_argument("--out-dir", default="lit_reviews_test", help="Dataset subdir to save reference XMLs")
    parser.add_argument("--overwrite", type=bool, default=False, help="Overwrite previous saved references?")
    args = parser.parse_args()

    input_dir = CACHE_DIR / args.in_dir
    output_dir = DATASET_DIR / args.out_dir
    if not input_dir.exists() or not output_dir.exists():
        raise FileNotFoundError(f"{input_dir} and {output_dir} must exist")

    json_files = sorted(input_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} lit review files in {input_dir}")

    total_saved = total_failed = total_skipped = 0
    request_count = [0]

    for json_path in json_files:
        paper = Paper.model_validate(json.loads(json_path.read_text()))
        parent_pmcid = json_path.stem
        refs_out_dir = output_dir / parent_pmcid / "refs"
        if args.overwrite and refs_out_dir.exists():
            shutil.rmtree(refs_out_dir)
        refs_out_dir.mkdir(parents=True, exist_ok=True)

        for ref in tqdm(paper.references, desc="{json_path.stem} refs:"):
            if ref.pub_type != "journal":
                total_skipped += 1
                logger.warning(f"Reference is not journal: {ref.title} in {json_path.name}")
                continue

            if ref.pmcid:
                try:
                    download_by_pmcid(ref.pmcid, refs_out_dir, request_count)
                    total_saved += 1
                except Exception as e:
                    logger.error(f"PMC error on '{ref.title}' in {json_path}: {e}")
                    total_failed += 1
            elif ref.doi:
                try:
                    download_by_doi(ref.doi, refs_out_dir)
                    total_saved += 1
                except Exception as e:
                    logger.error(f"DOI download error on '{ref.title}' in {json_path}: {e}")
                    total_failed += 1
            else:
                logger.warning(f"No pmcid or doi for '{ref.title}' in {json_path.name}, skipping")
                total_skipped += 1

    logger.info(f"Done. Total: {total_saved} saved, {total_skipped} skipped, {total_failed} failed")
