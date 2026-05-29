import os
import time
import urllib
import shutil
import logging
import requests
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET

from config import DATASET_DIR
from utils.logging import init_logging

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

logger = logging.getLogger(__name__)


def search_pmc(query: str, retmax: int, retstart: int) -> tuple[list[str], int]:
    encoded = urllib.parse.quote(query)
    url = BASE_URL + f"esearch.fcgi?db=pmc&term={encoded}&retmax={retmax}&retstart={retstart}"
    r = requests.get(url)
    root = ET.fromstring(r.text)
    total = int(root.findtext("Count") or 0)
    ids = [elem.text for elem in root.findall(".//Id")]
    return ids, total


def fetch_full_text(id: str, db: str) -> str:
    r = requests.get(BASE_URL + f"efetch.fcgi?db={db}&id={id}")
    return r.text


if __name__ == "__main__":
    init_logging("download_pmc.log")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of articles to download")
    parser.add_argument("--output-dir", default="pmc_xml", help="Directory to save XML files")
    parser.add_argument("--overwrite", type=bool, default=False, help="Overwrite folders?")
    args = parser.parse_args()

    output_dir = DATASET_DIR / args.output_dir
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    batch_size = args.n * 2
    query = "\"systematic review\"[ti]"
    saved = 0
    retstart = 0
    request_count = 0

    logger.info(f"Searching for {args.n} articles...")
    with tqdm(total=args.n) as pbar:
        while saved < args.n:
            pmcids, total = search_pmc(query, batch_size, retstart)

            for pmcid in pmcids:
                if saved >= args.n:
                    break

                paper_dir = output_dir / f"PMC{pmcid}" / "paper"
                os.makedirs(paper_dir, exist_ok=True)
                xml_path = paper_dir / f"PMC{pmcid}.xml"

                pbar.set_description(f"[{saved+1}/{args.n}] Fetching PMC{pmcid}")
                xml_text = fetch_full_text(pmcid, "pmc")
                request_count += 1
                if request_count % 3 == 0:
                    time.sleep(1)

                with open(xml_path, "w") as f:
                    f.write(xml_text)

                saved += 1
                pbar.update(1)

            retstart += len(pmcids)
            if not pmcids or retstart >= total:
                logger.warning(f"Exhausted all {total} PMC results after saving {saved} articles.")
                break

    logger.info(f"Done. Saved {saved} XML files to {args.output_dir}/")
