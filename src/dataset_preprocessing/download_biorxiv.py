# download_biorxiv.py
# 
# date: May 5th 2026
#
# fetches and downloads the latest n papers in from biorxiv s3 bucket

import io
import boto3
import logging
import zipfile
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.logging import init_logging
from config import DATASET_DIR, COHERE_COMPATIBLE_FORMATS
from utils.image_processing import save_as_jpg, resize_and_save

s3_client = boto3.client("s3", region_name="us-east-1")
logger = logging.getLogger(__name__)

BUCKET = "biorxiv-src-monthly"

# ------------------------------------------------------------------------------

def get_latest_folder() -> str:
    '''
    gets the latest folder in the biorxiv s3 bucket
    '''

    response = s3_client.list_objects_v2(
        Bucket=BUCKET,
        Prefix="Current_Content/",
        Delimiter="/",
        RequestPayer="requester"
    )
    folders = [cp["Prefix"] for cp in response.get("CommonPrefixes", [])]
    
    def parse_folder_date(prefix):
        date = prefix.strip("/").split("/")[-1]
        try:
            return datetime.strptime(date, "%B_%Y")
        except ValueError:
            return datetime.min
    
    return max(folders, key=parse_folder_date)


def list_recent_keys(n: int, folder: str = None) -> list[str]:
    '''
    gets the top n most recent keys in either the specified folder, or the
    most recent one if not given
    '''
    folder = f"Current_Content/{folder}" or get_latest_folder()
    logger.info(f"Fetching from folder: {folder}")
    
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(
        Bucket=BUCKET,
        Prefix=folder,
        RequestPayer="requester"
    )
    
    all_keys = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".meca"):
                all_keys.append((obj["LastModified"], obj["Key"]))
    
    all_keys.sort(key=lambda x: x[0], reverse=True)
    
    return [key for _, key in all_keys[:n]]


def fetch_paper(key: str) -> tuple[str, dict[str, bytes]]:
    '''
    fetches a single .meca from s3 and returns its content files as a dict
    of {filename: bytes}.
    '''
    obj = s3_client.get_object(Bucket=BUCKET, Key=key, RequestPayer="requester")
    data = obj["Body"].read()
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for file_path in z.namelist():
            if file_path.startswith("content/"):
                filename = Path(file_path).name
                with z.open(file_path) as f:
                    files[filename] = f.read()
    return key, files


def build_s3_index(folder: str = None) -> dict[str, str]:
    """
    Lists all .meca keys in the given folder (defaults to latest) and returns
    a dict mapping accession number -> S3 key.  One S3 listing for all DOIs.
    """
    folder = folder or get_latest_folder()
    logger.info(f"Building S3 index from {folder}")
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET, Prefix=folder, RequestPayer="requester")
    index: dict[str, str] = {}
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".meca"):
                accession = Path(key).stem
                index[accession] = key
    logger.info(f"Indexed {len(index)} .meca files")
    return index


def fetch_xmls_by_dois(
    dois: list[str],
    s3_index: dict[str, str],
    max_workers: int = 10,
) -> dict[str, str]:
    """
    Given a list of DOIs and a pre-built S3 index, fetches the XML from each
    matching .meca in parallel. Returns {doi: xml_text} for found papers only.
    """
    doi_to_key: dict[str, str] = {}
    for doi in dois:
        accession = doi.split("/")[-1]
        if accession in s3_index:
            doi_to_key[doi] = s3_index[accession]
        else:
            logger.warning(f"No S3 entry for DOI {doi} (accession {accession})")

    results: dict[str, str] = {}

    def _fetch_xml(doi: str, key: str) -> tuple[str, str | None]:
        _, files = fetch_paper(key)
        for filename, content in files.items():
            if filename.endswith(".xml"):
                return doi, content.decode("utf-8", errors="replace")
        return doi, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doi = {executor.submit(_fetch_xml, doi, key): doi for doi, key in doi_to_key.items()}
        for future in tqdm(as_completed(future_to_doi), total=len(future_to_doi), desc="bioRxiv XMLs"):
            doi = future_to_doi[future]
            try:
                _, xml = future.result()
                if xml is not None:
                    results[doi] = xml
                else:
                    logger.warning(f"No XML found in .meca for {doi}")
            except Exception as e:
                logger.error(f"Failed to fetch XML for {doi}: {e}")

    return results


def download_paper(key: str, files: dict[str, bytes], output_dir: Path) -> None:
    '''
    saves a paper's content files to disk under output_dir/<stem>/.
    '''
    key_dir = output_dir / Path(key).stem
    if key_dir.exists():
        logger.warning(f"Already downloaded content for {key}")
        return
    
    for filename, content in files.items():

        # download main paper
        if Path(filename).stem.isdigit():
            outpath = key_dir / "paper" / filename
            outpath.parent.mkdir(parents=True, exist_ok=True)
            outpath.write_bytes(content)

        # download media files
        else:
            outpath = key_dir / "media" / filename
            outpath.parent.mkdir(parents=True, exist_ok=True)
            if Path(filename).suffix.lower() in COHERE_COMPATIBLE_FORMATS:
                resize_and_save(outpath, content)
            elif Path(filename).suffix:
                try:
                    save_as_jpg(outpath, content)
                except Exception as e:
                    logger.warning(f"Could not convert {filename} to .png, saving as-is: {e}")
                    outpath.write_bytes(content)
        

def download_papers(n: int, output_dir: Path, max_workers: int = 10, folder: str = None) -> list[str]:
    '''
    downloads the latest n papers from the specified (or most recent if unspecified)
    biorxiv s3 dump. returns a list of failed keys
    '''
    logger.info(f"Downloading {n} papers...")
    keys = list_recent_keys(n=n, folder=folder or get_latest_folder())

    def process_paper(key, output_dir):
        key, files = fetch_paper(key)
        download_paper(key, files, output_dir)

    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future2key = {executor.submit(process_paper, k, output_dir): k for k in keys}
        for future in tqdm(as_completed(future2key), total=len(future2key)):
            key = future2key[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Download failed for {key}: {e}")
                failed.append(key)

    logger.info(f"Done downloading!")
    return failed

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    init_logging("download_biorxiv.log")
    parser = argparse.ArgumentParser(description="Download preprints from bioRxiv S3 bucket")
    parser.add_argument("--n-files", type=int, default=10, help="Number of preprints to download")
    parser.add_argument("--out-dir", type=Path, default="papers", help="Output directory (within dataset dir)")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of parallel s3 fetches")
    parser.add_argument("--s3-folder", required=False, help="The folder to pull from in the s3 bucket")
    args = parser.parse_args()

    output_dir = DATASET_DIR / args.out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = download_papers(args.n_files, output_dir, max_workers=args.max_workers, folder=args.s3_folder)

    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")

    