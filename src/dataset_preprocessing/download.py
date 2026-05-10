# download.py
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
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import DATASET_DIR

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
    folder = folder or get_latest_folder()
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


def fetch_papers(n: int, max_workers: int = 10, folder: str = None) -> Iterator[tuple[str, dict[str, bytes]]]:
    '''
    streams the latest n papers from the specified (or most recent if unspecified)
    biorxiv s3 dump. yields (key, files) tuples where files is a dict of {filename: bytes}.
    '''
    keys = list_recent_keys(n=n, folder=folder or get_latest_folder())
 
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future2key = {executor.submit(fetch_paper, k): k for k in keys}
        for future in as_completed(future2key):
            key = future2key[future]
            try:
                yield future.result()
            except Exception as e:
                logger.error(f"Fetch failed for {key}: {e}")


def save_paper(key: str, files: dict[str, bytes], output_dir: Path) -> None:
    '''
    saves a paper's content files to disk under output_dir/<stem>/.
    '''
    key_dir = output_dir / Path(key).stem
    if key_dir.exists():
        logger.warning(f"Already downloaded content for {key}")
        return
    key_dir.mkdir(parents=True)
    for filename, content in files.items():
        (key_dir / filename).write_bytes(content)

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download preprints from bioRxiv S3 bucket")
    parser.add_argument("--n-files", type=int, default=10, help="Number of preprints to download")
    parser.add_argument("--output-dir", type=Path, default="papers", help="Output directory (within dataset dir)")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of parallel downloads")
    args = parser.parse_args()
 
    output_dir = DATASET_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
 
    failed = []
    for key, files in tqdm(fetch_papers(n=args.n_files, max_workers=args.max_workers),
                           total=args.n_files, desc="Downloading"):
        try:
            save_paper(key, files, output_dir)
        except Exception as e:
            failed.append(key)
            logger.error(f"Save failed for {key}: {e}")
 
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")

    