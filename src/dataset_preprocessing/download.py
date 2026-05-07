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


def list_keys(n: int, folder: str, max_mb: int = 15) -> list[str]:
    '''
    lists the first n keys (lexicographical order) in the specified folder
    skips .meca files that are over 15mb (overly large compared to median)
    '''

    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=folder, RequestPayer="requester")
    keys = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".meca"):
                size_mb = obj["Size"] / 1e6
                if size_mb <= max_mb:
                    keys.append(obj["Key"])
                if len(keys) >= n:
                    return keys
                
    return keys


def download_single(key: str, output_dir: Path) -> None:
    '''
    downloads the content folder associated with the given key
    '''
    key_dir = output_dir / (Path(key).stem)
    if key_dir.exists():
        logger.warning(f"Already downloaded content for {key}")
        return
    else:
        key_dir.mkdir()
    
    obj = s3_client.get_object(Bucket=BUCKET, Key=key, RequestPayer="requester")
    data = obj["Body"].read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        content_files = [f for f in z.namelist() if f.startswith("content/")]

        for file_path in content_files:
            filename = Path(file_path).name
            output_path = key_dir / filename
            with z.open(file_path) as f:
                output_path.write_bytes(f.read())


def download_papers(n: int, output_dir: str, max_workers: int = 10) -> list[str]:
    '''
    downloads the first n papers (as pdfs) from the latest biorxiv s3 dump.
    returns the papers where download failed
    '''
    keys = list_recent_keys(n=n, folder=get_latest_folder())
    
    completed = 0
    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future2key = {executor.submit(download_single, k, output_dir): k for k in keys}
        for future in tqdm(as_completed(future2key), total=len(future2key), desc="Downloading"):
            key = future2key[future]
            try:
                future.result()
                completed += 1
            except Exception as e:
                failed.append(key)
                logger.error(f"Download failed for {key}: {e}")

    logger.info(f"{completed} downloaded, {len(failed)} failed")
    return failed

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download preprints from bioRxiv S3 bucket")
    parser.add_argument("--n-files", type=int, default=10, help="Number of preprints to download")
    parser.add_argument("--output-dir", type=Path, default="papers", help="Output directory (within dataset dir)")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of parallel downloads")
    args = parser.parse_args()

    output_dir = DATASET_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    failed = download_papers(n=args.n_files, output_dir=output_dir, max_workers=args.max_workers)
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")

    