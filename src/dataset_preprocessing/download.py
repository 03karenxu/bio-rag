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

from utils.log import init_logging
from config import DATASET_DIR, COHERE_COMPATIBLE_FORMATS, LOG_DIR
from utils.image_handling import save_as_png, resize_and_save

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


def download_paper(key: str, files: dict[str, bytes], output_dir: Path) -> None:
    '''
    saves a paper's content files to disk under output_dir/<stem>/.
    '''
    key_dir = output_dir / Path(key).stem
    if key_dir.exists():
        logger.warning(f"Already downloaded content for {key}")
        return
    key_dir.mkdir(parents=True)
    for filename, content in files.items():
        outpath = key_dir / filename
        if Path(filename).stem.isdigit():
            outpath.write_bytes(content)
        elif Path(filename).suffix.lower() in COHERE_COMPATIBLE_FORMATS:
            # if main file or is already compatible, save as-is
            resize_and_save(outpath, content)
        elif Path(filename).suffix:
            try:
                save_as_png(filename, content, key_dir)
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
    init_logging(LOG_DIR/"download.log")
    parser = argparse.ArgumentParser(description="Download preprints from bioRxiv S3 bucket")
    parser.add_argument("--n-files", type=int, default=10, help="Number of preprints to download")
    parser.add_argument("--output-dir", type=Path, default="papers", help="Output directory (within dataset dir)")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of parallel s3 fetches")
    args = parser.parse_args()

    output_dir = DATASET_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = download_papers(args.n_files, output_dir, max_workers=args.max_workers)

    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")

    