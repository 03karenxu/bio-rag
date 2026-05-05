# download_pdfs.py
# 
# date: May 5th 2026
#
# fetches and downloads the first n papers in the most recent biorxiv s3 dump

import io
import boto3
import logging
import zipfile
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from paths import DATASET_DIR

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


def list_n_keys(n: int, prefix: str) -> list[str]:
    '''
    lists the first n keys (lexicographical order) in the specified folder
    '''

    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=prefix, RequestPayer="requester")
    keys = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".meca"):
                keys.append(obj["Key"])
                if len(keys) >= n:
                    return keys
    return keys


def download_single_pdf(key: str, output_dir: Path) -> None:
    '''
    downloads the pdf associated with the given key
    '''

    output_path = output_dir / (Path(key).stem + ".pdf")
    if output_path.exists():
        logger.warning(f"Already downloaded PDF for {key}")
        return
    
    obj = s3_client.get_object(Bucket=BUCKET, Key=key, RequestPayer="requester")
    data = obj["Body"].read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        pdfs = [f for f in z.namelist() if f.startswith("content/") and f.endswith(".pdf")]
        if not pdfs:
            logger.warning(f"No PDF found in {key}")
            return
        
        pdf_path = pdfs[0] # there should just be one
        with z.open(pdf_path) as pdf:
            output_path.write_bytes(pdf.read())
        
def download_n_pdfs(n: int, output_dir: str, max_workers: int = 10) -> list[str]:
    '''
    downloads the first n papers (as pdfs) from the latest biorxiv s3 dump.
    returns the papers where download failed
    '''
    keys = list_n_keys(n=n, prefix=get_latest_folder())

    completed = 0
    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future2key = {executor.submit(download_single_pdf, k, output_dir): k for k in keys}
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
    parser = argparse.ArgumentParser(description="Download preprint PDFs from bioRxiv S3 bucket")
    parser.add_argument("--n-files", type=int, default=100, help="Number of PDFs to download")
    parser.add_argument("--output-dir", type=Path, default=(DATASET_DIR / "PDFs"), help="Output directory")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of parallel downloads")
    args = parser.parse_args()

    failed = download_n_pdfs(n=args.n_files, output_dir=args.output_dir, max_workers=args.max_workers)
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")

    