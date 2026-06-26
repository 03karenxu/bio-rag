import io
import boto3
import logging
import zipfile
from tqdm import tqdm
from pathlib import Path
from typing import Iterator
from datetime import datetime
from ingest.schema import FetchedPaper
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------

class BiorxivFetcher():
    '''
    fetches papers from biorxiv's s3 bucket
    '''
    def __init__(self, folder: str | None = None):
        self.s3_bucket = "biorxiv-src-monthly"
        self.s3_client = boto3.client("s3", region_name="us-east-1")
        if not folder:
            folder = self._get_latest_folder()
        else:
            folder = f"Current_Content/{folder}"
        self.folder = folder

    def fetch_by_key(self, key: str) -> FetchedPaper:
        '''
        fetches a single .meca from s3 and returns its content files as a dict
        of {filename: bytes}.
        '''
        obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key, RequestPayer="requester")
        data = obj["Body"].read()
        files: dict[str, bytes] = {}
        main_xml = None
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for file_path in z.namelist():
                if file_path.startswith("content/"):
                    filename = Path(file_path).name
                    if Path(filename).stem.isdigit() and Path(filename).suffix == ".xml" and not main_xml:
                        main_xml = filename
                    with z.open(file_path) as f:
                        files[filename] = f.read()

        main_file: tuple[str, bytes] = (main_xml, files[main_xml])
        media_files: dict[str, bytes] = {k: v for k, v in files.items() if not Path(k).stem.isdigit()}
        
        return FetchedPaper(identifier=key, main_file=main_file, media_files=media_files)

    def fetch_by_count(self, n: int, max_workers: int = 10) -> Iterator[FetchedPaper]:
        '''
        fetches n papers, returns an iterator
        '''
        keys = self._list_recent_keys(n=n)
        count_failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future2key = {executor.submit(self.fetch_by_key, k): k for k in keys}
            for future in tqdm(as_completed(future2key), total=len(future2key)):
                key = future2key[future]
                try:
                    yield future.result()
                except Exception as e:
                    logger.error(f"Fetch failed for {key}: {e}")
                    count_failed += 1

        logger.info(f"Done fetching! ({count_failed} failed)")

    # --------------------------------------------------------------------------

    def _list_recent_keys(self, n: int) -> list[str]:
        '''
        gets the top n most recent keys in either the specified folder, or the
        most recent one if not given
        '''
        logger.info(f"Fetching from folder: {self.folder}")
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=self.s3_bucket,
            Prefix=self.folder,
            RequestPayer="requester"
        )
        
        all_keys = []
        for page in page_iterator:
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".meca"):
                    all_keys.append((obj["LastModified"], obj["Key"]))
        
        all_keys.sort(key=lambda x: x[0], reverse=True)
        
        return [key for _, key in all_keys[:n]]
    
    def _get_latest_folder(self) -> str:
        '''
        gets the latest folder in the biorxiv s3 bucket
        '''

        response = self.s3_client.list_objects_v2(
            Bucket=self.s3_bucket,
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
