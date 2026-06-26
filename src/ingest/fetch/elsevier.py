import os
import logging
import requests
from ingest.schema import FetchedPaper
from ingest.fetch.rate_limiter import RateLimiter
from dotenv import load_dotenv
from config import SRC, DATASET_DIR

logger = logging.getLogger(__name__)

_sd_limiter = RateLimiter(rate=1, period=60.0)
load_dotenv(SRC/".env")

_ELSEVIER_PREFIXES = ("10.1016/", "10.1006/", "10.1053/", "10.1054/")

class SDFetcher():
    '''
    fetches full-text XML by doi from elsevier sciencedirect api
    '''

    def __init__(self):
        self.base_url = "https://api.elsevier.com"
        self.api_key = os.getenv("ELSEVIER_KEY")
        if not self.api_key:
            raise ValueError("ELSEVIER_KEY not set")
    
    def fetch_by_doi(self, doi: str) -> FetchedPaper:
        '''
        fetches a single paper by doi as elsevier full-text xml
        '''
        if not doi.startswith(_ELSEVIER_PREFIXES):
            raise ValueError(f"Likely not available on Elsevier: {doi}")
        
        safe_doi = doi.replace("/", "_")

        _sd_limiter.wait()
        r = requests.get(
            f"{self.base_url}/content/article/doi/{doi}",
            headers={
                "X-ELS-APIKey": self.api_key,
                "Accept": "text/xml",
            },
            params={
                "view": "FULL"
            },
            timeout=30,
        )
        r.raise_for_status()

        return FetchedPaper(identifier=doi, main_file=(f"{safe_doi}.xml", r.content))
    
if __name__ == "__main__":
    doi = "10.1016/j.archoralbio.2010.06.016"
    sd = SDFetcher()
    paper = sd.fetch_by_doi(doi)
    xml_path = DATASET_DIR / f"{paper.main_file[0]}"
    xml_path.write_bytes(paper.main_file[1])
    print(f"Written to {xml_path}")
