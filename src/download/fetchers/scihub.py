import logging
import requests
from bs4 import BeautifulSoup
from download.core import FetchedPaper
from download.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

_scihub_limiter = RateLimiter(rate=1, period=3.0)


class ScihubFetcher():
    '''
    fetches papers from sci-hub
    '''

    def __init__(self):
        self.base_url = "https://sci-hub.box"
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

    def fetch_by_doi(self, doi: str) -> FetchedPaper:
        '''
        fetches a single paper by doi
        '''
        safe_doi = doi.replace("/", "_")

        _scihub_limiter.wait()
        r = requests.get(f"{self.base_url}/{doi}", timeout=30, headers=self.headers)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, 'html.parser')
        pdf_url_elem = soup.find(attrs={"name": "citation_pdf_url"})
        if not pdf_url_elem:
            raise ValueError(f"Could not find pdf url at {self.base_url}/{doi}")

        pdf_url = pdf_url_elem.get("content")
        if not pdf_url.startswith("http"):
            pdf_url = f"{self.base_url}{pdf_url}"

        _scihub_limiter.wait()
        r = requests.get(pdf_url, timeout=30, headers=self.headers)
        r.raise_for_status()

        return FetchedPaper(identifier=doi, main_file=(f"{safe_doi}.pdf", r.content))