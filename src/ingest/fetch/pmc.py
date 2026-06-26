import urllib
import logging
import requests
from typing import Iterator
import xml.etree.ElementTree as ET
from ingest.schema import FetchedPaper
from ingest.fetch.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)
_pmc_limiter = RateLimiter(rate=3, period=1.5)

# ------------------------------------------------------------------------------

class PMCFetcher():
    '''
    fetches papers from pubmed central
    '''

    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def fetch_by_pmcid(self, pmcid: str) -> FetchedPaper:
        '''
        fetch single paper by pmcid
        '''

        xml_text = self._fetch_full_text(pmcid)

        if "<article" not in xml_text and "<pmc-articleset" not in xml_text:
            raise ValueError(f"No article content for PMC{pmcid}: {xml_text}")

        return FetchedPaper(
            identifier=f"PMC{pmcid}",
            main_file=(f"{pmcid}.xml", xml_text.encode())
        )

    def fetch_by_count(self, n: int, query: str | None = None) -> Iterator[FetchedPaper]:
        '''
        fetch n papers with optional query. returns an iterator
        '''
        logger.info(f"Searching for {n} articles...")
        retstart = 0
        batch_size = n * 2
        fetched = 0

        while fetched < n:
            pmcids, total = self._search_pmc(batch_size, retstart, query)

            for pmcid in pmcids:
                if fetched >= n:
                    break
                try:
                    yield self.fetch_by_pmcid(pmcid)
                    fetched += 1
                except Exception as e:
                    logger.warning(f"Failed to fetch PMC{pmcid}: {e}")

            retstart += len(pmcids)
            if not pmcids or retstart >= total:
                logger.warning(f"Exhausted all {total} PMC results after fetching {fetched}.")
                break
    
    # --------------------------------------------------------------------------
    
    def _search_pmc(self, retmax: int, retstart: int, query: str | None = None) -> tuple[list[str], int]:
        params = f"esearch.fcgi?db=pmc&retmax={retmax}&retstart={retstart}"
        if query:
            params += f"&term={urllib.parse.quote(query)}"
        _pmc_limiter.wait()
        r = requests.get(self.base_url + params)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        total = int(root.findtext("Count") or 0)
        ids = [elem.text for elem in root.findall(".//Id")]
        return ids, total

    def _fetch_full_text(self, pmcid: str) -> str:
        _pmc_limiter.wait()
        r = requests.get(self.base_url + f"efetch.fcgi?db=pmc&id={pmcid}")
        r.raise_for_status()
        return r.text