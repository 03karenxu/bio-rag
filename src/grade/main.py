import asyncio
import logging
from pathlib import Path
from utils.log import init_logging
from config import DATASET_DIR, CACHE_DIR
from utils.paper_schema import Paper
from grade.claims import extract_claims
from grade.triples import get_triples
from grade.qa import generate_queries
from grade.schema import Triple

logger = logging.getLogger(__name__)

async def process_paper(paper: Paper) -> list[Triple]:
    logger.info(f"Extracting claims from {paper.front.hash}...")
    claims = extract_claims(paper)
    if not claims:
        logger.warning(f"No claims extracted for {paper.front.hash}")
        return []
    logger.info(f"Extracted {len(claims)} claims from {paper.front.hash}")

    triple_cache_path = CACHE_DIR / "triples" / f"{paper.front.hash}.jsonl"
    logger.info(f"Extracting triples from {len(claims)} claims...")
    triples = await get_triples(claims, triple_cache_path)
    logger.info(f"Extracted {len(triples)} raw triples from {paper.front.hash}")

    qa_cache_path = CACHE_DIR / "qa" / f"{paper.front.hash}.json"
    logger.info("Generating QA from triples...")
    queries = generate_queries(triples, qa_cache_path)

    return queries

def collect_litrevs(in_dir: Path) -> list[Path]:
        litrevs = []
        for parent_dir in in_dir.iterdir():
            if not parent_dir.is_dir():
                continue
            litrev = next((parent_dir / "paper").glob("*.json"), None)
            if litrev is None:
                logger.warning(f"No paper JSON found under {parent_dir / 'paper'} — skipping")
                continue
            litrevs.append(litrev)
        return litrevs

async def _main() -> None:
    init_logging()

    lit_revs = collect_litrevs(DATASET_DIR / "papers_test_json")
    logger.info(f"Found {len(lit_revs)} lit reviews")

    for lit_rev in lit_revs:
        logger.info(f"Processing {lit_rev.name}...")
        paper = Paper.model_validate_json(lit_rev.read_text())
        await process_paper(paper)
        break

if __name__ == "__main__":
    asyncio.run(_main())