# preprocess.py
#
# May 6th, 2026
#
# processes the evaluation dataset into metadata + chunks + embeddings

from __future__ import annotations

import logging
import asyncio
import argparse
import tiktoken
from uuid import uuid4
from pathlib import Path
from nltk import sent_tokenize
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio
import xml.etree.ElementTree as ET

from utils.logging import init_logging
from utils.embed import embed_with_retry
from config import DATASET_DIR, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE, CACHE_DIR, MAX_CONCURRENT_EMBED

logger = logging.getLogger(__name__)

TT = tiktoken.get_encoding("cl100k_base")

class Paper(BaseModel):
    title: str
    doi: str
    abstract: list[Chunk] # will only have more than 1 element if abstract has more than max tokens
    keywords: list[str]
    authors: list[str]
    date: str
    categories: list[str]
    body: list[Chunk]

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    n_tokens: int
    section: str
    text: str
    img_path: str | None # if Chunk represents a figure, store path to img
    embedding: list[float] | None = None


def _get_all_text(element, path: str) -> str:
    '''
    gets all the text inside an element, including the text inside child elements
    '''
    e = element.find(path)
    if e is None:
        return ""
    return "".join(e.itertext()).strip()


def _make_chunk(text: str, section: str, img_path: str | None = None) -> list[Chunk]:
        '''
        creates a Chunk from a paragraph. if the number of tokens in the paragraph is larger than
        MAX_CHUNK_SIZE, it gets split in half. assumes that paragraphs will not
        be more than double MAX_CHUNK_SIZE.
        '''
        n_tokens = len(TT.encode(text))
        if n_tokens > MAX_CHUNK_SIZE:
            # split in half semantically (at sentence boundaries)
            sentences = sent_tokenize(text)
            mid = len(sentences) // 2
            halves = [" ".join(sentences[:mid]), " ".join(sentences[mid:])]
            return [
                Chunk(n_tokens=len(TT.encode(h)), section=f"{section} [{i+1}/2]", text=h, img_path=img_path)
                for i, h in enumerate(halves)
            ]
        return [Chunk(n_tokens=n_tokens, section=section, text=text, img_path=img_path)]


def _process_body(root: ET.element) -> list[Chunk]:
    '''
    breaks the body of a paper into chunks. each paragraph/figure becomes a
    chunk. if the chunk size is greater than MAX_CHUNK_SIZE, it is split.
    '''

    def process_element(child, section: str) -> None:
        if child.tag == "fig":
            # figure
            label = "".join(child.find("label").itertext()).strip() if child.find("label") is not None else ""
            caption_el = child.find(".//caption")
            caption = "".join(caption_el.itertext()).strip() if caption_el is not None else ""
            graphic = child.find("graphic")
            img_path = graphic.get("{http://www.w3.org/1999/xlink}href") if graphic is not None else None
            text = f"{label} {caption}".strip()
            if text:
                chunks.extend(_make_chunk(text, section, img_path=img_path))
        else:
            # regular paragraph
            text = "".join(child.itertext()).strip()
            if text and len(TT.encode(text)) > MIN_CHUNK_SIZE:
                chunks.extend(_make_chunk(text, section))

    def process_section(sec, parent_title: str | None = None) -> None:
        title = "".join(sec.find("title").itertext()).strip() if sec.find("title") is not None else "Unknown"
        section_label = f"{parent_title} > {title}" if parent_title else title

        for child in sec:
            if child.tag == "sec":
                # subsections
                process_section(child, parent_title=section_label)
            elif child.tag != "title":
                process_element(child, section_label)

    chunks = []

    for section in root.findall(".//body/sec"):
        process_section(section)

    return chunks


def parse_paper(xml_path: Path) -> Paper:
    '''
    unpacks relevant information from a paper's xml file into a Paper model
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    def get_date(date_type: str) -> str:
        date = root.find(f".//history/date[@date-type='{date_type}']")
        if date is None:
            return ""
        year = date.findtext("year", default="")
        month = date.findtext("month", default="").zfill(2)
        day = date.findtext("day", default="").zfill(2)
        return f"{year}-{month}-{day}"

    def get_abstract() -> list[Chunk]:
        abstract = root.find(".//abstract")
        if abstract is None:
            return ""
        texts = []
        for child in abstract:
            if child.tag != "title":
                texts.extend(child.itertext())
        full_text = "".join(texts).strip()
        return _make_chunk(text=full_text, section="Abstract")

    return Paper(
        title=_get_all_text(root, ".//article-title"),
        doi=_get_all_text(root, ".//article-id[@pub-id-type='doi']"),
        abstract=get_abstract(),
        keywords=["".join(kwd.itertext()).strip() for kwd in root.findall(".//kwd") if "".join(kwd.itertext()).strip()],
        authors=[
            f"{_get_all_text(c, 'name/surname')}, {_get_all_text(c, 'name/given-names')}".strip(", ")
            for c in root.findall(".//contrib[@contrib-type='author']")
        ],
        date=get_date("accepted") or get_date("received"),
        categories=[s.text for s in root.findall(".//subj-group/subject") if s.text],
        body=_process_body(root)
    )


async def _embed_paper(paper: Paper) -> Paper:

    def make_batches(max_tokens: int = 7000) -> list[list[Chunk]]:
        batches, current, current_tokens = [], [], 0
        
        for chunk in (paper.abstract + paper.body):
            if current_tokens + chunk.n_tokens > max_tokens:
                if current:
                    batches.append(current)
                current, current_tokens = [chunk], chunk.n_tokens
            else:
                current.append(chunk)
                current_tokens += chunk.n_tokens
        
        if current:
            batches.append(current)
        
        return batches
    
    batches = make_batches()
    for batch in batches:
        to_embed = [ f"Section: {chunk.section} Content: {chunk.text}" for chunk in batch ]
        embeddings = await embed_with_retry(to_embed=to_embed)
        for i, chunk in enumerate(batch):
            chunk.embedding = embeddings[i]

    return paper


async def _process_single_paper(paper_dir: Path, out_dir: Path, sem: asyncio.Semaphore) -> None:
    xml_files = list(paper_dir.glob("*.xml"))
    if len(xml_files) > 1:
        logger.warning(f"Mutliple XML files found in {paper_dir.name}, using first")
    elif len(xml_files) == 0:
        raise ValueError(f"No XML found in {paper_dir.name}")
    async with sem:
        paper: Paper = parse_paper(xml_files[0])
        paper = await _embed_paper(paper)
    out_path = out_dir / f"{paper_dir.name}.json"
    with open(out_path, "w") as f:
        f.write(paper.model_dump_json())


async def process_dataset(dataset: Path, out_dir: Path, max_concurrent: int = MAX_CONCURRENT_EMBED) -> None:
    '''
    generates preprocessed cache files for the evaluation dataset.
    '''
    sem = asyncio.Semaphore(max_concurrent)
    paper_dirs = list(dataset.iterdir())
    tasks = [_process_single_paper(d, out_dir, sem) for d in paper_dirs]
    results = await tqdm_asyncio.gather(*tasks)
    return [p for p in results if p is not None]
        

if __name__ == "__main__":
    init_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="papers_test", help="The name of the folder to process")
    parser.add_argument("--out-dir", type=Path, default=CACHE_DIR, help="The name of the folder where processed files should be dumped")
    args = parser.parse_args()

    in_dir = DATASET_DIR / args.dataset
    if not in_dir.exists():
        raise FileNotFoundError(f"{in_dir} does not exist")
    
    Path(args.out_dir).mkdir(exist_ok=True)

    logger.info(f"Processing {in_dir}, dumping to {args.out_dir}")
    asyncio.run(process_dataset(in_dir, args.out_dir))