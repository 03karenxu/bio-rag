from wtpsplit import SaT
import json
import logging
import asyncio
from pathlib import Path

from utils.schemas import Paper
from utils.embedding import embed_with_retry
from preprocess.img_processing import get_image_paths
from preprocess.xml_parsing import parse_file
from config import BATCH_MAX_TOKENS, COHERE_BATCH_MAX
 
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------

SEGMENTER_MODEL = "sat-6l-sm"
PARAGRAPH_THRESHOLD = 0.5
MIN_CHUNK_TOKENS_THRESH = 20

class TextSegmenter:
    """
    Segments text into chunks using wtpsplit ("Segment Any Text"):
    https://github.com/segment-any-text/wtpsplit
    """
    _model: str | None = None
    _instance: SaT | None = None

    @classmethod
    def configure(cls, model: str) -> None:
        cls._model = model
        cls._instance = None

    @classmethod
    def instance(cls) -> SaT:
        if cls._model is None:
            raise RuntimeError("TextSegmenter model not configured")

        if cls._instance is None:
            cls._instance = SaT(
                cls._model,
                ort_providers=["CPUExecutionProvider"],
            )

        return cls._instance

    @classmethod
    def _normalize_segments(cls, segments: list[str]) -> None:
        """
        mutably collapse 2 segments when the boundary is a dash, sanitize
        """
        out = []
        i = 0
        n = len(segments)

        while i < n:
            s = segments[i]
            if s.endswith('-') and i + 1 < n:
                s = s[:-1] + segments[i + 1]
                i += 2
            else:
                i += 1

            out.append(s)

        segments.clear()
        segments.extend(out)

    @classmethod
    def create_segments(cls, text: str) -> list[str]:

        instance = cls.instance()
        segments = instance.split(text,
            strip_whitespace=True, 
            remove_whitespace_before_inference=True, 
            paragraph_threshold=PARAGRAPH_THRESHOLD)
        
        cls._normalize_segments(segments)

        return segments

TextSegmenter.configure(model=SEGMENTER_MODEL)

# ------------------------------------------------------------------------------

async def process_paper(xml_path: Path,
                        out_path: Path,
                        embed_sem: asyncio.Semaphore | None,
                        process_sem: asyncio.Semaphore | None,
                        media_folder: Path | None = None) -> None:
    '''
    processes a single paper. reads the xml file and generates a preprocessed,
    embedded version at out_path.
    '''
    try:
        if out_path.exists():
            logger.info(f"Skipping {xml_path.stem}, already processed")
            return

        # async with process_sem:
        # paper = await asyncio.to_thread(parse_file, xml_path)
        paper = parse_file(xml_path)

        flat_text = paper.full_text()

        # chunk with SaT
        chunk_texts:list[str] = TextSegmenter.create_segments(flat_text)

        with open("test.json", "w") as f:
            json.dump({"chunks": chunk_texts}, f)

        # paper = await _embed_paper(paper, media_folder, embed_sem)

        # with open(out_path, "w") as f:
        #     f.write(paper.model_dump_json())
    except Exception as e:
        raise ValueError(f"{xml_path} - {e}")

# ------------------------------------------------------------------------------

# async def _embed_paper(paper: Paper, media_folder: Path | None, sem: asyncio.Semaphore) -> Paper:
#     batches = _make_batches(paper)
#     n = len(batches)

#     async def _embed_batch(i: int, batch: list) -> list:
#         to_embed = [_chunk_to_input(chunk, media_folder) for chunk in batch]
#         async with sem:
#             embeddings = await embed_with_retry(input_=to_embed)
#         logger.info(f"Done batch {i+1}/{n} ({len(batch)} items) of {paper.title}")
#         return embeddings

#     results = await asyncio.gather(*[_embed_batch(i, batch) for i, batch in enumerate(batches)])
#     for batch, embeddings in zip(batches, results):
#         for chunk, embedding in zip(batch, embeddings):
#             chunk.embedding = embedding
#     return paper

# def _chunk_to_input(chunk: Chunk, media_folder: Path | None) -> dict:
#     '''
#     converts a chunk to the format expected for Cohere embed v4
#     '''
#     if chunk.section:
#         text = f"Section: {chunk.section} Content: {chunk.text}"
#     else:
#         text = chunk.text
        
#     if not media_folder:
#         return {"content": [{"type": "text", "text": text}]}
    
#     content: list[dict] = []
#     last = 0
#     for m in MEDIA_MARKER.finditer(text):
#         before = text[last:m.start()].strip()
#         if before:
#             content.append({"type": "text", "text": before})
        
#         for path in get_image_paths(media_folder, m.group(1)):
#             ext = path.suffix.lstrip(".").lower()
#             if ext == "jpg":
#                 ext = "jpeg"
#             else:
#                 logger.warning(f"NOT JPG: {path}")
#             b64 = base64.b64encode(path.read_bytes()).decode()
#             content.append({
#                 "type": "image_url",
#                 "image_url": {"url": f"data:image/{ext};base64,{b64}"},
#             })

#         last = m.end()

#     after = text[last:].strip()
#     if after:
#         content.append({"type": "text", "text": after})

#     logger.debug(content)
    
#     return {"content": content}


# def _make_batches(paper: Paper, max_tokens: int = BATCH_MAX_TOKENS) -> list[list[Chunk]]:
#     '''
#     takes a paper and batches its chunks. enforces cohere's max batch size of 
#     96 items, and ensures that each batch has a token size <= max_tokens.
#     '''
#     batches, current, current_tokens = [], [], 0

#     for chunk in (paper.abstract + paper.body):
#         if current_tokens + chunk.n_tokens > max_tokens or len(current) >= COHERE_BATCH_MAX:
#             if current:
#                 batches.append(current)
#             current, current_tokens = [chunk], chunk.n_tokens
#         else:
#             current.append(chunk)
#             current_tokens += chunk.n_tokens

#     if current:
#         batches.append(current)

#     return batches

if __name__ == "__main__":
    paper = "/Users/karenxu/Documents/Code/USRA/datasets/papers/0c7e0602-7c45-1014-93c3-be5d83700d97/paper/720744.xml"
    paper = parse_file(Path(paper))
    flat_text = paper.full_text(with_tables=True, with_media=True)
    chunk_texts:list[str] = TextSegmenter.create_segments(flat_text)
    print("hello")
    with open("test.json", "w") as f:
        json.dump({"chunks": chunk_texts}, f)
    print("goodbye")