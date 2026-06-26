from utils.paper_schema import Paper
from config import CHUNK_TOKEN_TARGET
from collections import deque
from pydantic import BaseModel, Field, field_serializer
import uuid

## chunking ##

class Chunk(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    text: str
    section_header: str
    source_doc: str
    embedding: list[float] | None = None

    @field_serializer("id")
    def serialize_id(self, v: uuid.UUID) -> str:
        return str(v)

def chunk_paper(
    paper: Paper, target_tokens: int = CHUNK_TOKEN_TARGET
) -> list[Chunk]:
    '''
    chunks a paper into chunks of size ~target_tokens, while preserving sentence boundaries
    '''
    # flatten sentences but keep each one attached to its section header
    sentences: deque[tuple[str, str]] = deque(
        (p.section_header, s.flat_text()) for p in paper.body for s in p.items
    )

    source_doc = paper.front.hash
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    buffer_section = ""
    while sentences:
        section, s = sentences.popleft()
        n_tokens = len(s.split())
        if buffer and abs(buffer_tokens + n_tokens - target_tokens) > abs(
            buffer_tokens - target_tokens
        ):
            chunks.append(
                Chunk(
                    text=" ".join(buffer),
                    section_header=buffer_section,
                    source_doc=source_doc,
                )
            )
            buffer, buffer_tokens = [], 0
        if not buffer:
            buffer_section = section
        buffer.append(s)
        buffer_tokens += n_tokens

    if buffer:
        chunks.append(
            Chunk(
                text=" ".join(buffer),
                section_header=buffer_section,
                source_doc=source_doc,
            )
        )

    return chunks
