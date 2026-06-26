from __future__ import annotations

import xml.etree.ElementTree as ET
from enum import StrEnum
import uuid
from pydantic import BaseModel, Field

class MissingContentError(Exception):
    def __init__(self, element: str, e: ET.Element):
        super().__init__(f"Missing required element: {element}")

class RefType(StrEnum):
    BIBR = "bibr"
    FIG = "fig"
    TABLE = "table"

class PubType(StrEnum):
    JOURNAL = "journal"
    BOOK = "book"
    CHAPTER = "book-chapter"
    PATENT = "patent"
    OTHER = "other"

class InlineRef(BaseModel):
    type: RefType
    target: str

class Sentence(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    text: str
    refs: list[InlineRef] = Field(default_factory=list)

    def flat_text(self) -> str:
        return self.text

class Paragraph(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    section_header: str = ""
    items: list[Sentence]

    def flat_text(self) -> str:
        return " ".join(s.flat_text() for s in self.items)

    def to_markdown(self) -> str:
        return " ".join(s.flat_text() for s in self.items)

class InlineTable(BaseModel):
    id: str
    label: str
    caption: str
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def flat_text(self) -> str:
        return "\n".join([" | ".join(r) for r in self.rows])

    def to_markdown(self) -> str:
        lines = []

        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")

        for r in self.rows:
            lines.append("| " + " | ".join(r) + " |")

        return "\n".join(lines)

class Media(BaseModel):
    id: str
    label: str
    caption: str
    filename: str = ""

    def flat_text(self) -> str:
        return " ".join(x for x in [self.label, self.caption] if x)

    def markdown(self) -> str:
        return f"**{self.label}**\n\n{self.caption}"

class Reference(BaseModel):
    id: str
    title: str
    authors: str | list[str]
    pub_type: str
    source: str
    year: int | None
    volume: str
    page_start: str
    page_end: str
    doi: str = ""
    pmcid: str = ""

    def to_markdown(self) -> str:
        authors = (
            ", ".join(self.authors)
            if isinstance(self.authors, list)
            else self.authors
        )
        year = self.year if self.year is not None else "n.d."
        return f"- {authors} ({year}). *{self.title}*. {self.source}."

def _paragraphs_to_markdown(paragraphs: list[Paragraph]) -> str:
    parts: list[str] = []
    current_header = None
    for p in paragraphs:
        if p.section_header != current_header:
            current_header = p.section_header
            if current_header:
                parts.append(f"## {current_header}")
        parts.append(p.to_markdown())
    return "\n\n".join(parts)

class Front(BaseModel):
    title: str
    authors: list[str]
    hash: str
    publication_date: str
    abstract: list[Paragraph] = Field(default_factory=list)

    def flat_text(self) -> str:
        return (
            f"TITLE: {self.title}\n"
            + f"AUTHORS: {', '.join(self.authors)}\n"
            + f"DATE: {self.publication_date}"
        )

    def to_markdown(self) -> str:
        authors = ", ".join(self.authors)
        parts = [
            f"# {self.title}",
            f"**Authors:** {authors}",
            f"**Date:** {self.publication_date}",
        ]
        if self.abstract:
            parts.append(_paragraphs_to_markdown(self.abstract))
        return "\n\n".join(parts)

class Paper(BaseModel):
    front: Front
    body: list[Paragraph]
    media: list[Media]
    inline_tables: list[InlineTable]
    references: list[Reference]


    def flat_text(self) -> str:
        parts = [self.front.flat_text()]
        parts.extend(p.flat_text() for p in self.body)
        return "\n\n".join(parts)

    def to_markdown(self) -> str:
        parts = [self.front.to_markdown()]

        if self.body:
            parts.append(_paragraphs_to_markdown(self.body))

        if self.media:
            parts.append("## Figures")
            parts.extend(m.markdown() for m in self.media)

        if self.inline_tables:
            parts.append("## Tables")
            parts.extend(t.to_markdown() for t in self.inline_tables)

        if self.references:
            parts.append("## References")
            parts.extend(r.to_markdown() for r in self.references)

        return "\n\n".join(parts)

    def sentences(self) -> list[Sentence]:
        return [s for p in self.body for s in p.items]