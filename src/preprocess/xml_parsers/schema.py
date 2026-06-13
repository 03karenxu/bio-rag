from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

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

@dataclass
class InlineRef:
    type: RefType
    target: str

@dataclass
class Sentence:
    id: str
    text: str
    refs: list[InlineRef] = field(default_factory=list)

    def flat_text(self) -> str:
        return self.text

@dataclass
class List:
    items: list[list[Sentence]]
    def flat_text(self) -> str:
        bullets = []
        for bullet in self.items:
            flat_bullet = " ".join([s.flat_text() for s in bullet])
            bullets.append(f"- {flat_bullet}")
        return "\n".join(bullets)

    def to_markdown(self, *args) -> str:
        bullets = []
        for bullet in self.items:
            flat_bullet = " ".join([s.flat_text() for s in bullet])
            bullets.append(f"* {flat_bullet}")
        return "\n".join(bullets)

@dataclass
class Paragraph:
    id: str
    sentences: list[Sentence | List]

    def flat_text(self) -> str:
        body = " ".join(s.flat_text() for s in self.sentences)
        return body

    def to_markdown(self, *args) -> str:
        body = " ".join(s.text for s in self.sentences)
        return body

@dataclass
class Section:
    id: str
    header: str
    content: list["Paragraph | Section | List"]

    def flat_text(self) -> str:
        parts = []
        if self.header:
            parts.append(self.header)
        parts.extend(p.flat_text() for p in self.content)
        return "\n\n".join(parts)

    def to_markdown(self, is_sub: bool = False) -> str:
        parts = []
        if self.header and not is_sub:
            parts.append(f"## {self.header}")
        elif self.header:
            parts.append(f"### {self.header}")
        parts.extend(p.to_markdown(True) for p in self.content)
        return "\n\n".join(parts)

@dataclass
class InlineTable:
    headers: list = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

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

@dataclass
class Media:
    id: str
    label: str
    caption: str
    filename: str = ""

    def flat_text(self) -> str:
        return " ".join(x for x in [self.label, self.caption] if x)

    def markdown(self) -> str:
        return f"**{self.label}**\n\n{self.caption}"

@dataclass
class Reference:
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

@dataclass
class Front:
    title: str
    authors: list[str]
    hash: str
    publication_date: str
    abstract: Section

    def flat_text(self) -> str:
        return f"TITLE: {self.title}\n" + f"AUTHORS: {", ".join(self.authors)}\n" + f"DATE: {self.publication_date}"

    def to_markdown(self) -> str:
        authors = ", ".join(self.authors)
        parts = [
            f"# {self.title}",
            f"**Authors:** {authors}",
            f"**Date:** {self.publication_date}",
            "",
            self.abstract.to_markdown(),
        ]
        return "\n\n".join(parts)

@dataclass
class Paper:
    front: Front
    body: list[Section]
    media: list[Media]
    inline_tables: list[InlineTable]
    references: list[Reference]

    def flat_text(self, with_headers: bool = False) -> str:
        parts = [self.front.flat_text()]
        parts.extend(s.flat_text(with_headers=with_headers) for s in self.body)
        return "\n\n".join(parts)

    def to_markdown(self) -> str:
        parts = [self.front.to_markdown()]

        parts.extend(s.to_markdown() for s in self.body)

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