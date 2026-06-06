from __future__ import annotations

from uuid import uuid4
from datetime import date
from dataclasses import dataclass
from pydantic import BaseModel, Field

class TextSpan(BaseModel):
    # a section of continuous text. may contain table refs, figure refs, or in-text citations
    text: str
    table_ids: list[str] | None = None
    fig_ids: list[str] | None = None
    ref_ids: list[str] | None = None

class MediaSpan(BaseModel):
    # a media item
    media_id: str
    caption: str
    label: str
    name: str | None

class TableSpan(BaseModel):
    # a table converted to markdown text
    markdown: str | None
    label: str
    caption: str
    table_id: str

class ListSpan(BaseModel):
    list_items: list[ContentItem]

ContentItem = TextSpan | MediaSpan | TableSpan | ListSpan

# ------------------------------------------------------------------------------

class Section(BaseModel):
    header: str
    content: list[ContentItem | Section]

class Reference(BaseModel):
    ref_id: str
    authors: list[str] = []
    title: str = ""
    pub_type: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    raw_text: str = ""

# ------------------------------------------------------------------------------

class PaperFront(BaseModel):
    title: str
    doi: str
    abstract: Section
    keywords: list[str]
    authors: list[str]
    date: date | None
    categories: list[str]

class PaperBack(BaseModel):
    references: list[Reference] = []

class Paper(BaseModel):
    front: PaperFront
    back: PaperBack
    body: list[Section]
