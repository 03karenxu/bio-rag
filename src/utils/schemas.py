from __future__ import annotations

from uuid import uuid4
from pathlib import Path
from datetime import date
from pydantic import BaseModel, Field

class Paper(BaseModel):
    title: str
    doi: str
    abstract: list[Chunk]
    keywords: list[str]
    authors: list[str]
    date: date
    categories: list[str]
    body: list[Chunk]
    supp_info: list[Chunk]
    references: list[Reference] = []

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    n_tokens: int
    section: str
    text: str
    embedding: list[float] | None = None

class Reference(BaseModel):
    ref_id: str
    authors: list[str] = []
    title: str = ""
    pub_type: str = ""
    journal: str = ""
    year: int | None = None
    volume: str = ""
    pages: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    raw_text: str = ""