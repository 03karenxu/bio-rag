from __future__ import annotations

from uuid import uuid4
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

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    n_tokens: int
    section: str
    text: str
    embedding: list[float] | None = None