import uuid
from pydantic import BaseModel, Field

class Claim(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_sentence: str

class Triple(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str
    predicate: str
    object: str
    source_claim: str
    source_sentence: str

class EntityEquivalence(BaseModel):
    exact_equivalents: list[list[str]] = Field(default_factory=list)
    contextual_equivalents: list[list[str]] = Field(default_factory=list)

class Query(BaseModel):
    node_path: list[Triple] = Field(default_factory=list)
    n_hops: int
    query: str
    answer: str
    source_sentences: list[str] = Field(default_factory=list)