from grade.mg_client import MemgraphClient
from grade.schema import Triple, Query
import logging
import json
import dspy
from tqdm import tqdm
from pathlib import Path

from config import CACHE_DIR, QA_GEN_MODEL
from utils.lm import configure_lm

logger = logging.getLogger(__name__)

configure_lm(model_str=QA_GEN_MODEL)

class _QAGenSignature(dspy.Signature):
    """
    Generate a multi-hop question and answer pair from a chain of knowledge graph triples.

    Each triple represents a directed relationship: subject → predicate → object.
    The triples form a connected path where the object of one triple is the subject of the next.
    Use the source_claim field of each triple as grounding evidence.

    Rules:
    - The question must require reasoning through ALL triples in order to arrive at the answer.
      A question answerable from a single triple is not acceptable.
    - The answer must be a specific, concrete entity or short phrase — the terminal node of the
      reasoning chain. Never an abstract concept like "risk factors" or "study limitations".
    - The question should not give away the answer — avoid phrasing that makes the answer
      obvious without traversing the full chain.
    - Ground the question and answer strictly in the provided triples and claims.
      Do not introduce external knowledge or assumptions.
    - Preserve hedging from the source claims ("may", "is associated with") when relevant.
    """

    triples: list[dict] = dspy.InputField(
        desc=(
            "An ordered list of triples forming a reasoning chain. Each triple has fields: "
            "subject, predicate, object (the relationship), source_claim (the atomic claim "
            "this triple was extracted from), and source_sentence (the original sentence). "
            "The object of each triple is the subject of the next."
        )
    )
    query: str = dspy.OutputField(
        desc=(
            "A natural language question that requires chaining through all provided triples "
            "to answer. The question should start from the subject of the first triple and "
            "lead to the object of the last triple."
        )
    )
    answer: str = dspy.OutputField(
        desc=(
            "The answer to the query — the object of the final triple in the chain. "
            "Must be a specific entity, term, or concise concept, not a vague or abstract phrase."
        )
    )

_qa_generator = dspy.Predict(_QAGenSignature)

MAX_PER_HOP = 10
def generate_queries(triples: list[Triple], qa_cache_path: Path) -> list[Query]:

    _get_paths(triples)

    queries = []
    in_dir = CACHE_DIR / "shortest_paths"
    for file in in_dir.iterdir():
        with open(file, "r") as f:
            data = json.load(f)

        for raw_path in tqdm(data["paths"][:MAX_PER_HOP]):
            sample_path = [Triple(**t) for t in raw_path]
            n_hops = len(sample_path)
            resp = _qa_generator(triples=raw_path)
            q = Query(
                node_path=sample_path,
                n_hops=n_hops,
                query=resp.query,
                answer=resp.answer,
                source_sentences=[t.source_sentence for t in sample_path],
            )
            queries.append(q)

    qa_cache_path.parent.mkdir(exist_ok=True, parents=True)
    with open(qa_cache_path, "w") as f:
        json.dump([q.model_dump() for q in queries], f, indent=2)

    return queries

def _get_paths(triples: list[Triple]) -> dict[int, list[list[Triple]]]:
    with MemgraphClient() as mg:
        mg.load_triples(triples)
        shortest_paths = mg.get_shortest_paths()
    out_dir = CACHE_DIR / "shortest_paths"
    out_dir.mkdir(exist_ok=True, parents=True)
    for n_hops, paths in shortest_paths.items():
        out_path = out_dir / f"{n_hops}hops.json"
        with open(out_path, "w") as f:
            json.dump({"paths": [[t.model_dump() for t in path] for path in paths]}, f, indent=2)
    return shortest_paths