import dspy
import asyncio
import logging
import unicodedata
import numpy as np
from pathlib import Path
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio
from collections import Counter

from utils.embedding import embed
from utils.log import init_logging
from utils.lm import openrouter_config
from utils.rag import chunk_paper
from utils.paper_schema import Paper
from utils.visualization import visualize_graph
from config import TRIPLE_MODEL, DEDUP_SIM_THRESHOLD

logger = logging.getLogger(__name__)
openrouter_config(model_str=TRIPLE_MODEL)

## models ##

class Triple(BaseModel):
    subject: str
    predicate: str
    object: str

class Graph(BaseModel):
    entities: list[str]
    edges: list[str]
    triples: list[Triple]
    source_doc: str

## extract entities and relations ##

class _EntityExtractSignature(dspy.Signature):
    """
    Extract key entities from the source text. These can be either subjects or objects.

    Rules:
    - entities must be specific named entities rather than pronouns or vague references.
    - entities must be copied as they appear in the text — do not normalize or expand.
    - be thorough and accurate
    """

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="An array of entities extracted from the source text")

class _TripleExtractSignature(dspy.Signature):
    """
    Extract subject-predicate-object triples from the source text. Subjects and objects
    MUST appear in the entities list. 

    Rules:
    - subjects/objects must appear exactly as they do in the entities list, preserving spelling, capitalization, etc.
    - predicate must be a concise, active verb phrase describing the relationship
      (e.g. "increases risk of", "is a type of", "requires", "is associated with").
    - do not infer relationships not explicitly stated in the text.
    - be thorough and accurate

    Return a dict array under 'triples' like:
    [
      {"subject": "...", "predicate": "...", "object": "..."},
      ...
    ]
    """
    source_text: str = dspy.InputField()
    entities: list[str] = dspy.InputField()
    triples: list[dict] = dspy.OutputField(
        desc="List of {subject, predicate, object} triple dicts extracted from the source text."
    )

_entity_extractor = dspy.Predict(_EntityExtractSignature)
_triple_extractor  = dspy.Predict(_TripleExtractSignature)

async def extract_entities_relations(
    chunks: list[str],
) -> tuple[list[str], list[Triple]]:
    all_entities: list[str] = []
    all_triples: list[Triple] = []

    async def task(sem: asyncio.Semaphore, chunk: str) -> None:
        async with sem:
            entities = (await _entity_extractor.acall(source_text=chunk)).entities or []
            if not entities:
                logger.warning("No entities returned for chunk")
            triples = (await _triple_extractor.acall(
                source_text=chunk, entities=entities
            )).triples or []

        for triple in triples:
            if not isinstance(triple, dict):
                continue
            try:
                all_triples.append(Triple(**triple))
            except Exception as e:
                logger.warning(f"Skipping invalid triple {triple}: {e}")

        all_entities.extend(entities)

    sem = asyncio.Semaphore(3)
    await tqdm_asyncio.gather(*[task(sem, chunk) for chunk in chunks])

    return all_entities, all_triples

## deduplication ##

class _DeduplicateSignature(dspy.Signature):
    """
    Identify equivalent items in a list and group them together with a canonical representative.
    
    These are items that refer to the same underlying entity or concept, including:
    - Acronyms and their expansions (e.g. "RAG" and "Retrieval Augmented Generation")
    - Plural and singular forms (e.g. "neuron" and "neurons")
    - Case variations (e.g. "BERT" and "Bert")
    - Abbreviations and shorthand (e.g. "Fig." and "Figure")
    - Tense variations (e.g. "inhibits" and "inhibited")
    - Stem variations (e.g. "activation" and "activate")
    - Semantically equivalent phrases (e.g. "is a type of" and "is a kind of")

    Rules:
    - Only group items that clearly refer to the same entity — do not over-merge.
    - Do not group items that are merely related or similar in topic.
    - The canonical form should be the most complete, human-readable representation.
    - Items with no duplicates should not appear in the output.
    """

    items: list[str] = dspy.InputField(desc="List of items that may contain duplicates.")
    duplicate_groups: list[dict] = dspy.OutputField(
        desc=(
            "List of duplicate groups. Each group is a dict with keys: "
            "'canonical' (the best representative string for the group) and "
            "'duplicates' (list of all items in the group including the canonical). "
            "Only include groups with more than one item. "
            "Example: [{'canonical': 'Retrieval Augmented Generation', 'duplicates': ['Retrieval Augmented Generation', 'RAG', 'retrieval-augmented generation']}]"
        )
    )

_deduplicator = dspy.Predict(_DeduplicateSignature)

def _normalize(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", s).split())

def _connected_components(vecs: np.ndarray, threshold: float) -> list[list[int]]:
    norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = norm @ norm.T
    n = len(vecs)
    parent = list(range(n))
    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    iu, ju = np.triu_indices(n, 1)
    mask = sim[iu, ju] >= threshold
    for i, j in zip(iu[mask], ju[mask]):
        parent[find(int(i))] = find(int(j))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())

async def _dedupe(items: list[str]) -> dict[str, str]:
    '''
    performs deduplication in 2 passes:
    1. exact match deduplication
    2. llm-based deduplication (within clusters)
    '''
    counts = Counter(_normalize(i) for i in items)
    total = len(items)

    # exact match deduplication
    canon_of_key: dict[str, str] = {}
    exact_groups: dict[str, list[str]] = {}
    for normalized in counts:
        key = normalized.casefold()
        cur = canon_of_key.get(key)
        if cur is None or (counts[normalized], len(normalized)) > (counts[cur], len(cur)):
            canon_of_key[key] = normalized
        exact_groups.setdefault(canon_of_key[key], []).append(normalized)
    survivors = list(dict.fromkeys(canon_of_key.values()))
    logger.info(f"[dedup] {total} items -> {len(survivors)} survivors after exact-match collapse")

    for canon, group in exact_groups.items():
        if len(group) > 1:
            logger.info("[dedup] exact collapse: %r <- %r", canon, [g for g in group if g != canon])

    # llm deduplication: embed -> cluster -> llm
    variant2canon = {v: v for v in survivors}
    n_clusters = 0
    if len(survivors) >= 2:
        vecs = np.array(await embed(survivors))
        for idxs in _connected_components(vecs, DEDUP_SIM_THRESHOLD):
            if len(idxs) < 2:
                continue
            n_clusters += 1
            group_items = [survivors[i] for i in idxs]
            logger.info(f"[dedup] similarity cluster #{n_clusters} ({DEDUP_SIM_THRESHOLD}): {group_items}")
            dup_groups = (await _deduplicator.acall(items=group_items)).duplicate_groups or []
            for g in dup_groups:
                canon = g.get("canonical")
                if canon not in group_items:
                    logger.warning(f"[dedup] LLM returned unknown canonical {canon}, skipping")
                    continue
                dups = [d for d in g["duplicates"] if d != canon]
                if dups:
                    logger.info(f"[dedup] LLM collapse: {canon} <- {dups}")
                for d in g["duplicates"]:
                    variant2canon[d] = canon
            if not dup_groups:
                logger.info("[dedup] cluster left unmerged by LLM")

    return {norm: variant2canon[canon_of_key[norm.casefold()]] for norm in counts}

async def deduplicate_graph(graph: Graph) -> Graph:
    '''
    deduplicates graph edges, entities, and triples
    '''
    entity_strs = [t.subject for t in graph.triples] + [t.object for t in graph.triples]
    logger.info("Deduplicating entities...")
    entity_map = await _dedupe(entity_strs)
    logger.info("Deduplicating edges...")
    edge_map = await _dedupe([t.predicate for t in graph.triples])

    seen: set[tuple[str, str, str]] = set()
    triples: list[Triple] = []
    n_dup_triples = 0
    for t in graph.triples:
        triple = Triple(
            subject=entity_map[_normalize(t.subject)],
            predicate=edge_map[_normalize(t.predicate)],
            object=entity_map[_normalize(t.object)],
        )
        key = (triple.subject, triple.predicate, triple.object)
        if key in seen:
            n_dup_triples += 1
            continue
        seen.add(key)
        triples.append(triple)

    if n_dup_triples:
        logger.info(f"Collapsed {n_dup_triples} duplicate triples after canonicalization")

    return Graph(
        source_doc=graph.source_doc,
        entities=list({t.subject for t in triples} | {t.object for t in triples}),
        edges=list({t.predicate for t in triples}),
        triples=triples,
    )

## main ##

async def generate_graph(source_file: Path, cache_path: Path | None = None) -> Graph:
    '''
    generates a Graph from a Paper (saved to source_file)
    '''
    if cache_path and cache_path.exists():
        with open(cache_path, "r") as f:
            graph = Graph.model_validate_json(f.read())
            visualize_graph(graph)
            return graph
        
    with open(source_file, "r") as f:
        paper = Paper.model_validate_json(f.read())

    chunks = chunk_paper(paper)
    logger.info("Extracting entities and relations...")
    entities, triples = await extract_entities_relations(chunks)

    graph = Graph(
        source_doc=source_file.name,
        entities=entities,
        edges=list({t.predicate for t in triples}),
        triples=triples,
    )

    logger.info("Deduplicating graph...")
    graph = await deduplicate_graph(graph)

    with open(cache_path, "w") as f:
        f.write(graph.model_dump_json())

    visualize_graph(graph)

    return graph

if __name__ == "__main__":
    from config import CACHE_DIR
    init_logging()
    test_file = Path("/Users/karenxu/Documents/Code/USRA/datasets/papers_test_json/0c7e0602-7c45-1014-93c3-be5d83700d97/paper/720744.json")
    asyncio.run(generate_graph(test_file, CACHE_DIR / "graphs" / test_file.name))
