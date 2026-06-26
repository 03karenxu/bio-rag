import math
import dspy
import uuid
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from itertools import batched, product
from sklearn.mixture import GaussianMixture

from config import TRIPLE_MODEL
from utils.graph import triple_extractor
from utils.embedding import embed
from utils.lm import openrouter_config
from grade.schema import Claim, Triple, EntityEquivalence

logger = logging.getLogger(__name__)

openrouter_config(model_str=TRIPLE_MODEL)

class _EntityEquivalenceSignature(dspy.Signature):
    """
    Identify equivalent entity strings within a set of entities from a knowledge graph.

    Two entities are EXACTLY equivalent if they refer to the same concept regardless of context:
    - Abbreviations and their expansions: "CIEDs" and "cardiac implantable electronic devices"
    - Spelling variants or hyphenation differences
    - Capitalisation differences: "Heart Failure" and "heart failure"
    - Synoynms or near synonyms: eg. "formation" and "development"

    Two entities are CONTEXTUALLY equivalent if they refer to the same concept only within
    this specific domain context, but would not be interchangeable in general:
    - Domain synonyms: "leadless pacemakers" and "miniaturized intracardiac devices"
    - Partial references to the same entity: "pocket complications" and "pocket-related complications"

    IMPORTANT:
    CONTEXTUAL equivalence requires that the entities are interchangeable in the claims they
    appear in. Do NOT group entities that are merely related, co-occurring, or part of the
    same category.

    Rules:
    - Every tuple must contain at least two entity strings.
    - Entity strings must be copied exactly as they appear in the input — do not normalize or modify.
    - A single entity string cannot appear in both exact_equivalents and contextual_equivalents.
    - Exact equivalence takes priority over contextual equivalence.
    - Groups are transitive: if A=B and B=C, return (A, B, C) not (A, B) and (B, C).
    - If no equivalences exist, return empty lists.
    """

    entities: list[str] = dspy.InputField(
        desc=(
            "A list of entity strings extracted from knowledge graph triples within a semantic cluster. "
            "These are surface forms of subjects and objects as they appeared in atomic claims."
        )
    )
    claim_contexts: list[str] = dspy.InputField(
        desc=(
            "The atomic claim texts associated with each entity, providing domain context. "
            "Use these to judge whether entities refer to the same concept in this specific domain."
        )
    )
    equivalences: EntityEquivalence = dspy.OutputField(
        desc=(
            "exact_equivalents: list of tuples where all members are exactly equivalent entity strings. "
            "contextual_equivalents: list of tuples where all members are contextually equivalent entity strings. "
            "Entity strings must be copied exactly from the input entities list."
        )
    )

equivalence_detector = dspy.Predict(_EntityEquivalenceSignature)

async def get_triples(claims: list[Claim], cache_path: Path | None = None, batch_size: int = 20) -> list[Triple]:
    if cache_path.exists():
        logger.info(f"Reading triples from cache")
        with open(cache_path) as f:
            return [Triple.model_validate_json(line) for line in f if line.strip()]
        
    all_triples: list[Triple] = []

    for batch in tqdm(batched(claims, batch_size), total=math.ceil(len(claims)/batch_size)):
        batch = list(batch)
        texts = [c.text for c in batch]
        try:
            # there will be one list of triples per claim in the batch
            raw_output = triple_extractor(sentences=texts).triples
        except Exception as e:
            logger.warning(f"Triple extraction failed for batch of {len(batch)} claims: {e}")
            continue

        if len(raw_output) != len(batch):
            logger.warning( f"IO mismatch: expected {len(batch)} outer lists, got {len(raw_output)} — skipping batch")
            continue

        all_triples.extend(_parse_triples_from_batch(batch, raw_output))

    logger.info(f"Extracted {len(all_triples)} triples, detecting entity equivalences...")
    equivalences = await _detect_entity_equivalences(all_triples)
    logger.info(f"Resolving entity equivalences...")
    all_triples = _apply_equivalences(all_triples, equivalences)

    cache_path.parent.mkdir(exist_ok=True, parents=True)
    with open(cache_path, "w") as f:
        for triple in all_triples:
            f.write(triple.model_dump_json() + "\n")

    return all_triples

def _parse_triples_from_batch(batch: list[Claim], raw_triples: list[list[tuple[str, str, str]]]) -> list[Triple]:
    results: list[Triple] = []
    for claim, triples in zip(batch, raw_triples):
        for triple in triples:
            try:
                subject, predicate, obj = triple
            except (TypeError, ValueError):
                logger.warning(f"Malformed triple skipped (claim={claim.id!r}): {triple!r}")
                continue
            if not subject.strip() or not predicate.strip() or not obj.strip():
                logger.warning(f"Triple with empty component skipped: {triple!r}")
                continue
            results.append(
                Triple(
                    subject=subject.lower(),
                    predicate=predicate,
                    object=obj.lower(),
                    source_claim=claim.text,
                    source_sentence=claim.source_sentence,
                )
            )
    return results

## entity equivalence resolution ##

async def _embed_claims(claims: list[str], batch_size: int = 512) -> np.ndarray:
    all_embeddings: list[list[float]] = []
    for batch in batched(claims, batch_size):
        batch_embeddings = await embed(list(batch))
        all_embeddings.extend(batch_embeddings)
    return np.array(all_embeddings)

def _get_optimal_gmm(embeddings: np.ndarray, max_cluster: int = 50) -> int:
    n = len(embeddings)
    if n < 2:
        raise ValueError(f"Need at least 2 embeddings to fit a GMM, got {n}")

    k_max = min(max_cluster, n - 1)
    best_k = 1
    best_bic = np.inf

    for k in range(1, k_max + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42)
        gmm.fit(embeddings)
        bic = gmm.bic(embeddings)
        if bic < best_bic:
            best_bic = bic
            best_k = k

    return best_k

def _cluster_claims(claims: list[str], embeddings: np.ndarray, soft_threshold: float) -> dict[int, list[str]]:
    k = _get_optimal_gmm(embeddings)
    gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42)
    gmm.fit(embeddings)
    probs = gmm.predict_proba(embeddings)

    cluster_to_claims: dict[int, list[str]] = defaultdict(list)
    for i, claim in enumerate(claims):
        for cluster_idx, prob in enumerate(probs[i]):
            if prob >= soft_threshold:
                cluster_to_claims[cluster_idx].append(claim)

    return cluster_to_claims

async def _detect_entity_equivalences(triples: list[Triple], soft_threshold: float = 0.3) -> EntityEquivalence:
    seen_claims: set[str] = set()
    unique_claims: list[str] = []
    for triple in triples:
        if triple.source_claim not in seen_claims:
            seen_claims.add(triple.source_claim)
            unique_claims.append(triple.source_claim)

    if len(unique_claims) < 2:
        return EntityEquivalence()

    logger.info(f"Embedding {len(unique_claims)} unique claims...")
    embeddings = await _embed_claims(unique_claims)

    logger.info("Clustering claims with GMM...")
    cluster_to_claims = _cluster_claims(unique_claims, embeddings, soft_threshold)

    # build claim -> entities index for passing context to equivalence detector
    claim_to_entities: dict[str, set[str]] = defaultdict(set)
    for triple in triples:
        claim_to_entities[triple.source_claim].add(triple.subject)
        claim_to_entities[triple.source_claim].add(triple.object)

    all_exact: list[tuple[str, ...]] = []
    all_contextual: list[tuple[str, ...]] = []
    seen_exact: set[frozenset] = set()
    seen_contextual: set[frozenset] = set()

    for cluster_idx, cluster_claims in cluster_to_claims.items():
        # collect unique entities mentioned in this cluster's claims
        cluster_entities = list({e for c in cluster_claims for e in claim_to_entities[c]})
        if len(cluster_entities) < 2:
            continue
        try:
            claim_contexts = [" | ".join(c for c in cluster_claims if e in claim_to_entities[c]) for e in cluster_entities]
            result = equivalence_detector(entities=cluster_entities, claim_contexts=claim_contexts).equivalences
            equiv = (
                EntityEquivalence.model_validate(result)
                if isinstance(result, dict)
                else EntityEquivalence.model_validate(result.__dict__)
            )
        except Exception as e:
            logger.warning(f"Equivalence detection failed for cluster {cluster_idx}: {e}")
            continue

        for group in equiv.exact_equivalents:
            key = frozenset(group)
            if key not in seen_exact:
                seen_exact.add(key)
                all_exact.append(group)

        for group in equiv.contextual_equivalents:
            key = frozenset(group)
            if key not in seen_contextual and key not in seen_exact:
                seen_contextual.add(key)
                all_contextual.append(group)

    return EntityEquivalence(
        exact_equivalents=all_exact,
        contextual_equivalents=all_contextual,
    )

def _apply_equivalences(triples: list[Triple], equivalences: EntityEquivalence) -> list[Triple]:
    triples = _resolve_exact_equivalents(triples, equivalences)
    triples = _propagate_edges(triples, equivalences)
    return triples

def _resolve_exact_equivalents(triples: list[Triple], equivalences: EntityEquivalence) -> list[Triple]:
    exact_lookup: dict[str, str] = {
        surface_form: max(group, key=len)
        for group in equivalences.exact_equivalents
        for surface_form in group
    }

    resolved = [
        triple.model_copy(update={
            "subject": exact_lookup.get(triple.subject, triple.subject),
            "object": exact_lookup.get(triple.object, triple.object),
        })
        for triple in triples
    ]

    return [t for t in resolved if t.subject != t.object]

def _propagate_edges(triples: list[Triple], equivalences: EntityEquivalence) -> list[Triple]:
    entity_to_group: dict[str, set[str]] = {
        entity: set(group)
        for group in equivalences.contextual_equivalents
        for entity in group
    }

    new_triples: list[Triple] = []
    for triple in triples:
        subj_group = entity_to_group.get(triple.subject, {triple.subject})
        obj_group = entity_to_group.get(triple.object, {triple.object})

        for new_subj, new_obj in product(subj_group, obj_group):
            if new_subj == new_obj:
                continue
            if new_subj == triple.subject and new_obj == triple.object:
                continue  # skip original
            new_triples.append(
                triple.model_copy(update={
                    "id": str(uuid.uuid4()),
                    "subject": new_subj,
                    "object": new_obj,
                })
            )

    return triples + new_triples