import dspy
import math
import logging
from tqdm import tqdm
from itertools import batched

from config import CACHE_DIR, CLAIM_MODEL
from utils.lm import configure_lm
from grade.schema import Claim
from preprocess.xml_parsers.schema import Section, Paper

logger = logging.getLogger(__name__)

configure_lm(model_str=CLAIM_MODEL)

TARGET_SECTIONS = {"discussion", "conclusion"}

class _ClaimGenSignature(dspy.Signature):
    """
    Extract atomic, self-contained factual claims from a focal sentence.
    A claim is a single, direct assertion grounded entirely in the focal sentence.

    Rules:
    - State claims as direct assertions. Never wrap in attribution: eg. "Tissue adhesives reduce infection risk", not "The study found that tissue adhesives reduce infection risk."
    - One claim per relationship. Decompose compound sentences: eg. "X causes A and B" → two claims.
    - Resolve all ambiguous references: replace "it", "they", "these", "the procedure" with the full entity name.
    - Paraphrase faithfully — preserve hedging and qualifiers ("may", "is associated with", "was not significant"). Do not strengthen or soften.
    - Do not add information not present in the focal sentence. Do not infer.
    - If no factual claim can be formed, return an empty list.
    """

    context: str = dspy.InputField(
        desc=(
            "Surrounding text from the document. Use only to resolve "
            "pronouns, abbreviations, acronyms, ellipsis, and other references "
            "needed to make claims self-contained. Do not extract claims from "
            "the context itself."
        )
    )

    focal_sentence: str = dspy.InputField(
        desc=(
            "The sentence to convert into atomic standalone claims. All claims "
            "must be grounded in this sentence. Use context only for reference "
            "resolution and disambiguation."
        )
    )

    claims: list[str] = dspy.OutputField(
        desc=(
            "A list of atomic standalone claims derived from the focal sentence.\n"
            "- Each claim must express exactly one relationship or assertion.\n"
            "- Decompose enumerations: one claim per enumerated item.\n"
            "- Each claim must be a complete declarative statement.\n"
            "- Resolve all pronouns, acronyms, and implicit references.\n"
            "- Preserve hedging, uncertainty, modality, and qualifiers.\n"
            "- Do not add information not stated or implied by the focal sentence.\n"
            "- Do not extract claims from the context.\n"
            "- If no factual claim can be formed, return an empty list."
        )
    )

class _ClaimConsistencySignature(dspy.Signature):
    """
    Assess whether each claim is strictly entailed by its corresponding source sentence.

    Return True only if every element of the claim is directly supported by the literal
    content of the sentence — no inference, elaboration, or background knowledge required.
    Return False if the claim adds, omits, contradicts, or distorts any information.

    Rules:
    - A claim that is more specific than the sentence should be False: the sentence
      must explicitly support the specific assertion, not merely be consistent with it.
    - A claim that weakens the sentence (e.g. adds "may" when the sentence is definitive)
      should be False.
    - Ignore differences in tone, style, and figurative language.
    - Ignore minor paraphrase as long as the meaning is preserved exactly.
    """

    sentences: list[str] = dspy.InputField(desc="List of sentences to evaluate")
    claims: list[str] = dspy.InputField(desc="List of claims, aligned by index with sentences")
    results: list[bool] = dspy.OutputField(desc="Consistency verdict per pair, in the same order as input")

_claim_generator = dspy.Predict(_ClaimGenSignature)
_claim_validator = dspy.Predict(_ClaimConsistencySignature)

def extract_claims(paper: Paper) -> list[Claim]:
    # check if cache exists
    cache_path = CACHE_DIR / "claims" / f"{paper.front.hash}.jsonl"
    if cache_path.exists():
        logger.info(f"Reading claims for {paper.front.hash} from cache")
        with open(cache_path) as f:
            return [Claim.model_validate_json(line) for line in f if line.strip()]

    logger.info(f"Generating claims for {paper.front.hash}")

    # only get sentences from target sections
    relevant = [s for s in paper.body if _is_claim_section(s)]
    all_sentences = [sent for section in relevant for sent in section.sentences()]
    all_claims: list[Claim] = []

    # generate claims from each sentence while writing to cache
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    with open(cache_path, "w") as f:
        for i, sentence in tqdm(enumerate(all_sentences), total=len(all_sentences)):
            window = all_sentences[max(0, i - 5) : i + 2]
            context = " ".join(s.text for s in window if s is not sentence)
            try:
                result = _claim_generator(context=context, focal_sentence=sentence.text).claims
                if isinstance(result, str): result = [result]
                for claim_text in result:
                    if not claim_text.strip():
                        logger.warning(f"No claim extracted from sentence {sentence.id!r}: {sentence.text!r}")
                        continue
                    claim = Claim(text=claim_text, source_sentence=sentence.text)
                    all_claims.append(claim)
                    f.write(claim.model_dump_json() + "\n")
            except Exception as e:
                logger.warning(f"Claim generation failed for sentence {sentence.id!r}: {e}")
                continue
    
    all_claims = _validate_claims(all_claims)
    return all_claims

def _validate_claims(claims: list[Claim], batch_size: int = 50) -> list[Claim]:
    verified: list[Claim] = []

    for batch in tqdm(batched(claims, batch_size), total=math.ceil(len(claims) / batch_size)):
        batch = list(batch)
        try:
            results = _claim_validator(
                sentences=[c.source_sentence for c in batch],
                claims=[c.text for c in batch],
            ).results
        except Exception as e:
            logger.warning(f"Consistency check failed for batch of {len(batch)} claims: {e}")
            continue

        if len(results) != len(batch):
            logger.warning(f"IO mismatch: expected {len(batch)} results, got {len(results)} — skipping batch")
            continue

        for claim, consistent in zip(batch, results):
            if consistent:
                verified.append(claim)
            else:
                logger.debug(f"Claim failed consistency check (source={claim.source_sentence!r}): {claim.text!r}")

    logger.info(f"Consistency check: {len(verified)}/{len(claims)} claims passed")
    return verified

def _is_claim_section(section: Section) -> bool:
    return any(kw in section.header.lower() for kw in TARGET_SECTIONS)
