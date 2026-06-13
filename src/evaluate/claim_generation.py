import re
import nltk
import json
import dspy
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Iterator
from preprocess.xml_parsers.jats import parse_file
from preprocess.xml_parsers.schema import Paper
from utils.logging import init_logging
from config import CLAIM_EXTRACTION_MODEL, DATASET_DIR
from tqdm.contrib.logging import logging_redirect_tqdm

CLAIM_EXTR_MODEL = dspy.LM(CLAIM_EXTRACTION_MODEL)

logger = logging.getLogger(__name__)
dspy.configure(lm=CLAIM_EXTR_MODEL)

def split_sentences(text: str) -> list[str]:
    """Split text into all sentences (cited and non-cited)."""
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def extract_cited_sentences(sentences: list[str]) -> set[str]:
    """Return the subset of sentences that contain citations."""
    citation_pattern = re.compile(
        r'et al\.'
        r'|\[\d+(?:[,–\-]\s*\d+)*\]'
    )
    return {s for s in sentences if citation_pattern.search(s)}


def sliding_window(
    all_sentences: list[str],
    cited_sentences: set[str],
    window_size: int = 5,
) -> Iterator[tuple[int, str, str]]:
    """
    Yields (center_index, focal_sentence, window_context) for each cited sentence.

    focal_sentence  — the cited sentence to extract claims from
    window_context  — surrounding sentences from the FULL sentence list, so the
                      referent is available even if it lives in a non-cited sentence.
    """
    for i, sentence in enumerate(all_sentences):
        if sentence not in cited_sentences:
            continue
        start = max(0, i - window_size // 2)
        end = min(len(all_sentences), i + window_size // 2 + 1)
        context = " ".join(all_sentences[start:end])
        yield i, sentence, context


def collect_litrevs(in_dir: Path) -> list[Path]:
    litrevs = []
    for parent_dir in in_dir.iterdir():
        if not parent_dir.is_dir():  # fix: skip stray files
            continue
        litrev = next((parent_dir / "paper").glob("*.xml"), None)
        assert litrev, f"No XML found under {parent_dir / 'paper'}"
        litrevs.append(litrev)
    return litrevs


def collect_full_text(lit_revs: list[Path]) -> Iterator[str]:
    """
    Collects the entire body of text (excluding abstract, supp. materials,
    and references) from a paper. Returns an iterator.
    """
    for f in lit_revs:
        paper = parse_file(f)
        full_text = " ".join(body_p.text.strip() for body_p in paper.body)
        yield full_text


def collect_target_sections(lit_revs: list[Path]) -> Iterator[str]:
    """
    Collects text ONLY from results or discussion sections.
    """
    for f in lit_revs:
        paper: Paper = parse_file(f)
        all_texts = []
        for body_p in paper.body:
            if not body_p.head_section:
                logger.warning(f"No section header for body paragraph {body_p.id} in {f}")
                continue
            if any(kw in body_p.head_section.lower() for kw in ("results", "discussion")):
                all_texts.append(body_p.text.strip())
        yield " ".join(all_texts)

class _CorefResolutionSignature(dspy.Signature):
    """Resolve all coreferences in the focal sentence using the surrounding context.

    Coreference resolution means replacing every pronoun and referential
    expression in the focal sentence with its explicit named referent:
    - Pronouns: "it", "they", "he", "she", "this", "that", "these", "those"
    - Nominal references: "the model", "the proposed method", "this approach",
      "the authors", "our framework", "the system"
    - Only resolve references that can be unambiguously identified in the context.
    - Do NOT change any other wording — preserve numbers, dates, and proper nouns exactly.
    - If a reference cannot be resolved from the context, leave it as-is."""

    context: str = dspy.InputField(
        desc=(
            "A window of surrounding sentences providing document context. "
            "Use this to identify what pronouns and referential expressions refer to. "
            "Do not extract claims from this field."
        )
    )
    focal_sentence: str = dspy.InputField(
        desc=(
            "The single sentence to resolve coreferences in. "
            "Only modify pronouns and referential expressions — do not paraphrase or rewrite."
        )
    )
    resolved_sentence: str = dspy.OutputField(
        desc=(
            "The focal sentence with all resolvable pronouns and referential expressions "
            "replaced by their explicit named referents. All other wording identical to input."
        )
    )


class _AtomicClaimSignature(dspy.Signature):
    """Decompose a sentence into atomic claims.

    An atomic claim is the smallest possible standalone factual assertion:
    - Exactly one subject and one predicate.
    - Self-contained: verifiable without reading any other claim.
    - No conjunctions bundling multiple facts ("and", "but", "while", "also").
    - No compound subjects or objects — split them into separate claims.
    - All pronouns replaced with their explicit referents.
    - Nested relative clauses ("who …", "which …") extracted as their own claims.
    - Each attribute of an event (time, place, manner, quantity) is its own claim.
    - Only what is explicitly stated — do NOT infer or interpret."""

    text: str = dspy.InputField(
        desc=(
            "The raw sentence to decompose. All coreferences should already be resolved. "
            "Preserve all proper nouns, numbers, and dates exactly as written."
        )
    )
    atomic_claims: list[str] = dspy.OutputField(
        desc=(
            "An exhaustive list of atomic claims extracted from `text`. "
            "Each string must: (1) express exactly one fact, (2) stand alone without "
            "context from any other claim, (3) use no pronouns — all entities named "
            "explicitly, (4) contain no conjunctions bundling two facts. "
            "Temporal, spatial, and quantitative attributes are separate items. "
            "Return only claims explicitly stated; never infer."
        )
    )


coref_resolver = dspy.Predict(_CorefResolutionSignature)
claim_extractor = dspy.Predict(_AtomicClaimSignature)
claim_extractor.demos = [
    dspy.Example(
        text=(
            "For instance, hyperglycemia and insulin resistance (IR) enhance hepatic "
            "de novo lipogenesis and increase adipose fatty acid metabolism, contributing "
            "to dyslipidemia in diabetes [1, 2]."
        ),
        atomic_claims=[
            "Hyperglycemia enhances hepatic de novo lipogenesis.",
            "Insulin resistance enhances hepatic de novo lipogenesis.",
            "Hyperglycemia increases adipose fatty acid metabolism.",
            "Insulin resistance increases adipose fatty acid metabolism.",
            "Enhanced hepatic de novo lipogenesis contributes to dyslipidemia in diabetes.",
            "Increased adipose fatty acid metabolism contributes to dyslipidemia in diabetes.",
        ],
    ).with_inputs("text")
]


def extract_litrev_claims(
    lit_rev_fp: Path,
    window_size: int = 5,
) -> list[str]:
    logger.info(f"Extracting atomic claims from {lit_rev_fp}...")
    all_atomic_claims = []

    for full_text in collect_full_text([lit_rev_fp]):
        all_sentences = split_sentences(full_text)
        cited = extract_cited_sentences(all_sentences)
        logger.info(f"{len(all_sentences)} total sentences, {len(cited)} cited.")

        windows = list(sliding_window(all_sentences, cited, window_size=window_size))

        with logging_redirect_tqdm():
            for i, focal_sentence, context in tqdm(windows, desc="Extracting claims", total=len(windows)):
                # Step 1: resolve coreferences using surrounding context
                resolved = coref_resolver(
                    context=context,
                    focal_sentence=focal_sentence,
                ).resolved_sentence

                # Step 2: extract atomic claims from resolved sentence
                claims = claim_extractor(text=resolved).atomic_claims
                logger.info(f"Sentence {i}: extracted {len(claims)} claims.")
                all_atomic_claims.append({"statement": focal_sentence, "claims": claims})

    logger.info(f"Found {len(all_atomic_claims)} total atomic claims in {lit_rev_fp}.")
    return all_atomic_claims


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_logging()
    lit_revs = collect_litrevs(DATASET_DIR / "lit_reviews_test")
    logger.info(f"Found {len(lit_revs)} lit reviews")

    atomic_claims = extract_litrev_claims(lit_revs[0])

    out_path = Path("test.json")
    with open(out_path, "w") as f:
        logger.info(f"Dumping claims to {out_path}...")
        json.dump({"paper": lit_revs[0].name, "content": atomic_claims}, f, indent=2)

    logger.info("All done.")