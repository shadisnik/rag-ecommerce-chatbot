from __future__ import annotations

import os
import re
from collections import defaultdict, deque
from typing import Any


COLBERT_MODEL_NAME = os.getenv(
    "COLBERT_MODEL",
    "colbert-ir/colbertv2.0",
)

_colbert_model = None


INTENT_PATTERNS = {
    "symptoms": (
        "symptom",
        "symptoms",
        "sign",
        "signs",
    ),
    "causes": (
        "cause",
        "causes",
        "caused by",
    ),
    "treatment": (
        "treatment",
        "treat",
        "therapy",
        "management",
    ),
    "diagnosis": (
        "diagnosis",
        "diagnose",
        "diagnosed",
        "testing",
        "tests",
    ),
    "prevention": (
        "prevention",
        "prevent",
        "lower my risk",
        "reduce the risk",
    ),
    "definition": (
        "what is",
        "what are",
        "definition",
    ),
}


TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "common",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "their",
    "this",
    "to",
    "what",
    "which",
    "with",
    "your",
}


INTENT_TOKENS = {
    "symptom",
    "symptoms",
    "sign",
    "signs",
    "cause",
    "causes",
    "caused",
    "treatment",
    "treat",
    "therapy",
    "management",
    "diagnosis",
    "diagnose",
    "diagnosed",
    "testing",
    "tests",
    "prevention",
    "prevent",
    "definition",
}

ENTITY_GENERIC_TOKENS = {
    "type",
    "disease",
    "disorder",
    "syndrome",
    "condition",
    "common",
    "guide",
    "information",
    "overview",
    "risk",
}


def get_colbert_model():
    """
    Load the ColBERT model once per application process.
    """
    global _colbert_model

    if _colbert_model is None:
        from ragatouille import RAGPretrainedModel

        print(f"Loading ColBERT reranker: {COLBERT_MODEL_NAME}")

        _colbert_model = RAGPretrainedModel.from_pretrained(
            COLBERT_MODEL_NAME
        )

    return _colbert_model


def extract_section(
    pattern: str,
    text: str,
) -> str:
    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return ""

    return re.sub(
        r"\s+",
        " ",
        match.group(1),
    ).strip()


def extract_medquad_question(content: str) -> str:
    return extract_section(
        r"Question:\s*(.*?)(?:\nAnswer:|\Z)",
        content,
    )


def extract_medquad_topic(content: str) -> str:
    return extract_section(
        r"Medical topic:\s*(.*?)(?:\nQuestion:|\Z)",
        content,
    )


def tokenize(text: str) -> set[str]:
    return set(
        re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
    )


def important_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if token not in TOKEN_STOPWORDS
    }


def detect_intent(text: str) -> str:
    text_lower = text.lower()

    # Definition must be checked last because many questions
    # begin with "what are" but request symptoms or causes.
    ordered_intents = (
        "symptoms",
        "causes",
        "treatment",
        "diagnosis",
        "prevention",
        "definition",
    )

    for intent in ordered_intents:
        if any(
            pattern in text_lower
            for pattern in INTENT_PATTERNS[intent]
        ):
            return intent

    return ""


def prepare_scoring_text(
    candidate: dict[str, Any],
) -> str:
    content = str(candidate.get("content", "")).strip()

    if candidate.get("document_source") != "MedQuAD":
        return content

    stored_question = extract_medquad_question(content)
    medical_topic = extract_medquad_topic(content)

    parts = []

    if stored_question:
        parts.append(f"Question: {stored_question}")

    if medical_topic:
        parts.append(f"Medical topic: {medical_topic}")

    if parts:
        return "\n".join(parts)

    return content

def calculate_medquad_alignment_bonus(
    query: str,
    candidate: dict[str, Any],
) -> float:
    """
    Combine semantic similarity with MedQuAD-specific checks for
    question intent, disease identity, and numerical subtype.
    """
    if candidate.get("document_source") != "MedQuAD":
        return 0.0

    content = str(candidate.get("content", ""))

    stored_question = extract_medquad_question(content)
    medical_topic = extract_medquad_topic(content)

    comparison_text = (
        f"{stored_question} {medical_topic}"
    ).strip()

    query_terms = important_tokens(query)
    candidate_terms = important_tokens(comparison_text)

    bonus = 0.0

    # General lexical overlap has a limited effect.
    if query_terms:
        overlap = len(
            query_terms.intersection(candidate_terms)
        )

        coverage = overlap / len(query_terms)
        bonus += coverage * 2.0

    # Match the requested question type.
    query_intent = detect_intent(query)
    candidate_intent = detect_intent(stored_question)

    if query_intent:
        if query_intent == candidate_intent:
            bonus += 4.0
        else:
            bonus -= 4.0

    # Disease or medical-entity terms must also match.
    query_entity_terms = {
        token
        for token in query_terms
        if token not in INTENT_TOKENS
        and token not in ENTITY_GENERIC_TOKENS
        and not token.isdigit()
    }

    candidate_entity_terms = {
        token
        for token in candidate_terms
        if token not in INTENT_TOKENS
        and token not in ENTITY_GENERIC_TOKENS
        and not token.isdigit()
    }

    if query_entity_terms:
        entity_overlap = query_entity_terms.intersection(
            candidate_entity_terms
        )

        if entity_overlap:
            entity_coverage = (
                len(entity_overlap)
                / len(query_entity_terms)
            )

            bonus += entity_coverage * 8.0
        else:
            bonus -= 10.0

    # Preserve exact subtype numbers, such as type 2 rather than type 1.
    query_numbers = set(
        re.findall(r"\b\d+\b", query)
    )

    candidate_numbers = set(
        re.findall(r"\b\d+\b", comparison_text)
    )

    if query_numbers:
        if query_numbers.issubset(candidate_numbers):
            bonus += 3.0
        else:
            bonus -= 6.0

        extra_numbers = candidate_numbers - query_numbers

        if extra_numbers:
            bonus -= 1.5 * len(extra_numbers)

    return bonus

def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Rerank retrieved candidates with ColBERT while preserving
    the original document metadata.
    """
    if not candidates or top_k <= 0:
        return []

    valid_candidates = [
        dict(candidate)
        for candidate in candidates
        if str(candidate.get("content", "")).strip()
    ]

    if not valid_candidates:
        return candidates[:top_k]

    # Prepare the text that ColBERT will actually score and remove
    # duplicates after this transformation. This is important for
    # MedQuAD, where different full records can produce the same
    # question/topic scoring text.
    unique_scoring_candidates = {}

    for candidate in valid_candidates:
        scoring_text = prepare_scoring_text(candidate).strip()

        if not scoring_text:
            continue

        existing = unique_scoring_candidates.get(scoring_text)

        if (
            existing is None
            or candidate.get("relevance_score", 0)
            > existing.get("relevance_score", 0)
        ):
            unique_scoring_candidates[scoring_text] = candidate

    scoring_documents = list(unique_scoring_candidates.keys())
    valid_candidates = list(unique_scoring_candidates.values())

    if not scoring_documents:
        return []

    candidate_queues = defaultdict(deque)

    for scoring_text, candidate in zip(
        scoring_documents,
        valid_candidates,
    ):
        candidate_queues[scoring_text].append(candidate)
    model = get_colbert_model()

    # Retrieve scores for every surviving candidate so the
    # MedQuAD alignment checks can be applied before final top-k.
    reranked_results = model.rerank(
        query=query,
        documents=scoring_documents,
        k=len(scoring_documents),
    )

    final_results = []

    for result in reranked_results:
        scoring_text = str(
            result.get("content", "")
        ).strip()

        if not candidate_queues[scoring_text]:
            continue

        candidate = candidate_queues[scoring_text].popleft()

        colbert_score = float(
            result.get("score", 0.0)
        )

        # Use the ColBERT score directly for final ranking.
        # Dataset-specific hand-written bonuses are disabled so that
        # reranking remains model-based and consistent across sources.
        alignment_bonus = 0.0

        candidate["colbert_score"] = colbert_score

        candidate["colbert_original_rank"] = (
            int(
                result.get(
                    "rank",
                    len(final_results),
                )
            )
            + 1
        )

        candidate["medquad_alignment_bonus"] = (
            alignment_bonus
        )

        candidate["final_rerank_score"] = (
            colbert_score + alignment_bonus
        )

        final_results.append(candidate)

    final_results.sort(
        key=lambda item: (
            item.get("final_rerank_score", 0.0),
            item.get("colbert_score", 0.0),
        ),
        reverse=True,
    )

    for final_rank, candidate in enumerate(
        final_results,
        start=1,
    ):
        candidate["colbert_rank"] = final_rank

    return final_results[:top_k]
