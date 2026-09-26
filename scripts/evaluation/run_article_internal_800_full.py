#!/usr/bin/env python3
"""
Evaluate the final 800-case article internal holdout dataset.

This runner:
- calls app.agent.medical_workflow.run_medical_workflow directly;
- supports all four evaluation groups;
- preserves the raw workflow response and agent trace;
- measures routing, retrieval, verification, latency, and heuristic answer quality;
- writes one JSONL row after every case;
- safely resumes after interruption;
- creates CSV and aggregate JSON metrics.

Important:
The lexical answer score is an automatic heuristic, not a clinical expert judgment.
Use the saved predictions for later LLM-judge and/or human review if required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.medical_workflow import run_medical_workflow  # noqa: E402


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "article_internal_holdout_800_final.jsonl"
)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"

EXPECTED_GROUP_COUNTS = {
    "timeline_day_specific": 200,
    "timeline_summary": 200,
    "ehr_clinical_qa": 200,
    "pediatric_oncology_qa": 200,
}

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "by", "for", "from", "has", "have", "had", "in", "into", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "there",
    "these", "this", "those", "to", "was", "were", "will", "with",
}

SUBJECT_ID_KEYS = (
    "subject_id",
    "expected_subject_id",
    "patient_subject_id",
)

HADM_ID_KEYS = (
    "hadm_id",
    "expected_hadm_id",
    "admission_id",
)

EXPECTED_SOURCE_KEYS = (
    "expected_document_url",
    "source_file",
    "expected_source_file",
    "document_url",
    "expected_document_source",
)


def clean_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_space(value)).lower()
    text = (
        text.replace("≥", " greater than or equal to ")
        .replace("≤", " less than or equal to ")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("°c", " c ")
        .replace("°", " ")
        .replace("²", "2")
        .replace("³", "3")
    )
    text = re.sub(r"[^a-z0-9%+./<>=\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(value: Any) -> list[str]:
    return [
        token
        for token in normalize_text(value).split()
        if len(token) > 1 and token not in STOP_WORDS
    ]


def token_f1(prediction: Any, reference: Any) -> tuple[float, float, float]:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def sequence_similarity(prediction: Any, reference: Any) -> float:
    return SequenceMatcher(
        None,
        normalize_text(reference),
        normalize_text(prediction),
    ).ratio()


def extract_numbers(value: Any) -> list[str]:
    normalized = normalize_text(value)
    return re.findall(
        r"(?<![a-z])(?:\d+(?:\.\d+)?)(?:%|(?:\s*mg/m2)|(?:\s*c))?",
        normalized,
    )


def normalize_number_token(value: str) -> str:
    return re.sub(r"\s+", "", value.lower())


def number_coverage(prediction: Any, reference: Any) -> Optional[float]:
    expected = [
        normalize_number_token(item)
        for item in extract_numbers(reference)
    ]

    if not expected:
        return None

    predicted = {
        normalize_number_token(item)
        for item in extract_numbers(prediction)
    }

    matched = sum(item in predicted for item in expected)
    return matched / len(expected)


def answer_metrics(answer: str, expected_answer: Any) -> dict[str, Any]:
    expected = clean_space(expected_answer)

    if not expected:
        return {
            "answer_exact_match": None,
            "answer_token_precision": None,
            "answer_token_recall": None,
            "answer_token_f1": None,
            "answer_sequence_similarity": None,
            "answer_number_coverage": None,
            "heuristic_answer_score": None,
            "heuristic_answer_pass": None,
        }

    precision, recall, f1 = token_f1(answer, expected)
    sequence = sequence_similarity(answer, expected)
    numeric_coverage = number_coverage(answer, expected)

    # F1 is primary. Sequence similarity adds modest credit for near-verbatim
    # phrasing. Number coverage protects clinically important quantities.
    score = (0.80 * f1) + (0.20 * sequence)

    if numeric_coverage is not None:
        score *= 0.70 + (0.30 * numeric_coverage)

    exact = normalize_text(answer) == normalize_text(expected)

    # This is intentionally named a heuristic rather than semantic accuracy.
    passed = score >= 0.55 and (
        numeric_coverage is None or numeric_coverage >= 1.0
    )

    return {
        "answer_exact_match": exact,
        "answer_token_precision": round(precision, 6),
        "answer_token_recall": round(recall, 6),
        "answer_token_f1": round(f1, 6),
        "answer_sequence_similarity": round(sequence, 6),
        "answer_number_coverage": (
            round(numeric_coverage, 6)
            if numeric_coverage is not None
            else None
        ),
        "heuristic_answer_score": round(score, 6),
        "heuristic_answer_pass": passed,
    }



def _empty_answer_metrics() -> dict[str, Any]:
    return {
        "answer_exact_match": None,
        "answer_token_precision": None,
        "answer_token_recall": None,
        "answer_token_f1": None,
        "answer_sequence_similarity": None,
        "answer_number_coverage": None,
        "heuristic_answer_score": None,
        "heuristic_answer_pass": None,
    }


def _contains_normalized(answer: Any, expected: Any) -> bool:
    expected_text = normalize_text(expected)
    return bool(expected_text) and expected_text in normalize_text(answer)


def _timeline_day_metrics(
    case: dict[str, Any],
    answer: str,
) -> dict[str, Any]:
    expected = case.get("expected_answer")
    question_type = clean_space(case.get("question_type"))

    metrics = _empty_answer_metrics()
    score: float

    if question_type == "laboratory_results" and isinstance(expected, dict):
        expected_pairs = [
            (clean_space(key), clean_space(value))
            for key, value in expected.items()
            if clean_space(value)
        ]

        if not expected_pairs:
            return answer_metrics(answer, expected)

        answer_norm = normalize_text(answer)
        matches = 0

        aliases = {
            "wbc": ("wbc", "white blood cell"),
            "anc": ("anc", "absolute neutrophil count"),
            "hemoglobin": ("hemoglobin", "haemoglobin", "hgb"),
            "platelets": ("platelets", "platelet count"),
        }

        for key, value in expected_pairs:
            key_norm = normalize_text(key)
            key_aliases = aliases.get(key_norm, (key_norm,))
            key_found = any(
                normalize_text(alias) in answer_norm
                for alias in key_aliases
            )
            value_found = normalize_text(value) in answer_norm

            if key_found and value_found:
                matches += 1

        score = matches / len(expected_pairs)
        passed = matches == len(expected_pairs)

    elif question_type == "temperature":
        expected_numbers = extract_numbers(expected)
        answer_numbers = {
            normalize_number_token(item)
            for item in extract_numbers(answer)
        }

        if not expected_numbers:
            return answer_metrics(answer, expected)

        matched = sum(
            normalize_number_token(item) in answer_numbers
            for item in expected_numbers
        )
        score = matched / len(expected_numbers)
        passed = matched == len(expected_numbers)

    elif question_type == "symptoms":
        if isinstance(expected, list):
            expected_items = [
                clean_space(item)
                for item in expected
                if clean_space(item)
            ]
        elif isinstance(expected, dict):
            expected_items = [
                clean_space(value)
                for value in expected.values()
                if clean_space(value)
            ]
        else:
            expected_items = [
                item.strip()
                for item in re.split(r"[,;|]", clean_space(expected))
                if item.strip()
            ]

        if not expected_items:
            return answer_metrics(answer, expected)

        matches = sum(
            _contains_normalized(answer, item)
            for item in expected_items
        )
        score = matches / len(expected_items)
        passed = matches == len(expected_items)

    elif question_type == "risk_level":
        expected_text = clean_space(expected)
        score = 1.0 if _contains_normalized(answer, expected_text) else 0.0
        passed = score == 1.0

    elif question_type == "recommended_action":
        precision, recall, f1 = token_f1(answer, expected)
        score = recall
        passed = (
            _contains_normalized(answer, expected)
            or recall >= 0.70
        )
        metrics.update(
            {
                "answer_token_precision": round(precision, 6),
                "answer_token_recall": round(recall, 6),
                "answer_token_f1": round(f1, 6),
            }
        )

    else:
        return answer_metrics(answer, expected)

    metrics.update(
        {
            "answer_exact_match": normalize_text(answer)
            == normalize_text(expected),
            "heuristic_answer_score": round(score, 6),
            "heuristic_answer_pass": bool(passed),
        }
    )
    return metrics


def _parse_summary_risk_counts(expected: Any) -> dict[str, int]:
    text = clean_space(expected)
    counts: dict[str, int] = {}

    for risk in ("low", "medium", "high"):
        match = re.search(
            rf"\b{risk}\s*:\s*(\d+)\b",
            text,
            re.IGNORECASE,
        )
        if match:
            counts[risk] = int(match.group(1))

    return counts


def _timeline_summary_metrics(
    case: dict[str, Any],
    answer: str,
    retrieved_document_count: int,
) -> dict[str, Any]:
    metrics = _empty_answer_metrics()
    checks: list[bool] = []

    expected_record_count = case.get("expected_record_count")
    if expected_record_count not in (None, ""):
        try:
            checks.append(
                retrieved_document_count == int(expected_record_count)
            )
        except (TypeError, ValueError):
            pass

    expected_counts = _parse_summary_risk_counts(
        case.get("expected_answer")
    )
    answer_norm = normalize_text(answer)

    for risk, count in expected_counts.items():
        patterns = (
            rf"\b{count}\s+{risk}(?:-risk)?\b",
            rf"\b{risk}(?:-risk)?\s*:\s*{count}\b",
        )
        checks.append(
            any(
                re.search(pattern, answer_norm, re.IGNORECASE)
                for pattern in patterns
            )
        )

    if not checks:
        return answer_metrics(
            answer,
            case.get("expected_answer"),
        )

    score = sum(checks) / len(checks)
    metrics.update(
        {
            "answer_exact_match": normalize_text(answer)
            == normalize_text(case.get("expected_answer")),
            "heuristic_answer_score": round(score, 6),
            "heuristic_answer_pass": all(checks),
        }
    )
    return metrics


def evaluate_answer_for_case(
    case: dict[str, Any],
    answer: str,
    retrieved_document_count: int,
) -> dict[str, Any]:
    group = clean_space(case.get("evaluation_group"))

    if group == "timeline_day_specific":
        return _timeline_day_metrics(case, answer)

    if group == "timeline_summary":
        return _timeline_summary_metrics(
            case,
            answer,
            retrieved_document_count,
        )

    return answer_metrics(
        answer,
        case.get("expected_answer"),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"ERROR: Input file not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"ERROR: Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise SystemExit(
                    f"ERROR: Expected a JSON object at {path}:{line_number}"
                )

            rows.append(row)

    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def first_value(row: dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def get_agent_trace(response: dict[str, Any]) -> dict[str, Any]:
    trace = response.get("agent_trace", {})
    return trace if isinstance(trace, dict) else {}


def get_retrieved_document_count(response: dict[str, Any]) -> int:
    trace = get_agent_trace(response)
    retrieval = trace.get("retrieval_agent", {})

    if isinstance(retrieval, dict):
        value = retrieval.get("retrieved_documents")
        if isinstance(value, int):
            return value

    sources = response.get("sources", [])
    return len(sources) if isinstance(sources, list) else 0


def get_retrieved_categories(response: dict[str, Any]) -> list[str]:
    trace = get_agent_trace(response)
    retrieval = trace.get("retrieval_agent", {})

    if isinstance(retrieval, dict):
        categories = retrieval.get("retrieved_categories")
        if isinstance(categories, list):
            return sorted({clean_space(item) for item in categories if item})

    sources = response.get("sources", [])
    if not isinstance(sources, list):
        return []

    return sorted(
        {
            clean_space(source.get("category"))
            for source in sources
            if isinstance(source, dict) and source.get("category")
        }
    )


def get_source_urls(response: dict[str, Any]) -> list[str]:
    sources = response.get("sources", [])
    if not isinstance(sources, list):
        return []

    urls: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue

        value = (
            source.get("document_url")
            or source.get("source_file")
            or source.get("url")
            or source.get("document_source")
            or ""
        )
        value = clean_space(value)
        if value:
            urls.append(value)

    return urls


def source_match(actual: str, expected: str) -> bool:
    actual_norm = normalize_text(Path(actual).name)
    expected_norm = normalize_text(Path(expected).name)

    return bool(
        actual_norm
        and expected_norm
        and (
            actual_norm == expected_norm
            or actual_norm in expected_norm
            or expected_norm in actual_norm
        )
    )


def retrieval_hits(
    source_urls: list[str],
    expected_source: Optional[str],
) -> tuple[Optional[bool], Optional[bool], Optional[bool]]:
    if not expected_source:
        return None, None, None

    hits = [
        source_match(actual, expected_source)
        for actual in source_urls
    ]

    hit_1 = bool(hits[:1] and any(hits[:1]))
    hit_3 = bool(hits[:3] and any(hits[:3]))
    hit_5 = bool(hits[:5] and any(hits[:5]))

    return hit_1, hit_3, hit_5


def get_verification(response: dict[str, Any]) -> dict[str, Any]:
    verification = (
        get_agent_trace(response)
        .get("verification_agent", {})
    )
    return verification if isinstance(verification, dict) else {}


def get_architecture_metrics(
    response: dict[str, Any],
) -> dict[str, Any]:
    trace = get_agent_trace(response)

    retrieval = trace.get("retrieval_agent", {})
    if not isinstance(retrieval, dict):
        retrieval = {}

    sufficiency = trace.get("evidence_sufficiency_agent", {})
    if not isinstance(sufficiency, dict):
        sufficiency = {}

    sufficient_value = sufficiency.get("sufficient")
    evidence_sufficient = (
        sufficient_value
        if isinstance(sufficient_value, bool)
        else None
    )

    return {
        "evidence_sufficient": evidence_sufficient,
        "evidence_assessment_method": clean_space(
            sufficiency.get("assessment_method")
        ),
        "refinement_attempted": bool(
            retrieval.get("refinement_attempted", False)
        ),
        "additional_documents": int(
            retrieval.get("additional_documents", 0) or 0
        ),
        "web_fallback_attempted": bool(
            retrieval.get("web_fallback_attempted", False)
        ),
        "web_fallback_used": bool(
            retrieval.get("web_fallback_used", False)
        ),
        "web_documents": int(
            retrieval.get("web_documents", 0) or 0
        ),
    }


def select_cases(
    cases: list[dict[str, Any]],
    groups: Optional[set[str]],
    smoke_one_per_group: bool,
    limit: Optional[int],
) -> list[dict[str, Any]]:
    selected = [
        case
        for case in cases
        if not groups or case.get("evaluation_group") in groups
    ]

    if smoke_one_per_group:
        one_per_group: list[dict[str, Any]] = []
        seen: set[str] = set()

        for case in selected:
            group = clean_space(case.get("evaluation_group"))
            if group and group not in seen:
                one_per_group.append(case)
                seen.add(group)

        selected = one_per_group

    if limit is not None:
        selected = selected[:limit]

    return selected



def is_fatal_provider_error(error_message: str) -> bool:
    text = str(error_message or "").lower()

    fatal_markers = (
        "error code: 402",
        "status code: 402",
        "credit limit exceeded",
        "'type': 'credit_limit'",
        '"type": "credit_limit"',
        "payment required",
        "error code: 401",
        "status code: 401",
        "invalid api key",
        "authentication error",
    )

    return any(marker in text for marker in fatal_markers)


def run_case(
    case: dict[str, Any],
    max_retries: int,
    retry_wait_seconds: float,
) -> tuple[dict[str, Any], float, str]:
    question = clean_space(case.get("question"))
    subject_id = first_value(case, SUBJECT_ID_KEYS)
    hadm_id = first_value(case, HADM_ID_KEYS)

    last_error = ""

    for attempt in range(1, max_retries + 1):
        started = time.perf_counter()

        try:
            response = run_medical_workflow(
                query=question,
                session_id=f"evaluation_{case.get('case_id', 'unknown')}",
                subject_id=subject_id,
                hadm_id=hadm_id,
            )
            elapsed = time.perf_counter() - started

            if not isinstance(response, dict):
                raise TypeError(
                    f"Workflow returned {type(response).__name__}, expected dict"
                )

            return response, elapsed, ""

        except Exception as exc:
            elapsed = time.perf_counter() - started
            last_error = f"{type(exc).__name__}: {exc}"

            if is_fatal_provider_error(last_error):
                raise SystemExit(
                    "\nFATAL PROVIDER ERROR: "
                    + last_error
                    + "\nThe current case was not saved. "
                    + "Fix billing/API access and run the same command "
                    + "again with --resume."
                )

            if attempt < max_retries:
                time.sleep(retry_wait_seconds * attempt)
            else:
                return {}, elapsed, last_error

    return {}, 0.0, last_error


def evaluate_case(
    case: dict[str, Any],
    response: dict[str, Any],
    elapsed: float,
    error: str,
) -> dict[str, Any]:
    answer = clean_space(response.get("message"))
    predicted_route = clean_space(response.get("route"))
    expected_route = clean_space(case.get("expected_route"))

    expected_category = clean_space(case.get("expected_category"))
    retrieved_categories = get_retrieved_categories(response)
    source_urls = get_source_urls(response)

    expected_source = first_value(case, EXPECTED_SOURCE_KEYS)
    hit_1, hit_3, hit_5 = retrieval_hits(source_urls, expected_source)

    retrieved_document_count = get_retrieved_document_count(response)
    expected_record_count = case.get("expected_record_count")

    summary_count_correct: Optional[bool] = None
    if expected_record_count not in (None, ""):
        try:
            summary_count_correct = (
                retrieved_document_count == int(expected_record_count)
            )
        except (TypeError, ValueError):
            summary_count_correct = None

    verification = get_verification(response)
    verified = bool(verification.get("verified", False))

    architecture_metrics = get_architecture_metrics(response)

    request_success = (
        not error
        and bool(response)
        and bool(response.get("found", False))
        and bool(answer)
    )

    answer_eval = evaluate_answer_for_case(
        case=case,
        answer=answer,
        retrieved_document_count=retrieved_document_count,
    )

    route_correct = predicted_route == expected_route
    category_correct = (
        expected_category in retrieved_categories
        if expected_category
        else None
    )

    retrieval_requirement_met = (
        hit_5
        if expected_source
        else True
    )

    count_requirement_met = (
        summary_count_correct
        if summary_count_correct is not None
        else True
    )

    heuristic_answer_pass = answer_eval["heuristic_answer_pass"]
    answer_requirement_met = (
        heuristic_answer_pass
        if heuristic_answer_pass is not None
        else True
    )

    common_system_pass = bool(
        request_success
        and verified
        and heuristic_answer_pass is True
    )

    overall_automatic_pass = bool(
        request_success
        and route_correct
        and category_correct is not False
        and retrieval_requirement_met is not False
        and count_requirement_met is not False
        and verified
        and answer_requirement_met
    )

    return {
        "case_id": clean_space(case.get("case_id")),
        "evaluation_phase": clean_space(case.get("evaluation_phase")),
        "evaluation_group": clean_space(case.get("evaluation_group")),
        "question_type": clean_space(case.get("question_type")),
        "question": clean_space(case.get("question")),
        "expected_answer": case.get("expected_answer"),
        "answer": answer,
        "expected_route": expected_route,
        "predicted_route": predicted_route,
        "route_correct": route_correct,
        "expected_category": expected_category,
        "retrieved_categories": retrieved_categories,
        "category_correct": category_correct,
        "expected_source": expected_source or "",
        "source_urls": source_urls,
        "retrieval_hit_at_1": hit_1,
        "retrieval_hit_at_3": hit_3,
        "retrieval_hit_at_5": hit_5,
        "expected_record_count": (
            expected_record_count
            if expected_record_count not in (None, "")
            else None
        ),
        "retrieved_document_count": retrieved_document_count,
        "summary_record_count_correct": summary_count_correct,
        "verified": verified,
        **architecture_metrics,
        "verification_warning": clean_space(
            verification.get("warning")
        ),
        "verification_query_coverage": verification.get(
            "query_coverage"
        ),
        "verification_grounding_coverage": verification.get(
            "grounding_coverage"
        ),
        "verification_unsupported_numbers": verification.get(
            "unsupported_numbers",
            [],
        ),
        **answer_eval,
        "request_success": request_success,
        "common_system_pass": common_system_pass,
        "overall_automatic_pass": overall_automatic_pass,
        "response_time_seconds": round(elapsed, 6),
        "error": error,
        "subject_id": first_value(case, SUBJECT_ID_KEYS) or "",
        "hadm_id": first_value(case, HADM_ID_KEYS) or "",
        "agent_trace": get_agent_trace(response),
        "raw_response": response,
    }


def safe_mean(values: list[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def safe_median(values: list[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def boolean_rate(rows: list[dict[str, Any]], key: str) -> Optional[float]:
    values = [
        row.get(key)
        for row in rows
        if isinstance(row.get(key), bool)
    ]
    return (
        sum(bool(value) for value in values) / len(values)
        if values
        else None
    )


def numeric_mean(rows: list[dict[str, Any]], key: str) -> Optional[float]:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and not isinstance(row.get(key), bool)
    ]
    return safe_mean(values)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    response_times = [
        float(row["response_time_seconds"])
        for row in rows
        if row.get("request_success") is True
    ]

    return {
        "cases": len(rows),
        "request_success_rate": boolean_rate(rows, "request_success"),
        "route_accuracy": boolean_rate(rows, "route_correct"),
        "category_accuracy": boolean_rate(rows, "category_correct"),
        "retrieval_hit_at_1": boolean_rate(rows, "retrieval_hit_at_1"),
        "retrieval_hit_at_3": boolean_rate(rows, "retrieval_hit_at_3"),
        "retrieval_hit_at_5": boolean_rate(rows, "retrieval_hit_at_5"),
        "summary_record_count_accuracy": boolean_rate(
            rows,
            "summary_record_count_correct",
        ),
        "verification_success_rate": boolean_rate(rows, "verified"),
        "evidence_sufficiency_rate": boolean_rate(
            rows,
            "evidence_sufficient",
        ),
        "refinement_attempt_rate": boolean_rate(
            rows,
            "refinement_attempted",
        ),
        "web_fallback_attempt_rate": boolean_rate(
            rows,
            "web_fallback_attempted",
        ),
        "web_fallback_usage_rate": boolean_rate(
            rows,
            "web_fallback_used",
        ),
        "mean_additional_documents": numeric_mean(
            rows,
            "additional_documents",
        ),
        "mean_web_documents": numeric_mean(
            rows,
            "web_documents",
        ),
        "heuristic_answer_pass_rate": boolean_rate(
            rows,
            "heuristic_answer_pass",
        ),
        "common_system_pass_rate": boolean_rate(
            rows,
            "common_system_pass",
        ),
        "mean_heuristic_answer_score": numeric_mean(
            rows,
            "heuristic_answer_score",
        ),
        "mean_answer_token_f1": numeric_mean(
            rows,
            "answer_token_f1",
        ),
        "overall_automatic_pass_rate": boolean_rate(
            rows,
            "overall_automatic_pass",
        ),
        "average_response_time_seconds": safe_mean(response_times),
        "median_response_time_seconds": safe_median(response_times),
        "p95_response_time_seconds": percentile(response_times, 0.95),
    }


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[clean_space(row.get("evaluation_group"))].append(row)

    return {
        "metric_note": (
            "Answer correctness is reported as a lexical/numeric heuristic. "
            "It is not a substitute for clinical expert or LLM-judge review."
        ),
        "overall": summarize_rows(rows),
        "by_evaluation_group": {
            group: summarize_rows(group_rows)
            for group, group_rows in sorted(grouped.items())
        },
    }


def json_for_csv(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    field_names = [
        "case_id",
        "evaluation_phase",
        "evaluation_group",
        "question_type",
        "question",
        "expected_answer",
        "answer",
        "expected_route",
        "predicted_route",
        "route_correct",
        "expected_category",
        "retrieved_categories",
        "category_correct",
        "expected_source",
        "source_urls",
        "retrieval_hit_at_1",
        "retrieval_hit_at_3",
        "retrieval_hit_at_5",
        "expected_record_count",
        "retrieved_document_count",
        "summary_record_count_correct",
        "verified",
        "evidence_sufficient",
        "evidence_assessment_method",
        "refinement_attempted",
        "additional_documents",
        "web_fallback_attempted",
        "web_fallback_used",
        "web_documents",
        "verification_warning",
        "verification_query_coverage",
        "verification_grounding_coverage",
        "verification_unsupported_numbers",
        "answer_exact_match",
        "answer_token_precision",
        "answer_token_recall",
        "answer_token_f1",
        "answer_sequence_similarity",
        "answer_number_coverage",
        "heuristic_answer_score",
        "heuristic_answer_pass",
        "request_success",
        "overall_automatic_pass",
        "response_time_seconds",
        "error",
        "subject_id",
        "hadm_id",
        "agent_trace",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()

        for row in rows:
            csv_row = {
                key: row.get(key)
                for key in field_names
            }

            for key in (
                "expected_answer",
                "retrieved_categories",
                "source_urls",
                "verification_unsupported_numbers",
                "agent_trace",
            ):
                csv_row[key] = json_for_csv(csv_row.get(key))

            writer.writerow(csv_row)


def validate_input_dataset(cases: list[dict[str, Any]]) -> None:
    ids = [clean_space(case.get("case_id")) for case in cases]

    if len(ids) != len(set(ids)):
        raise SystemExit("ERROR: Duplicate case IDs in input dataset")

    if len(cases) == 800:
        group_counts = Counter(
            clean_space(case.get("evaluation_group"))
            for case in cases
        )

        if dict(group_counts) != EXPECTED_GROUP_COUNTS:
            raise SystemExit(
                "ERROR: Unexpected evaluation group counts: "
                f"{dict(group_counts)}"
            )


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    return {
        clean_space(row.get("case_id")): row
        for row in load_jsonl(path)
        if clean_space(row.get("case_id"))
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    overall = metrics["overall"]

    def pct(value: Any) -> str:
        return (
            "N/A"
            if value is None
            else f"{float(value) * 100:.2f}%"
        )

    print("=" * 72)
    print("CASES:", overall["cases"])
    print("REQUEST SUCCESS:", pct(overall["request_success_rate"]))
    print("ROUTE ACCURACY:", pct(overall["route_accuracy"]))
    print("CATEGORY ACCURACY:", pct(overall["category_accuracy"]))
    print("RETRIEVAL HIT@1:", pct(overall["retrieval_hit_at_1"]))
    print("RETRIEVAL HIT@3:", pct(overall["retrieval_hit_at_3"]))
    print("VERIFICATION SUCCESS:", pct(
        overall["verification_success_rate"]
    ))
    print("HEURISTIC ANSWER PASS:", pct(
        overall["heuristic_answer_pass_rate"]
    ))
    print("COMMON SYSTEM PASS:", pct(
        overall["common_system_pass_rate"]
    ))
    print("OVERALL AUTOMATIC PASS:", pct(
        overall["overall_automatic_pass_rate"]
    ))
    average_time = overall["average_response_time_seconds"]
    print(
        "AVERAGE RESPONSE TIME:",
        "N/A" if average_time is None else f"{average_time:.4f}s",
    )



def warmup_system() -> None:
    """
    Non-evaluation warm-up.
    Loads routing/retrieval/reranking/LLM components in the same
    Python process before evaluation latency measurement begins.
    The warm-up response is discarded and is not part of the dataset.
    """
    print("SYSTEM WARM-UP: starting...", flush=True)

    response = run_medical_workflow(
        query="What is anemia?",
        session_id="article_internal_non_evaluation_warmup",
        subject_id=None,
        hadm_id=None,
    )

    if not isinstance(response, dict):
        raise RuntimeError(
            "Warm-up workflow did not return a dictionary."
        )

    print("SYSTEM WARM-UP: PASS", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--run-name",
        default="article_internal_800_full",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Repeat to select one or more evaluation groups.",
    )
    parser.add_argument(
        "--smoke-one-per-group",
        action="store_true",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()


    warmup_system()
    input_path = (
        args.input
        if args.input.is_absolute()
        else PROJECT_ROOT / args.input
    )
    results_dir = (
        args.results_dir
        if args.results_dir.is_absolute()
        else PROJECT_ROOT / args.results_dir
    )

    results_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = results_dir / f"{args.run_name}_predictions.jsonl"
    csv_path = results_dir / f"{args.run_name}_predictions.csv"
    metrics_path = results_dir / f"{args.run_name}_metrics.json"

    if jsonl_path.exists() and not args.resume:
        raise SystemExit(
            f"ERROR: Output already exists: {jsonl_path}\n"
            "Use --resume to continue it or choose another --run-name."
        )

    cases = load_jsonl(input_path)
    validate_input_dataset(cases)

    groups = set(args.group) if args.group else None
    selected = select_cases(
        cases=cases,
        groups=groups,
        smoke_one_per_group=args.smoke_one_per_group,
        limit=args.limit,
    )

    completed = load_completed(jsonl_path) if args.resume else {}
    pending = [
        case
        for case in selected
        if clean_space(case.get("case_id")) not in completed
    ]

    print("INPUT:", input_path)
    print("SELECTED CASES:", len(selected))
    print("ALREADY COMPLETED:", len(completed))
    print("PENDING:", len(pending))
    print("JSONL OUTPUT:", jsonl_path)

    total_selected = len(selected)

    for index, case in enumerate(pending, start=1):
        case_id = clean_space(case.get("case_id"))

        response, elapsed, error = run_case(
            case=case,
            max_retries=args.max_retries,
            retry_wait_seconds=args.retry_wait_seconds,
        )

        result = evaluate_case(
            case=case,
            response=response,
            elapsed=elapsed,
            error=error,
        )

        append_jsonl(jsonl_path, result)
        completed[case_id] = result

        done = len(completed)
        print(
            f"[{done}/{total_selected}] "
            f"{case_id} | "
            f"route={result['predicted_route']} | "
            f"route_ok={result['route_correct']} | "
            f"verified={result['verified']} | "
            f"answer_score={result['heuristic_answer_score']} | "
            f"time={result['response_time_seconds']:.3f}s",
            flush=True,
        )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    ordered_results = [
        completed[clean_space(case.get("case_id"))]
        for case in selected
        if clean_space(case.get("case_id")) in completed
    ]

    save_csv(csv_path, ordered_results)

    metrics = calculate_metrics(ordered_results)
    metrics.update(
        {
            "input_file": str(input_path),
            "run_name": args.run_name,
            "selected_cases": len(selected),
            "completed_cases": len(ordered_results),
            "jsonl_output": str(jsonl_path),
            "csv_output": str(csv_path),
        }
    )

    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_metrics(metrics)
    print("CSV OUTPUT:", csv_path)
    print("METRICS OUTPUT:", metrics_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nStopped safely. Run the same command with --resume.",
            file=sys.stderr,
        )
        raise SystemExit(130)
