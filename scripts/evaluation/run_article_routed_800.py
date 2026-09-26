#!/usr/bin/env python3
"""
Article ablation runner for the two configurations missing from the thesis study.

Modes
-----
routed_rag
    Five-route router -> route-specific retrieval -> generic RAG generation.
    ColBERT reranking is disabled.

routed_rag_rerank
    Five-route router -> route-specific retrieval -> ColBERT reranking
    -> generic RAG generation.

Important experimental controls
-------------------------------
- Uses the frozen 800-case article evaluation dataset.
- Does NOT call evidence sufficiency, query refinement, trusted-web fallback,
  safety triage, or verification.
- Uses generate_llm_answer(), matching the answer-generation function used by
  the thesis simple_rag/no_agent_rag baselines, so the comparison isolates
  routing first and reranking second as cleanly as possible.
- Timeline routes retain the project's native route-specific exact lookup
  behavior; therefore ColBERT is not applicable to exact timeline retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.medical_workflow import (  # noqa: E402
    patient_context_agent,
    retrieval_agent,
    router_agent,
)
from app.services.medical_generation import generate_llm_answer  # noqa: E402
from scripts.evaluation.run_agent_evaluation_1500 import (  # noqa: E402
    EXPECTED_SOURCE_KEYS,
    HADM_ID_KEYS,
    SUBJECT_ID_KEYS,
    evaluate_answer_for_case,
    first_value,
    is_fatal_provider_error,
    retrieval_hits,
)


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "article_internal_holdout_800_final.jsonl"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"

# Frozen 800-case article evaluation dataset.
EXPECTED_SHA256 = (
    "9229c2fe4591b6f073a4f31c9dcbbefcc212a8fe515a3eab5ea4bce6695daadf"
)

VALID_MODES = {
    "routed_rag",
    "routed_rag_rerank",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_smoke_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one case from each of the four evaluation groups."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for case in cases:
        group = str(case.get("evaluation_group", "")).strip()
        if group and group not in seen:
            selected.append(case)
            seen.add(group)

    return selected


def run_routed_configuration(
    mode: str,
    case: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    question = str(case.get("question", "")).strip()

    subject_id = first_value(case, SUBJECT_ID_KEYS)
    hadm_id = first_value(case, HADM_ID_KEYS)

    # 1) The five-route classifier used by the proposed system.
    routing = router_agent(
        query=question,
        subject_id=subject_id,
        hadm_id=hadm_id,
    )

    # The holdout cases should contain the metadata required by their route.
    # If not, fail this case transparently rather than silently changing route.
    missing_information = routing.get("missing_information") or []
    if missing_information:
        raise RuntimeError(
            "Router requires missing information: "
            + ", ".join(str(item) for item in missing_information)
        )

    # 2) EHR identifiers are used only as legitimate retrieval constraints.
    patient_context = patient_context_agent(
        routing=routing,
        subject_id=subject_id,
        hadm_id=hadm_id,
    )

    # 3) Toggle only ColBERT between the two article configurations.
    if mode == "routed_rag":
        os.environ["USE_RERANKER"] = "false"
    elif mode == "routed_rag_rerank":
        os.environ["USE_RERANKER"] = "true"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Route-specific retrieval. No sufficiency/refinement/verification pipeline.
    docs = retrieval_agent(
        query=question,
        routing=routing,
        patient_context=patient_context,
    )

    # Keep generation aligned with the thesis RAG baselines.
    answer = generate_llm_answer(
        query=question,
        docs=docs,
        domain="medical",
    )

    return answer, docs, routing


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id", "")).strip()
            if case_id:
                completed.add(case_id)
    return completed


def evaluate_case(
    case: dict[str, Any],
    mode: str,
    answer: str,
    docs: list[dict[str, Any]],
    routing: dict[str, Any],
    elapsed: float,
    error: str,
) -> dict[str, Any]:
    retrieved_count = len(docs)

    answer_metrics = evaluate_answer_for_case(
        case=case,
        answer=answer,
        retrieved_document_count=retrieved_count,
    )

    source_urls = [
        str(doc.get("document_url", ""))
        for doc in docs
        if doc.get("document_url")
    ]

    expected_source = first_value(case, EXPECTED_SOURCE_KEYS)
    hit_1, hit_3, hit_5 = retrieval_hits(source_urls, expected_source)

    predicted_route = str(routing.get("route", "")).strip()
    expected_route = str(case.get("expected_route", "")).strip()
    route_correct = (
        predicted_route == expected_route
        if expected_route
        else None
    )

    retrieved_categories = sorted(
        {
            str(doc.get("category", "")).strip()
            for doc in docs
            if doc.get("category")
        }
    )

    row = {
        "case_id": case.get("case_id"),
        "evaluation_group": case.get("evaluation_group"),
        "question_type": case.get("question_type"),
        "mode": mode,
        "question": case.get("question"),
        "expected_answer": case.get("expected_answer"),
        "answer": answer,
        "expected_route": expected_route,
        "predicted_route": predicted_route,
        "route_correct": route_correct,
        "routing_method": routing.get("routing_method"),
        "expected_category": case.get("expected_category"),
        "retrieved_categories": retrieved_categories,
        "expected_source": expected_source,
        "source_urls": source_urls,
        "retrieved_document_count": retrieved_count,
        "retrieval_hit_at_1": hit_1,
        "retrieval_hit_at_3": hit_3,
        "retrieval_hit_at_5": hit_5,
        "response_time_seconds": round(elapsed, 6),
        "request_success": bool(answer) and not error,
        "error": error,
    }

    row.update(answer_metrics)
    return row


def _bool_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if isinstance(row.get(key), bool)]
    if not values:
        return None
    return sum(bool(value) for value in values) / len(values)


def _num_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and not isinstance(row.get(key), bool)
    ]
    return mean(values) if values else None


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    successful = [row for row in rows if row.get("request_success")]
    times = [
        float(row["response_time_seconds"])
        for row in rows
        if row.get("response_time_seconds") is not None
    ]

    print()
    print("===== ARTICLE ABLATION SUMMARY =====")
    print("CASES:", len(rows))
    print("REQUEST SUCCESS:", round(len(successful) / len(rows), 4))

    route_accuracy = _bool_rate(rows, "route_correct")
    if route_accuracy is not None:
        print("ROUTE ACCURACY:", round(route_accuracy, 4))

    for key, label in (
        ("retrieval_hit_at_1", "RETRIEVAL HIT@1"),
        ("retrieval_hit_at_3", "RETRIEVAL HIT@3"),
        ("retrieval_hit_at_5", "RETRIEVAL HIT@5"),
    ):
        value = _bool_rate(rows, key)
        if value is not None:
            print(f"{label}:", round(value, 4))

    token_f1 = _num_mean(rows, "answer_token_f1")
    if token_f1 is not None:
        print("MEAN TOKEN F1:", round(token_f1, 6))

    heuristic_score = _num_mean(rows, "heuristic_answer_score")
    if heuristic_score is not None:
        print("MEAN HEURISTIC SCORE:", round(heuristic_score, 6))

    pass_rate = _bool_rate(rows, "heuristic_answer_pass")
    if pass_rate is not None:
        print("HEURISTIC PASS RATE:", round(pass_rate, 4))

    if times:
        print("MEAN RESPONSE TIME:", round(mean(times), 6))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(VALID_MODES),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one question from each evaluation group (4 total).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument("--max-new", type=int, required=True)

    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--group", choices=[
        "ehr_clinical_qa",
        "pediatric_oncology_qa",
        "timeline_day_specific",
        "timeline_summary",
    ])

    args = parser.parse_args()
    if args.max_new <= 0:
        parser.error("--max-new must be a positive integer")

    if not INPUT_PATH.exists():
        raise SystemExit(f"ERROR: input file not found: {INPUT_PATH}")

    actual_sha = file_sha256(INPUT_PATH)
    if actual_sha != EXPECTED_SHA256:
        raise SystemExit(
            "ERROR: Ablation sample hash changed.\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Actual:   {actual_sha}"
        )

    cases = load_jsonl(INPUT_PATH)
    if len(cases) != 800:
        raise SystemExit(f"ERROR: expected 800 cases, found {len(cases)}")

    if args.smoke:
        cases = select_smoke_cases(cases)

    if args.group:
        cases = [c for c in cases if c.get("evaluation_group") == args.group]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "smoke4" if args.smoke else "800"
    output_path = (
        RESULTS_DIR
        / f"article800_routed_{args.mode}_{suffix}_predictions.jsonl"
    )

    if output_path.exists() and not args.resume:
        raise SystemExit(
            f"ERROR: output already exists: {output_path}\n"
            "Use --resume to continue it, or delete the old smoke output first."
        )

    completed_ids = load_completed_ids(output_path) if args.resume else set()
    pending = [
        case
        for case in cases
        if str(case.get("case_id", "")).strip() not in completed_ids
    ]

    total_pending = len(pending)
    pending = pending[:args.max_new]
    print("TOTAL PENDING:", total_pending)
    print("MODE:", args.mode)
    print("INPUT:", INPUT_PATH)
    print("INPUT SHA256:", actual_sha)
    print("SELECTED CASES:", len(cases))
    print("ALREADY COMPLETED:", len(completed_ids))
    print("PENDING:", len(pending))
    print("OUTPUT:", output_path)

    if args.dry_run:
        print("DRY RUN: No questions executed.")
        print("BATCH IDS:", [c["case_id"] for c in pending])
        return

    for index, case in enumerate(pending, start=len(completed_ids) + 1):
        case_id = str(case.get("case_id", "")).strip()
        started = time.perf_counter()

        answer = ""
        docs: list[dict[str, Any]] = []
        routing: dict[str, Any] = {}
        error = ""

        try:
            answer, docs, routing = run_routed_configuration(
                mode=args.mode,
                case=case,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

            if is_fatal_provider_error(error):
                raise SystemExit(
                    "\nFATAL PROVIDER ERROR: "
                    + error
                    + "\nThe current case was not saved. "
                    + "Fix billing/API access and rerun with --resume."
                )

        elapsed = time.perf_counter() - started

        row = evaluate_case(
            case=case,
            mode=args.mode,
            answer=answer,
            docs=docs,
            routing=routing,
            elapsed=elapsed,
            error=error,
        )

        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"[{index}/{len(cases)}] "
            f"{case_id} | "
            f"route={row.get('predicted_route')} | "
            f"route_ok={row.get('route_correct')} | "
            f"docs={len(docs)} | "
            f"score={row.get('heuristic_answer_score')} | "
            f"pass={row.get('heuristic_answer_pass')} | "
            f"time={elapsed:.3f}s",
            flush=True,
        )

    all_rows = load_jsonl(output_path)
    print_summary(all_rows)


if __name__ == "__main__":
    main()
