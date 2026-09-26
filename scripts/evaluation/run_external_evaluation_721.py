#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_agent_evaluation_1500 as base  # noqa: E402


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "external_evaluation_final_721.jsonl"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"


def load_cases() -> list[dict[str, Any]]:
    cases = base.load_jsonl(INPUT_FILE)

    if len(cases) != 721:
        raise SystemExit(
            f"ERROR: Expected exactly 721 external cases, found {len(cases)}"
        )

    ids = [base.clean_space(x.get("case_id")) for x in cases]

    if len(ids) != len(set(ids)):
        raise SystemExit("ERROR: Duplicate case IDs found.")

    return cases



def build_system_query(case: dict[str, Any]) -> str:
    question = base.clean_space(case.get("question"))
    source = base.clean_space(case.get("external_source"))

    if source == "TREC 2017 LiveQA Medical":
        subject = base.clean_space(case.get("subject"))
        if subject:
            return f"Subject: {subject}\nQuestion: {question}"

    if source == "MedicationQA":
        drug = base.clean_space(case.get("drug"))

        if drug and not drug.lower().startswith("n/a"):
            return f"Drug: {drug}\nQuestion: {question}"

    return question

def reference_answers(case: dict[str, Any]) -> list[str]:
    refs: list[str] = []

    single = base.clean_space(case.get("reference_answer"))
    if single:
        refs.append(single)

    multiple = case.get("reference_answers") or []
    if isinstance(multiple, list):
        for value in multiple:
            value = base.clean_space(value)
            if value and value not in refs:
                refs.append(value)

    return refs


def evaluate_answer(
    answer: str,
    references: list[str],
) -> dict[str, Any]:

    if not references:
        return {
            "best_reference": "",
            "best_reference_index": None,
            "answer_token_f1": None,
            "answer_sequence_similarity": None,
            "answer_number_coverage": None,
            "heuristic_answer_score": None,
            "heuristic_answer_pass": None,
        }

    scored = []

    for index, reference in enumerate(references):
        metrics = base.answer_metrics(answer, reference)
        score = metrics.get("heuristic_answer_score")

        scored.append((
            -1.0 if score is None else float(score),
            index,
            reference,
            metrics,
        ))

    scored.sort(key=lambda x: x[0], reverse=True)

    _, index, reference, metrics = scored[0]

    return {
        "best_reference": reference,
        "best_reference_index": index,
        "answer_token_f1": metrics["answer_token_f1"],
        "answer_sequence_similarity": metrics[
            "answer_sequence_similarity"
        ],
        "answer_number_coverage": metrics[
            "answer_number_coverage"
        ],
        "heuristic_answer_score": metrics[
            "heuristic_answer_score"
        ],
        "heuristic_answer_pass": metrics[
            "heuristic_answer_pass"
        ],
    }


def evaluate_case(
    case: dict[str, Any],
    response: dict[str, Any],
    elapsed: float,
    error: str,
) -> dict[str, Any]:

    answer = base.clean_space(response.get("message"))
    predicted_route = base.clean_space(response.get("route"))

    evaluation_group = base.clean_space(
        case.get("evaluation_group")
    )

    expected_route = (
        "general_medical_rag"
        if evaluation_group in {
            "external_medication",
            "external_general_medical",
        }
        else ""
    )

    route_correct = (
        predicted_route == expected_route
        if expected_route
        else None
    )

    verification = base.get_verification(response)
    verified = bool(verification.get("verified", False))

    agent_trace = base.get_agent_trace(response)

    retrieval_trace = (
        agent_trace.get("retrieval_agent", {})
        if isinstance(agent_trace, dict)
        else {}
    )

    knowledge_base_documents = retrieval_trace.get(
        "knowledge_base_documents"
    )

    web_fallback_allowed = bool(
        retrieval_trace.get(
            "web_fallback_allowed",
            False,
        )
    )

    web_fallback_attempted = bool(
        retrieval_trace.get(
            "web_fallback_attempted",
            False,
        )
    )

    web_fallback_used = bool(
        retrieval_trace.get("web_fallback_used", False)
    )

    web_fallback_reason = base.clean_space(
        retrieval_trace.get("web_fallback_reason")
    )

    web_documents = retrieval_trace.get(
        "web_documents",
        0,
    )

    web_sources = []
    web_source_names = []

    if web_fallback_used:
        for source in response.get("sources", []) or []:
            if not isinstance(source, dict):
                continue

            source_name = base.clean_space(
                source.get("document_source")
            )
            source_url = base.clean_space(
                source.get("document_url")
            )

            web_sources.append({
                "document_source": source_name,
                "document_url": source_url,
            })

            if source_name:
                web_source_names.append(source_name)

    web_source_names = sorted(
        set(web_source_names)
    )

    request_success = (
        not error
        and bool(response)
        and bool(response.get("found", False))
        and bool(answer)
    )

    refs = reference_answers(case)
    answer_eval = evaluate_answer(answer, refs)

    heuristic_answer_pass = answer_eval.get(
        "heuristic_answer_pass"
    )

    common_system_pass = bool(
        request_success
        and verified
        and heuristic_answer_pass is True
    )

    return {
        "case_id": base.clean_space(case.get("case_id")),
        "evaluation_group": base.clean_space(
            case.get("evaluation_group")
        ),
        "external_source": base.clean_space(
            case.get("external_source")
        ),
        "question_type": base.clean_space(
            case.get("question_type")
        ),
        "subject": base.clean_space(case.get("subject")),
        "drug": base.clean_space(case.get("drug")),
        "question": base.clean_space(case.get("question")),
        "query_sent_to_system": build_system_query(case),
        "reference_answers": refs,
        "answer": answer,
        "expected_route": expected_route,
        "predicted_route": predicted_route,
        "route_correct": route_correct,
        "retrieved_categories": base.get_retrieved_categories(response),
        "retrieved_sources": base.get_source_urls(response),
        "retrieved_document_count": (
            base.get_retrieved_document_count(response)
        ),
        "knowledge_base_documents": knowledge_base_documents,
        "web_fallback_allowed": web_fallback_allowed,
        "web_fallback_attempted": web_fallback_attempted,
        "web_fallback_used": web_fallback_used,
        "web_fallback_reason": web_fallback_reason,
        "web_documents": web_documents,
        "web_source_names": web_source_names,
        "web_sources": web_sources,
        "verified": verified,
        "verification_warning": base.clean_space(
            verification.get("warning")
        ),
        "verification_query_coverage": verification.get(
            "query_coverage"
        ),
        "verification_grounding_coverage": verification.get(
            "grounding_coverage"
        ),
        **answer_eval,
        "request_success": request_success,
        "common_system_pass": common_system_pass,
        "response_time_seconds": round(elapsed, 6),
        "error": error,
        "agent_trace": agent_trace,
        "raw_response": response,
    }


def mean_non_null(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:

    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None
    ]

    return statistics.mean(values) if values else None


def rate_non_null(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:

    values = [
        bool(row[key])
        for row in rows
        if row.get(key) is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)



def median_non_null(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:

    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None
    ]

    return statistics.median(values) if values else None


def percentile_non_null(
    rows: list[dict[str, Any]],
    key: str,
    q: float,
) -> float | None:

    values = sorted(
        float(row[key])
        for row in rows
        if row.get(key) is not None
    )

    if not values:
        return None

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def build_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:

    route_counts = Counter(
        row.get("predicted_route") or "EMPTY"
        for row in rows
    )

    source_counts = Counter(
        row.get("external_source") or "UNKNOWN"
        for row in rows
    )

    web_fallback_cases = sum(
        1
        for row in rows
        if bool(row.get("web_fallback_used", False))
    )

    no_web_fallback_cases = (
        len(rows) - web_fallback_cases
    )

    web_reason_counts = Counter(
        row.get("web_fallback_reason") or "UNKNOWN"
        for row in rows
        if bool(row.get("web_fallback_used", False))
    )

    web_source_counts = Counter(
        source_name
        for row in rows
        if bool(row.get("web_fallback_used", False))
        for source_name in (
            row.get("web_source_names") or []
        )
    )

    return {
        "cases": len(rows),
        "request_success_rate": rate_non_null(
            rows, "request_success"
        ),
        "route_accuracy": rate_non_null(
            rows, "route_correct"
        ),
        "verification_success_rate": rate_non_null(
            rows, "verified"
        ),
        "heuristic_answer_pass_rate": rate_non_null(
            rows, "heuristic_answer_pass"
        ),
        "common_system_pass_rate": rate_non_null(
            rows, "common_system_pass"
        ),
        "mean_heuristic_answer_score": mean_non_null(
            rows, "heuristic_answer_score"
        ),
        "mean_answer_token_f1": mean_non_null(
            rows, "answer_token_f1"
        ),
        "average_response_time_seconds": mean_non_null(
            rows, "response_time_seconds"
        ),
        "median_response_time_seconds": median_non_null(
            rows, "response_time_seconds"
        ),
        "p95_response_time_seconds": percentile_non_null(
            rows, "response_time_seconds", 0.95
        ),
        "predicted_route_distribution": dict(route_counts),
        "external_source_distribution": dict(source_counts),

        "web_fallback_cases": web_fallback_cases,

        "no_web_fallback_cases": no_web_fallback_cases,

        "web_fallback_usage_rate": (
            web_fallback_cases / len(rows)
            if rows
            else None
        ),

        "web_fallback_by_reason": dict(
            web_reason_counts
        ),

        "web_source_distribution": dict(
            web_source_counts
        ),
    }


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            converted = {}

            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    converted[key] = json.dumps(
                        value,
                        ensure_ascii=False,
                    )
                else:
                    converted[key] = value

            writer.writerow(converted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        default="external_evaluation_721",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--resume",
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cases = load_cases()

    if args.limit is not None:
        cases = cases[:args.limit]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = (
        RESULTS_DIR
        / f"{args.run_name}_predictions.jsonl"
    )

    csv_path = (
        RESULTS_DIR
        / f"{args.run_name}_predictions.csv"
    )

    metrics_path = (
        RESULTS_DIR
        / f"{args.run_name}_metrics.json"
    )

    if jsonl_path.exists() and not args.resume:
        raise SystemExit(
            f"ERROR: Output already exists: {jsonl_path}\n"
            "Use --resume or choose another --run-name."
        )

    completed = (
        base.load_completed(jsonl_path)
        if args.resume
        else {}
    )

    pending = [
        case
        for case in cases
        if base.clean_space(case.get("case_id"))
        not in completed
    ]

    print("INPUT:", INPUT_FILE)
    print("SELECTED CASES:", len(cases))
    print("ALREADY COMPLETED:", len(completed))
    print("PENDING:", len(pending))

    for case in pending:

        case_for_run = dict(case)
        case_for_run["question"] = build_system_query(case)

        response, elapsed, error = base.run_case(
            case=case_for_run,
            max_retries=args.max_retries,
            retry_wait_seconds=args.retry_wait_seconds,
        )

        result = evaluate_case(
            case=case,
            response=response,
            elapsed=elapsed,
            error=error,
        )

        base.append_jsonl(jsonl_path, result)

        case_id = base.clean_space(case.get("case_id"))
        completed[case_id] = result

        print(
            f"[{len(completed)}/{len(cases)}] "
            f"{case_id} | "
            f"route={result['predicted_route']} | "
            f"verified={result['verified']} | "
            f"web={result['web_fallback_used']} | "
            f"score={result['heuristic_answer_score']} | "
            f"time={result['response_time_seconds']:.3f}s",
            flush=True,
        )

    ordered = [
        completed[base.clean_space(case.get("case_id"))]
        for case in cases
        if base.clean_space(case.get("case_id"))
        in completed
    ]

    save_csv(csv_path, ordered)

    metrics = build_metrics(ordered)

    metrics.update({
        "input_file": str(INPUT_FILE),
        "run_name": args.run_name,
        "selected_cases": len(cases),
        "completed_cases": len(ordered),
        "note": (
            "For external_medication and external_general_medical cases, "
            "the expected route is general_medical_rag and is "
            "compared with the route predicted by the five-route "
            "router. Gold retrieval-document labels are not available."
        ),
    })

    metrics_path.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("CASES:", metrics["cases"])
    print(
        "REQUEST SUCCESS:",
        metrics["request_success_rate"],
    )
    print(
        "VERIFICATION SUCCESS:",
        metrics["verification_success_rate"],
    )
    print(
        "HEURISTIC ANSWER PASS:",
        metrics["heuristic_answer_pass_rate"],
    )
    print(
        "MEAN ANSWER TOKEN F1:",
        metrics["mean_answer_token_f1"],
    )
    print(
        "AVERAGE RESPONSE TIME:",
        metrics["average_response_time_seconds"],
    )
    print(
        "PREDICTED ROUTES:",
        metrics["predicted_route_distribution"],
    )

    print("\nJSONL:", jsonl_path)
    print("CSV:", csv_path)
    print("METRICS:", metrics_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nStopped safely. Run again with --resume.",
            file=sys.stderr,
        )
        raise SystemExit(130)
