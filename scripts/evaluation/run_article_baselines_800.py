#!/usr/bin/env python3

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

from openai import OpenAI

from app.services import medical_retrieval as retrieval_module
from app.services.colbert_reranker import rerank_candidates
from app.services.medical_generation import (
    generate_llm_answer,
    get_medical_api_key,
    get_medical_base_url,
    get_medical_model,
)

from scripts.evaluation.run_agent_evaluation_1500 import (
    evaluate_answer_for_case,
    first_value,
    retrieval_hits,
    is_fatal_provider_error,
    SUBJECT_ID_KEYS,
    HADM_ID_KEYS,
    EXPECTED_SOURCE_KEYS,
)


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "article_internal_holdout_800_final.jsonl"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"

EXPECTED_SHA256 = (
    "9229c2fe4591b6f073a4f31c9dcbbefcc212a8fe515a3eab5ea4bce6695daadf"
)

VALID_MODES = {
    "llm_only",
    "simple_rag",
    "no_agent_rag",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def select_smoke_cases(
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    seen = set()

    for case in cases:
        group = case.get("evaluation_group")

        if group not in seen:
            selected.append(case)
            seen.add(group)

    return selected


def generate_llm_only(question: str) -> str:
    api_key = get_medical_api_key()

    if not api_key:
        raise RuntimeError("No medical generation API key available.")

    client_kwargs = {
        "api_key": api_key,
    }

    base_url = get_medical_base_url()

    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    response = client.chat.completions.create(
        model=get_medical_model(),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical question-answering model. "
                    "Answer the user's question directly and concisely "
                    "using only the knowledge available in the language model. "
                    "Do not use retrieval, external tools, or external evidence. "
                    "Do not invent patient-specific facts that are not contained "
                    "in the question. If the requested information cannot be "
                    "determined from the question and your internal knowledge, "
                    "state that it cannot be determined. "
                    "Return a concise answer."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message.content

    if not answer:
        raise RuntimeError("LLM-only returned an empty response.")

    return answer.strip()


def retrieve_for_baseline(
    question: str,
    subject_id: str | None,
    hadm_id: str | None,
    use_reranker: bool,
) -> list[dict[str, Any]]:
    """
    Ablation-only retrieval.

    Simple RAG:
        BGE dense retrieval -> top 5

    No-Agent RAG:
        BGE dense retrieval -> ColBERT reranking -> top 5

    No router/category/gold-label filtering is used.
    Patient identifiers are used only when they are explicit
    task inputs for EHR cases.
    """

    client = retrieval_module.weaviate.connect_to_weaviate_cloud(
        cluster_url=retrieval_module.WEAVIATE_URL,
        auth_credentials=retrieval_module.AuthApiKey(
            retrieval_module.WEAVIATE_API_KEY
        ),
    )

    try:
        collection = client.collections.get("MedicalDocs")

        model = retrieval_module.get_embedding_model()

        query_for_embedding = (
            "Represent this sentence for searching relevant passages: "
            + question
        )

        query_vector = model.encode(
            query_for_embedding,
            normalize_embeddings=True,
        ).tolist()

        query_parameters = {
            "near_vector": query_vector,
            "limit": 50,
            "return_properties": [
                "content",
                "category",
                "document_source",
                "document_url",
            ],
        }

        # EHR patient/admission identifiers are legitimate task inputs.
        # Apply them BEFORE vector retrieval so unrelated patients do not
        # occupy the candidate pool.
        if subject_id or hadm_id:
            patient_filters = []

            if subject_id:
                patient_filters.append(
                    retrieval_module.Filter
                    .by_property("document_url")
                    .like(
                        f"*subject_id={str(subject_id).strip()}*"
                    )
                )

            if hadm_id:
                patient_filters.append(
                    retrieval_module.Filter
                    .by_property("document_url")
                    .like(
                        f"*hadm_id={str(hadm_id).strip()}*"
                    )
                )

            if len(patient_filters) == 1:
                query_parameters["filters"] = patient_filters[0]
            else:
                query_parameters["filters"] = (
                    retrieval_module.Filter.all_of(
                        patient_filters
                    )
                )

        response = collection.query.near_vector(
            **query_parameters
        )

        candidates: list[dict[str, Any]] = []
        seen_content: set[str] = set()

        for obj in response.objects:
            props = obj.properties or {}

            content = str(
                props.get("content") or ""
            ).strip()

            document_url = str(
                props.get("document_url") or ""
            ).strip()

            category = str(
                props.get("category") or ""
            ).strip()

            document_source = str(
                props.get("document_source") or ""
            ).strip()

            if not content:
                continue

            if retrieval_module.is_reference_like(content):
                continue

            # Defensive check after the database-level filter.
            document_url_lower = document_url.lower()

            if subject_id:
                expected_subject = (
                    f"subject_id={str(subject_id).strip().lower()}"
                )
                if expected_subject not in document_url_lower:
                    continue

            if hadm_id:
                expected_hadm = (
                    f"hadm_id={str(hadm_id).strip().lower()}"
                )
                if expected_hadm not in document_url_lower:
                    continue

            normalized_content = " ".join(
                content.lower().split()
            )

            if normalized_content in seen_content:
                continue

            seen_content.add(normalized_content)

            candidates.append(
                {
                    "content": content,
                    "category": category,
                    "document_source": document_source,
                    "document_url": document_url,
                    "relevance_score": 0,
                }
            )

        # Preserve Weaviate dense-retrieval order.
        if not use_reranker:
            return candidates[:5]

        if not candidates:
            return []

        # No-Agent RAG adds only ColBERT reranking.
        return rerank_candidates(
            query=question,
            candidates=candidates,
            top_k=5,
        )

    finally:
        client.close()


def run_baseline(
    mode: str,
    case: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    question = str(case.get("question", "")).strip()

    subject_id = first_value(
        case,
        SUBJECT_ID_KEYS,
    )

    hadm_id = first_value(
        case,
        HADM_ID_KEYS,
    )

    if mode == "llm_only":
        return generate_llm_only(question), []

    if mode == "simple_rag":
        docs = retrieve_for_baseline(
            question=question,
            subject_id=subject_id,
            hadm_id=hadm_id,
            use_reranker=False,
        )

    elif mode == "no_agent_rag":
        docs = retrieve_for_baseline(
            question=question,
            subject_id=subject_id,
            hadm_id=hadm_id,
            use_reranker=True,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    answer = generate_llm_answer(
        query=question,
        docs=docs,
        domain="medical",
    )

    return answer, docs


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed = set()

    with path.open(encoding="utf-8") as f:
        for line in f:
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
    elapsed: float,
    error: str,
) -> dict[str, Any]:
    retrieved_count = len(docs)

    metrics = evaluate_answer_for_case(
        case=case,
        answer=answer,
        retrieved_document_count=retrieved_count,
    )

    source_urls = [
        str(doc.get("document_url", ""))
        for doc in docs
        if doc.get("document_url")
    ]

    expected_source = first_value(
        case,
        EXPECTED_SOURCE_KEYS,
    )

    if mode == "llm_only":
        hit_1 = None
        hit_3 = None
        hit_5 = None
    else:
        hit_1, hit_3, hit_5 = retrieval_hits(
            source_urls,
            expected_source,
        )

    row = {
        "case_id": case.get("case_id"),
        "evaluation_group": case.get("evaluation_group"),
        "question_type": case.get("question_type"),
        "mode": mode,
        "question": case.get("question"),
        "expected_answer": case.get("expected_answer"),
        "answer": answer,
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

    row.update(metrics)

    return row


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    successful = [
        row
        for row in rows
        if row.get("request_success")
    ]

    f1_values = [
        float(row["answer_token_f1"])
        for row in rows
        if row.get("answer_token_f1") is not None
    ]

    score_values = [
        float(row["heuristic_answer_score"])
        for row in rows
        if row.get("heuristic_answer_score") is not None
    ]

    pass_values = [
        bool(row.get("heuristic_answer_pass"))
        for row in rows
        if row.get("heuristic_answer_pass") is not None
    ]

    times = [
        float(row["response_time_seconds"])
        for row in rows
        if row.get("response_time_seconds") is not None
    ]

    print()
    print("===== SUMMARY =====")
    print("CASES:", len(rows))
    print(
        "REQUEST SUCCESS:",
        round(len(successful) / len(rows), 4),
    )

    if f1_values:
        print(
            "MEAN TOKEN F1:",
            round(mean(f1_values), 6),
        )

    if score_values:
        print(
            "MEAN HEURISTIC SCORE:",
            round(mean(score_values), 6),
        )

    if pass_values:
        print(
            "HEURISTIC PASS RATE:",
            round(
                sum(pass_values) / len(pass_values),
                4,
            ),
        )

    if times:
        print(
            "MEAN RESPONSE TIME:",
            round(mean(times), 6),
        )


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
        raise SystemExit(
            f"ERROR: input file not found: {INPUT_PATH}"
        )

    actual_sha = file_sha256(INPUT_PATH)

    if actual_sha != EXPECTED_SHA256:
        raise SystemExit(
            "ERROR: Ablation sample hash changed.\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Actual:   {actual_sha}"
        )

    cases = load_jsonl(INPUT_PATH)

    if len(cases) != 800:
        raise SystemExit(
            f"ERROR: expected 800 cases, found {len(cases)}"
        )

    if args.smoke:
        cases = select_smoke_cases(cases)

    if args.group:
        cases = [c for c in cases if c.get("evaluation_group") == args.group]

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = "smoke4" if args.smoke else "800"

    output_path = (
        RESULTS_DIR
        / f"article800_baseline_{args.mode}_{suffix}_predictions.jsonl"
    )


    if output_path.exists() and not args.resume:
        raise SystemExit(
            f"STOP: Output already exists: {output_path}\\n"
            "Use --resume to continue without overwriting it."
        )

    completed_ids = (
        load_completed_ids(output_path)
        if args.resume
        else set()
    )

    pending = [
        case
        for case in cases
        if str(case.get("case_id")) not in completed_ids
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

    for index, case in enumerate(
        pending,
        start=len(completed_ids) + 1,
    ):
        case_id = str(case.get("case_id"))
        started = time.perf_counter()

        answer = ""
        docs: list[dict[str, Any]] = []
        error = ""

        try:
            answer, docs = run_baseline(
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
            elapsed=elapsed,
            error=error,
        )

        with output_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

        print(
            f"[{index}/{len(cases)}] "
            f"{case_id} | "
            f"mode={args.mode} | "
            f"docs={len(docs)} | "
            f"score={row.get('heuristic_answer_score')} | "
            f"pass={row.get('heuristic_answer_pass')} | "
            f"time={elapsed:.3f}s"
        )

    all_rows = load_jsonl(output_path)

    print_summary(all_rows)


if __name__ == "__main__":
    main()
