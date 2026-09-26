import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

BOOTSTRAP_SEED = 42
BOOTSTRAP_SAMPLES = 2000

# Fixed before final evaluation for reproducibility.
BERTSCORE_MODEL = "roberta-large"
BERTSCORE_LANGUAGE = "en"

TEXTUAL_QA_GROUPS = {
    "ehr_clinical_qa",
    "pediatric_oncology_qa",
}


def clean_space(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalize_text(value: Any) -> str:
    return clean_space(value).lower()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


# ---------------------------------------------------------
# MRR
# ---------------------------------------------------------

def reciprocal_rank(row: dict[str, Any]) -> Optional[float]:
    expected = clean_space(row.get("expected_source"))

    if not expected:
        return None

    source_urls = row.get("source_urls", [])

    if not isinstance(source_urls, list):
        return None

    for rank, actual in enumerate(source_urls, start=1):
        if source_match(str(actual), expected):
            return 1.0 / rank

    return 0.0


# ---------------------------------------------------------
# Routing Macro-F1 + Confusion Matrix
# ---------------------------------------------------------

def route_macro_f1(
    rows: list[dict[str, Any]],
) -> tuple[Optional[float], dict[str, Any], dict[str, Any]]:

    valid_rows = [
        row for row in rows
        if clean_space(row.get("expected_route"))
    ]

    labels = sorted({
        clean_space(row["expected_route"])
        for row in valid_rows
    })

    if not labels:
        return None, {}, {}

    confusion: dict[str, dict[str, int]] = {
        label: defaultdict(int)
        for label in labels
    }

    for row in valid_rows:
        expected = clean_space(row.get("expected_route"))
        predicted = clean_space(row.get("predicted_route"))

        if not predicted:
            predicted = "__missing__"

        confusion[expected][predicted] += 1

    per_route = {}
    f1_values = []

    for label in labels:
        tp = sum(
            1 for row in valid_rows
            if clean_space(row.get("expected_route")) == label
            and clean_space(row.get("predicted_route")) == label
        )

        fp = sum(
            1 for row in valid_rows
            if clean_space(row.get("expected_route")) != label
            and clean_space(row.get("predicted_route")) == label
        )

        fn = sum(
            1 for row in valid_rows
            if clean_space(row.get("expected_route")) == label
            and clean_space(row.get("predicted_route")) != label
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        per_route[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

        f1_values.append(f1)

    macro_f1 = statistics.mean(f1_values)

    confusion_plain = {
        expected: dict(predictions)
        for expected, predictions in confusion.items()
    }

    return macro_f1, per_route, confusion_plain


# ---------------------------------------------------------
# 95% confidence intervals
# ---------------------------------------------------------


def evaluation_cluster_id(
    row: dict[str, Any],
) -> str:
    """
    Cluster repeated questions from the same patient together.
    EHR: subject_id
    Timeline: expected_patient_id
    Other groups: case_id (independent case)
    """
    group = clean_space(
        row.get("evaluation_group")
    )

    if group == "ehr_clinical_qa":
        value = row.get("subject_id")

    elif group in {
        "timeline_day_specific",
        "timeline_summary",
    }:
        value = row.get("expected_patient_id")

    else:
        value = row.get("case_id")

    if value is None:
        value = (
            row.get("case_id")
            or row.get("source_signature")
            or row.get("question")
        )

    return f"{group}:{value}"


def cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    statistic,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> Optional[dict[str, float]]:

    if not rows:
        return None

    estimate = statistic(rows)

    if estimate is None:
        return None

    clusters = defaultdict(list)

    for row in rows:
        clusters[
            evaluation_cluster_id(row)
        ].append(row)

    cluster_keys = list(clusters)
    n_clusters = len(cluster_keys)

    if n_clusters == 1:
        return {
            "estimate": float(estimate),
            "lower": float(estimate),
            "upper": float(estimate),
            "n": len(rows),
            "clusters": n_clusters,
        }

    rng = random.Random(seed)
    bootstrap_values = []

    for _ in range(samples):
        sample_rows = []

        for _ in range(n_clusters):
            key = cluster_keys[
                rng.randrange(n_clusters)
            ]
            sample_rows.extend(clusters[key])

        value = statistic(sample_rows)

        if value is not None:
            bootstrap_values.append(float(value))

    if not bootstrap_values:
        return None

    bootstrap_values.sort()

    lower_index = int(
        0.025 * (len(bootstrap_values) - 1)
    )
    upper_index = int(
        0.975 * (len(bootstrap_values) - 1)
    )

    return {
        "estimate": float(estimate),
        "lower": bootstrap_values[lower_index],
        "upper": bootstrap_values[upper_index],
        "n": len(rows),
        "clusters": n_clusters,
    }


def wilson_ci(
    successes: int,
    total: int,
    z: float = 1.96,
) -> Optional[dict[str, float]]:

    if total == 0:
        return None

    p = successes / total
    denominator = 1 + (z * z / total)

    centre = (
        p + z * z / (2 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            p * (1 - p) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )

    return {
        "estimate": p,
        "lower": max(0.0, centre - margin),
        "upper": min(1.0, centre + margin),
        "n": total,
    }



def boolean_ci(
    rows: list[dict[str, Any]],
    key: str,
) -> Optional[dict[str, float]]:

    valid = [
        row
        for row in rows
        if isinstance(row.get(key), bool)
    ]

    if not valid:
        return None

    def statistic(
        sample_rows: list[dict[str, Any]],
    ) -> float:
        return (
            sum(
                row.get(key) is True
                for row in sample_rows
            )
            / len(sample_rows)
        )

    return cluster_bootstrap_ci(
        valid,
        statistic,
    )

def bootstrap_mean_ci(
    values: list[float],
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> Optional[dict[str, float]]:

    if not values:
        return None

    estimate = statistics.mean(values)

    if len(values) == 1:
        return {
            "estimate": estimate,
            "lower": estimate,
            "upper": estimate,
            "n": 1,
        }

    rng = random.Random(seed)
    n = len(values)

    bootstrap_means = []

    for _ in range(samples):
        sample = [
            values[rng.randrange(n)]
            for _ in range(n)
        ]

        bootstrap_means.append(
            statistics.mean(sample)
        )

    bootstrap_means.sort()

    lower_index = int(
        0.025 * (samples - 1)
    )

    upper_index = int(
        0.975 * (samples - 1)
    )

    return {
        "estimate": estimate,
        "lower": bootstrap_means[lower_index],
        "upper": bootstrap_means[upper_index],
        "n": n,
    }



def bootstrap_macro_f1_ci(
    rows: list[dict[str, Any]],
) -> Optional[dict[str, float]]:

    valid = [
        row for row in rows
        if clean_space(
            row.get("expected_route")
        )
    ]

    if not valid:
        return None

    def statistic(
        sample_rows: list[dict[str, Any]],
    ) -> Optional[float]:

        score, _, _ = route_macro_f1(
            sample_rows
        )
        return score

    return cluster_bootstrap_ci(
        valid,
        statistic,
    )

def expected_to_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(
            clean_space(item)
            for item in value
        )

    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {clean_space(val)}"
            for key, val in value.items()
        )

    return clean_space(value)


def add_bertscore(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:

    from bert_score import score
    import torch

    eligible_indices = []

    candidates = []
    references = []

    for index, row in enumerate(rows):
        group = clean_space(
            row.get("evaluation_group")
        )

        # BERTScore is used only for open-ended textual QA.
        if group not in TEXTUAL_QA_GROUPS:
            row["bert_score_f1"] = None
            continue

        answer = clean_space(row.get("answer"))
        reference = expected_to_text(
            row.get("expected_answer")
        )

        # Report BERTScore only for successfully generated
        # textual answers; request failures remain captured
        # separately by Request Success Rate.
        if (
            row.get("request_success") is not True
            or not answer
            or not reference
        ):
            row["bert_score_f1"] = None
            continue

        eligible_indices.append(index)
        candidates.append(answer)
        references.append(reference)

    if not candidates:
        return {
            "model": BERTSCORE_MODEL,
            "language": BERTSCORE_LANGUAGE,
            "rescale_with_baseline": True,
            "device": None,
            "scored_cases": 0,
            "mean_f1": None,
            "f1_95ci": None,
        }

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    _, _, f1_scores = score(
        candidates,
        references,
        model_type=BERTSCORE_MODEL,
        lang=BERTSCORE_LANGUAGE,
        rescale_with_baseline=True,
        device=device,
        verbose=True,
    )

    values = [
        float(value)
        for value in f1_scores.cpu().tolist()
    ]

    for index, value in zip(
        eligible_indices,
        values,
    ):
        rows[index]["bert_score_f1"] = value

    return {
        "model": BERTSCORE_MODEL,
        "language": BERTSCORE_LANGUAGE,
        "rescale_with_baseline": True,
        "device": device,
        "scored_cases": len(values),
        "mean_f1": statistics.mean(values),
        "f1_95ci": numeric_ci(rows, "bert_score_f1"),
    }


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

def numeric_values(
    rows: list[dict[str, Any]],
    key: str,
) -> list[float]:

    return [
        float(row[key])
        for row in rows
        if isinstance(
            row.get(key),
            (int, float),
        )
        and not isinstance(
            row.get(key),
            bool,
        )
    ]



def numeric_ci(
    rows: list[dict[str, Any]],
    key: str,
) -> Optional[dict[str, float]]:

    valid = [
        row
        for row in rows
        if isinstance(
            row.get(key),
            (int, float),
        )
        and not isinstance(
            row.get(key),
            bool,
        )
    ]

    if not valid:
        return None

    def statistic(
        sample_rows: list[dict[str, Any]],
    ) -> float:
        return statistics.mean(
            float(row[key])
            for row in sample_rows
        )

    return cluster_bootstrap_ci(
        valid,
        statistic,
    )


def summarize_group(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:

    mrr_values = [
        float(row["retrieval_reciprocal_rank"])
        for row in rows
        if isinstance(
            row.get("retrieval_reciprocal_rank"),
            (int, float),
        )
    ]

    token_f1_values = numeric_values(
        rows,
        "answer_token_f1",
    )

    bert_values = numeric_values(
        rows,
        "bert_score_f1",
    )

    return {
        "cases": len(rows),

        "request_success_95ci":
            boolean_ci(rows, "request_success"),

        "route_accuracy_95ci":
            boolean_ci(rows, "route_correct"),

        "category_accuracy_95ci":
            boolean_ci(rows, "category_correct"),

        "hit_at_1_95ci":
            boolean_ci(rows, "retrieval_hit_at_1"),

        "hit_at_3_95ci":
            boolean_ci(rows, "retrieval_hit_at_3"),

        "hit_at_5_95ci":
            boolean_ci(rows, "retrieval_hit_at_5"),

        "verification_success_95ci":
            boolean_ci(rows, "verified"),

        "heuristic_answer_pass_95ci":
            boolean_ci(
                rows,
                "heuristic_answer_pass",
            ),

        "overall_automatic_pass_95ci":
            boolean_ci(
                rows,
                "overall_automatic_pass",
            ),

        "mrr":
            statistics.mean(mrr_values)
            if mrr_values
            else None,

        "mrr_95ci": numeric_ci(
            rows,
            "retrieval_reciprocal_rank",
        ),

        "mean_token_f1":
            statistics.mean(token_f1_values)
            if token_f1_values
            else None,

        "token_f1_95ci": numeric_ci(
            rows,
            "answer_token_f1",
        ),

        "mean_bert_score_f1":
            statistics.mean(bert_values)
            if bert_values
            else None,

        "bert_score_f1_95ci": numeric_ci(
            rows,
            "bert_score_f1",
        ),

        "bert_score_cases":
            len(bert_values),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Evaluation result JSONL file",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Metrics JSON output path",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(
            input_path.stem
            + "_article_metrics.json"
        )
    )

    enriched_path = input_path.with_name(
        input_path.stem
        + "_extended_metrics.jsonl"
    )

    rows = load_jsonl(input_path)

    # Per-case MRR
    for row in rows:
        row["retrieval_reciprocal_rank"] = (
            reciprocal_rank(row)
        )

    # BERTScore
    bertscore_summary = add_bertscore(rows)

    # Routing metrics
    macro_f1, per_route, confusion = (
        route_macro_f1(rows)
    )

    grouped = defaultdict(list)

    for row in rows:
        grouped[
            clean_space(
                row.get("evaluation_group")
            )
        ].append(row)

    metrics = {
        "evaluation_cases": len(rows),

        "predefined_metric_configuration": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_cluster_definition": "subject_id for EHR; expected_patient_id for timeline; case_id for independent NCI cases",
            "proportion_ci_method": "cluster percentile bootstrap 95% CI",
            "continuous_metric_ci_method":
                "cluster percentile bootstrap 95% CI",
            "bert_score_model": BERTSCORE_MODEL,
            "bert_score_language":
                BERTSCORE_LANGUAGE,
            "bert_score_rescale_with_baseline":
                True,
            "bert_score_scope":
                "successful EHR and pediatric-oncology textual QA responses only",
        },

        "routing": {
            "macro_f1": macro_f1,
            "macro_f1_95ci":
                bootstrap_macro_f1_ci(rows),
            "per_route": per_route,
            "confusion_matrix": confusion,
        },

        "bertscore": bertscore_summary,

        "overall": summarize_group(rows),

        "by_evaluation_group": {
            group: summarize_group(group_rows)
            for group, group_rows
            in sorted(grouped.items())
            if group
        },
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_jsonl(
        enriched_path,
        rows,
    )

    print("DONE")
    print("Cases:", len(rows))
    print("Metrics:", output_path)
    print("Extended rows:", enriched_path)


if __name__ == "__main__":
    main()
