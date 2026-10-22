#!/usr/bin/env python3
"""Reproduce Table 6 (800-question baseline/ablation) from existing predictions.

No LLM/API calls, no evaluations rerun, and no source predictions modified.

Run from the repository root:
    python scripts/evaluation/reproduce_table6_statistics.py

Requires numpy (already in the project's requirements). Optional CLI overrides:
    --basic PATH --routed PATH --rerank PATH --full PATH --output PATH

Paired 95% intervals: percentile stratified cluster bootstrap (25,000 draws),
NOT adjusted for multiple comparisons.
Paired p values: exact two-sided cluster sign-flip test, then Holm adjustment.

Cases are aligned by case_id. EHR questions cluster by subject_id; two
longitudinal groups share the same synthetic patient identifier; oncology
questions are separate clusters by case_id. These 447 clusters are resampled
within three strata (EHR / oncology / timeline).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DEFAULTS = {
    "basic": "results/evaluation/article800_baseline_simple_rag_800_predictions.jsonl",
    "routed": "results/evaluation/article800_routed_routed_rag_800_predictions.jsonl",
    "rerank": "results/evaluation/article800_routed_routed_rag_rerank_800_predictions.jsonl",
    "full": "results/evaluation/article_internal_800_full_routerfix_v2_predictions.csv",
}
ORDER = ("basic", "routed", "rerank", "full")
NAMES = {
    "basic": "Basic RAG", "routed": "Routed RAG",
    "rerank": "Routed RAG + ColBERT", "full": "Full System",
}
TIMELINE_ID = re.compile(r"\bPONC[-_]\d+\b", re.I)


def truthy(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if str(value).strip().lower() in ("true", "1", "yes"):
        return True
    if str(value).strip().lower() in ("false", "0", "no", "", "none"):
        return False
    raise ValueError(f"Unexpected Boolean value: {value!r}")


def load_rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() == ".csv" or handle.readline().startswith("case_id,"):
            handle.seek(0)
            rows = list(csv.DictReader(handle))
        else:
            handle.seek(0)
            rows = [json.loads(line) for line in handle if line.strip()]
    by_id = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id or case_id in by_id:
            raise ValueError(f"Missing/duplicate case_id in {path}: {case_id!r}")
        by_id[case_id] = row
    return by_id


def cluster_and_stratum(row):
    group = row["evaluation_group"]
    case_id = row["case_id"]
    if group == "ehr_clinical_qa":
        subject_id = str(row.get("subject_id") or "").strip()
        if not subject_id:
            raise ValueError(f"Missing EHR subject_id for {case_id}")
        return "ehr", ("ehr", subject_id)
    if group in ("timeline_day_specific", "timeline_summary"):
        match = TIMELINE_ID.search(str(row.get("question", "")))
        if not match:
            raise ValueError(f"Missing timeline patient ID for {case_id}")
        return "timeline", ("timeline", match.group().upper().replace("_", "-"))
    if group == "pediatric_oncology_qa":
        return "oncology", ("oncology", case_id)
    raise ValueError(f"Unexpected evaluation_group for {case_id}: {group}")


def exact_two_sided_signflip(diff_by_cluster):
    """Exact sign-flip tail probability by subset-sum DP (not Monte Carlo)."""
    magnitudes = [abs(x) for x in diff_by_cluster.values() if x]
    if not magnitudes:
        return 1.0
    observed = abs(sum(diff_by_cluster.values()))
    max_sum = sum(magnitudes)
    ways = [0] * (max_sum + 1)
    ways[0] = 1
    for w in magnitudes:
        for j in range(max_sum, w - 1, -1):
            ways[j] += ways[j - w]
    count_extreme = sum(n for j, n in enumerate(ways) if abs(max_sum - 2 * j) >= observed)
    return count_extreme / (2 ** len(magnitudes))


def holm_adjust(p_values):
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    results = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, p_values[i] * (m - rank)))
        results[i] = running
    return results


def canonical_reference(value):
    if isinstance(value, str):
        value = value.strip()
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            value = value.strip('"')
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ORDER:
        parser.add_argument(f"--{key}", default=DEFAULTS[key], help=f"Input file (default: {DEFAULTS[key]})")
    parser.add_argument("--output", default="results/evaluation/article800_paired_statistics_reproduced.json")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--replicates", type=int, default=25000)
    args = parser.parse_args()

    runs = {key: load_rows(getattr(args, key)) for key in ORDER}
    ids = set(runs["full"])
    if len(ids) != 800 or any(set(rows) != ids for rows in runs.values()):
        raise ValueError("All four configurations must contain the same 800 unique case_ids")
    for case_id in ids:
        metadata = {(str(run[case_id]["question"]).strip(),
                     canonical_reference(run[case_id]["expected_answer"]),
                     run[case_id]["evaluation_group"]) for run in runs.values()}
        if len(metadata) != 1:
            raise ValueError(f"Question/reference/group mismatch for {case_id}")

    clusters = {}
    strata = defaultdict(list)
    for case_id, row in runs["full"].items():
        stratum, key = cluster_and_stratum(row)
        clusters[case_id] = key
        if key not in strata[stratum]:
            strata[stratum].append(key)
    for keys in strata.values():
        keys.sort()
    sizes = Counter(clusters.values())
    if len(sizes) != 447:
        raise ValueError(f"Unexpected cluster count {len(sizes)} (expected 447)")

    estimates = {}
    raw_pvalues = []
    rng = np.random.default_rng(args.seed)
    for before, after in zip(ORDER[:-1], ORDER[1:]):
        delta_by_cluster = defaultdict(int)
        improved = regressed = 0
        for case_id in ids:
            difference = int(truthy(runs[after][case_id]["heuristic_answer_pass"])) - int(
                truthy(runs[before][case_id]["heuristic_answer_pass"]))
            delta_by_cluster[clusters[case_id]] += difference
            improved += difference == 1
            regressed += difference == -1
        raw_p = exact_two_sided_signflip(delta_by_cluster)
        raw_pvalues.append(raw_p)

        total_difference = np.zeros(args.replicates, dtype=np.float64)
        total_cases = np.zeros(args.replicates, dtype=np.float64)
        # Stratified sampling: resample whole patient-level clusters within strata.
        # A patient cluster can contain multiple questions, so sampled case counts vary.
        for stratum in ("ehr", "oncology", "timeline"):
            keys = strata[stratum]
            cluster_counts = np.asarray([sizes[k] for k in keys], dtype=np.int32)
            differences = np.asarray([delta_by_cluster[k] for k in keys], dtype=np.int32)
            choice = rng.integers(0, len(keys), size=(args.replicates, len(keys)))
            total_difference += differences[choice].sum(axis=1)
            total_cases += cluster_counts[choice].sum(axis=1)
        replicates = 100 * total_difference / total_cases
        lower, upper = np.percentile(replicates, [2.5, 97.5])
        label = f"{NAMES[before]} -> {NAMES[after]}"
        estimates[label] = {
            "n_cases": len(ids), "improved": improved, "regressed": regressed,
            "difference_percentage_points": round(100 * (improved - regressed) / len(ids), 6),
            "percentile_95_ci_unadjusted": [round(float(lower), 6), round(float(upper), 6)],
            "signflip_p_unadjusted": raw_p,
            "signflip_p_holm_adjusted": None,
        }
    adjusted = holm_adjust(raw_pvalues)
    for name, p in zip(estimates, adjusted):
        estimates[name]["signflip_p_holm_adjusted"] = p
    result = {
        "n_cases": len(ids), "n_clusters": len(sizes),
        "clusters_by_stratum": {s: len(v) for s, v in strata.items()},
        "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
        "methods": {
            "confidence_intervals": "stratified, paired cluster percentile bootstrap; unadjusted for multiple comparisons",
            "significance": "exact, two-sided cluster sign-flip test followed by Holm step-down correction",
        },
        "comparisons": estimates,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Cases: {len(ids)}; clusters: {len(sizes)}; wrote: {output_path}")
    for name, values in estimates.items():
        lo, hi = values["percentile_95_ci_unadjusted"]
        print(f"{name}: delta={values['difference_percentage_points']:.3f} pp; "
              f"95% CI [{lo:.3f}, {hi:.3f}]; "
              f"p={values['signflip_p_unadjusted']:.9g}; "
              f"Holm p={values['signflip_p_holm_adjusted']:.9g}")


if __name__ == "__main__":
    main()
