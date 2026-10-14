"""Recalculate only cluster-bootstrap CIs from existing extended evaluation rows.

Does not call Together, recompute BERTScore, modify predictions or overwrite
original metrics. Run from the medical-rag-chatbot repository root.
"""
import importlib.util
import json
import math
import re
from collections import defaultdict
from pathlib import Path

RUN = "article_internal_800_full_routerfix_v2"
RESULTS = Path("results/evaluation")
SCRIPT = Path("scripts/evaluation/calculate_article_metrics.py")
METRICS = RESULTS / f"{RUN}_article_metrics.json"
EXTENDED = RESULTS / f"{RUN}_predictions_extended_metrics.jsonl"
OUTPUT = RESULTS / f"{RUN}_article_metrics_cluster_corrected.json"
PATIENT_RE = re.compile(r"\bPONC[-_]\d+\b", re.IGNORECASE)


def main():
    for path in (SCRIPT, METRICS, EXTENDED):
        if not path.is_file():
            raise SystemExit(f"Required input missing: {path}")

    spec = importlib.util.spec_from_file_location("frozen_article_metrics", SCRIPT)
    calc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calc)

    with METRICS.open(encoding="utf-8") as f:
        original = json.load(f)
    rows = calc.load_jsonl(EXTENDED)
    if len(rows) != 800 or original.get("evaluation_cases") != 800:
        raise SystemExit("Expected exactly 800 rows in both the stored metrics and extended results.")
    if len({r.get("case_id") for r in rows}) != 800:
        raise SystemExit("Missing or duplicate case IDs.")
    if any("bert_score_f1" not in r or "retrieval_reciprocal_rank" not in r for r in rows):
        raise SystemExit("Use the existing *_predictions_extended_metrics.jsonl, not the raw predictions.")

    def corrected_cluster_id(row):
        group = calc.clean_space(row.get("evaluation_group"))
        if group == "ehr_clinical_qa":
            pid = calc.clean_space(row.get("subject_id"))
            if not pid:
                raise ValueError(f"Missing EHR subject_id: {row.get('case_id')}")
            return f"ehr:{pid}"
        if group in ("timeline_day_specific", "timeline_summary"):
            explicit = calc.clean_space(row.get("expected_patient_id"))
            source = PATIENT_RE.search(calc.clean_space(row.get("expected_source")))
            question = PATIENT_RE.search(calc.clean_space(row.get("question")))
            candidates = [v for v in (explicit, source.group(0) if source else None,
                                      question.group(0) if question else None) if v]
            candidates = [v.upper().replace("_", "-") for v in candidates]
            if not candidates or len(set(candidates)) != 1:
                raise ValueError(f"Missing/inconsistent timeline patient: {row.get('case_id')}: {candidates}")
            # Both timeline question groups involving one patient form ONE cluster.
            return f"timeline:{candidates[0]}"
        case_id = calc.clean_space(row.get("case_id"))
        if not case_id:
            raise ValueError("Missing independent case_id")
        return f"{group}:{case_id}"

    # Validate IDs before calculating any intervals.
    ids = [corrected_cluster_id(r) for r in rows]
    by_group = defaultdict(set)
    for r, cid in zip(rows, ids):
        by_group[r["evaluation_group"]].add(cid)
    print("Clusters by group:", dict(sorted((g, len(v)) for g, v in by_group.items())))
    print("Independent clusters overall:", len(set(ids)))
    if (len(by_group["timeline_day_specific"]) != 49
            or len(by_group["timeline_summary"]) != 200
            or len(set(ids)) != 447):
        raise SystemExit("Unexpected patient clustering; stopped without writing output.")

    calc.evaluation_cluster_id = corrected_cluster_id
    updated = json.loads(json.dumps(original))
    updated["predefined_metric_configuration"]["bootstrap_cluster_definition"] = (
        "EHR: subject_id; both timeline groups: shared PONC patient ID "
        "parsed/validated from question and expected_source when expected_patient_id is absent; "
        "NCI: independent case_id."
    )
    updated["predefined_metric_configuration"]["cluster_correction"] = (
        "CI-only correction using existing extended rows; original predictions and point estimates unchanged."
    )
    updated["routing"]["macro_f1_95ci"] = calc.bootstrap_macro_f1_ci(rows)
    updated["bertscore"]["f1_95ci"] = calc.numeric_ci(rows, "bert_score_f1")
    updated["overall"] = calc.summarize_group(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["evaluation_group"]].append(row)
    updated["by_evaluation_group"] = {
        key: calc.summarize_group(grouped[key]) for key in sorted(grouped)
    }

    def same(a, b):
        return ((a is None and b is None)
                or (isinstance(a, (int, float)) and isinstance(b, (int, float))
                    and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)))

    # Every originally reported estimate/mean must remain exactly (numerically) the same.
    for section in ("overall",):
        for key, before in original[section].items():
            after = updated[section][key]
            if isinstance(before, dict) and "estimate" in before:
                if not same(before["estimate"], after["estimate"]):
                    raise SystemExit(f"Point estimate changed: {section}.{key}")
            elif not isinstance(before, dict) and not same(before, after):
                raise SystemExit(f"Point estimate changed: {section}.{key}")
    for group in original["by_evaluation_group"]:
        for key, before in original["by_evaluation_group"][group].items():
            after = updated["by_evaluation_group"][group][key]
            if isinstance(before, dict) and "estimate" in before:
                if not same(before["estimate"], after["estimate"]):
                    raise SystemExit(f"Point estimate changed: {group}.{key}")
            elif not isinstance(before, dict) and not same(before, after):
                raise SystemExit(f"Point estimate changed: {group}.{key}")
    if not same(original["routing"]["macro_f1_95ci"]["estimate"],
                updated["routing"]["macro_f1_95ci"]["estimate"]):
        raise SystemExit("Routing Macro F1 estimate changed.")
    if not same(original["bertscore"]["f1_95ci"]["estimate"],
                updated["bertscore"]["f1_95ci"]["estimate"]):
        raise SystemExit("BERTScore estimate changed.")

    if OUTPUT.exists():
        raise SystemExit(f"Output exists; refusing to overwrite: {OUTPUT}")
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)
    print("PASS: all original point estimates unchanged.")
    print("Original metrics preserved:", METRICS)
    print("Corrected CI output:", OUTPUT)
    print("Overall pass 95% CI:", updated["overall"]["overall_automatic_pass_95ci"])


if __name__ == "__main__":
    main()
