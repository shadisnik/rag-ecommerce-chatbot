import json
import random
from pathlib import Path

from scripts.synthetic_timeline import create_synthetic_timeline_dataset as base


SEED = 44
START_PATIENT_ID = 101
PATIENT_COUNT = 150
TIMELINE_DAYS = 90

OUTPUT_FILE = Path(
    "data/synthetic_patient_timeline/"
    "article_holdout_pediatric_oncology_timeline_extra150.jsonl"
)

random.seed(SEED)


def create_patients():
    patients = []

    for i in range(
        START_PATIENT_ID,
        START_PATIENT_ID + PATIENT_COUNT,
    ):
        profile = random.choice(base.CANCER_PROFILES)

        patients.append({
            "patient_id": f"PONC-{i:03d}",
            "age": random.randint(3, 17),
            "sex": random.choice(["female", "male"]),
            "cancer_type": profile["cancer_type"],
            "treatment_phase": random.choice(
                profile["treatment_phases"]
            ),
            "base_medications": profile["base_medications"],
        })

    return patients


def generate_records():
    records = []

    for patient in create_patients():
        for day in range(1, TIMELINE_DAYS + 1):
            pattern = base.choose_daily_pattern(day)

            record = {
                "patient_id": patient["patient_id"],
                "age": patient["age"],
                "sex": patient["sex"],
                "cancer_type": patient["cancer_type"],
                "treatment_phase": patient["treatment_phase"],
                "day": day,
                "symptoms": base.make_symptoms(pattern),
                "body_temperature_celsius":
                    base.make_temperature(pattern),
                "medication_intake":
                    patient["base_medications"],
                "lab_results": base.make_labs(pattern),
                "category":
                    "pediatric_patient_timeline",
                "document_source":
                    "Semi-Synthetic Pediatric Oncology Treatment Timeline Dataset",
                "document_url":
                    f"synthetic_timeline:{patient['patient_id']}:day_{day}",
                "source_design": (
                    "Cancer type and treatment concepts are informed by pediatric oncology resources. "
                    "Symptom fields are informed by pediatric oncology symptom-reporting concepts. "
                    "Daily timeline values are synthetic and rule-based."
                ),
            }

            risk_level, recommended_action, risk_rule = (
                base.classify_risk(record)
            )

            record["risk_level"] = risk_level
            record["recommended_action"] = recommended_action
            record["risk_rule_applied"] = risk_rule
            record["caregiver_note"] = (
                base.make_caregiver_note(record)
            )
            record["content"] = base.build_content(record)

            records.append(record)

    return records


def main():
    records = generate_records()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    patients = sorted({
        x["patient_id"]
        for x in records
    })

    print("Created:", OUTPUT_FILE)
    print("Seed:", SEED)
    print("Patients:", len(patients))
    print("First patient:", patients[0])
    print("Last patient:", patients[-1])
    print("Records:", len(records))


if __name__ == "__main__":
    main()
