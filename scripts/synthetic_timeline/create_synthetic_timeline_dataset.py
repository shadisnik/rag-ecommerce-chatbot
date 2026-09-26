import json
import random
from pathlib import Path


random.seed(42)

PATIENT_COUNT = 50
TIMELINE_DAYS = 90

OUTPUT_DIR = Path("data/synthetic_patient_timeline")
OUTPUT_FILE = OUTPUT_DIR / "synthetic_pediatric_oncology_timeline.jsonl"


CANCER_PROFILES = [
    {
        "cancer_type": "Acute Lymphoblastic Leukemia",
        "treatment_phases": ["induction chemotherapy", "consolidation therapy", "maintenance therapy"],
        "base_medications": ["vincristine", "corticosteroid", "asparaginase"],
    },
    {
        "cancer_type": "Acute Myeloid Leukemia",
        "treatment_phases": ["induction chemotherapy", "post-chemotherapy monitoring", "chemotherapy recovery monitoring"],
        "base_medications": ["antifungal prophylaxis", "antiemetic", "antibacterial prophylaxis"],
    },
    {
        "cancer_type": "Neuroblastoma",
        "treatment_phases": ["chemotherapy cycle monitoring", "post-chemotherapy monitoring", "follow-up after chemotherapy"],
        "base_medications": ["antiemetic", "pain medication", "supportive care medication"],
    },
    {
        "cancer_type": "Osteosarcoma",
        "treatment_phases": ["chemotherapy cycle monitoring", "post-chemotherapy monitoring", "follow-up after chemotherapy"],
        "base_medications": ["antiemetic", "pain medication", "hydration support"],
    },
    {
        "cancer_type": "Wilms Tumor",
        "treatment_phases": ["post-surgery chemotherapy monitoring", "chemotherapy cycle monitoring", "follow-up after chemotherapy"],
        "base_medications": ["antiemetic", "pain medication", "supportive care medication"],
    },
]


LOW_SYMPTOMS = [
    ["mild fatigue"],
    ["reduced appetite"],
    ["mild nausea"],
    ["mild pain"],
    ["reduced activity"],
]

MEDIUM_SYMPTOMS = [
    ["nausea", "fatigue"],
    ["vomiting", "poor oral intake"],
    ["mouth sore", "reduced appetite"],
    ["pain", "reduced activity"],
    ["fatigue", "poor appetite"],
]

HIGH_SYMPTOMS = [
    ["fever", "chills", "weakness"],
    ["fever", "poor oral intake"],
    ["fever", "mouth sore", "weakness"],
    ["fever", "vomiting", "fatigue"],
]


def create_patients():
    patients = []

    for i in range(1, PATIENT_COUNT + 1):
        profile = random.choice(CANCER_PROFILES)

        patient = {
            "patient_id": f"PONC-{i:03d}",
            "age": random.randint(3, 17),
            "sex": random.choice(["female", "male"]),
            "cancer_type": profile["cancer_type"],
            "treatment_phase": random.choice(profile["treatment_phases"]),
            "base_medications": profile["base_medications"],
        }

        patients.append(patient)

    return patients


def choose_daily_pattern(day):
    """
    Most days are low or medium risk.
    A smaller number of days become high risk to simulate urgent monitoring events.
    """

    if day % 21 == 0:
        return "high"

    if day % 14 == 0:
        return random.choice(["medium", "high"])

    if day % 7 == 0:
        return "medium"

    return random.choices(
        ["low", "medium", "high"],
        weights=[70, 25, 5],
        k=1,
    )[0]


def make_symptoms(pattern):
    if pattern == "high":
        return random.choice(HIGH_SYMPTOMS)

    if pattern == "medium":
        return random.choice(MEDIUM_SYMPTOMS)

    return random.choice(LOW_SYMPTOMS)


def make_temperature(pattern):
    if pattern == "high":
        return round(random.uniform(38.1, 39.2), 1)

    if pattern == "medium":
        return round(random.uniform(37.3, 37.9), 1)

    return round(random.uniform(36.5, 37.2), 1)


def make_labs(pattern):
    if pattern == "high":
        return {
            "wbc": "very low",
            "anc": "very low",
            "hemoglobin": random.choice(["low", "slightly low"]),
            "platelets": random.choice(["low", "very low"]),
        }

    if pattern == "medium":
        return {
            "wbc": random.choice(["low", "slightly low"]),
            "anc": random.choice(["low", "moderately low"]),
            "hemoglobin": random.choice(["stable", "slightly low"]),
            "platelets": random.choice(["low", "slightly low"]),
        }

    return {
        "wbc": random.choice(["normal", "slightly low"]),
        "anc": random.choice(["normal", "moderately low"]),
        "hemoglobin": random.choice(["stable", "slightly low"]),
        "platelets": random.choice(["normal", "slightly low"]),
    }


def classify_risk(record):
    symptoms = record["symptoms"]
    temperature = record["body_temperature_celsius"]
    anc_status = record["lab_results"]["anc"]
    treatment_phase = record["treatment_phase"].lower()

    if temperature >= 38.0 and "chemotherapy" in treatment_phase:
        return (
            "high",
            "Contact the oncology care team immediately because fever during chemotherapy can be high risk.",
            "Rule 1: fever during chemotherapy",
        )

    if temperature >= 38.0 and anc_status == "very low":
        return (
            "high",
            "Seek urgent medical advice because fever with very low ANC may indicate possible infection risk.",
            "Rule 2: fever with very low ANC",
        )

    if "vomiting" in symptoms or "poor oral intake" in symptoms:
        return (
            "medium",
            "Monitor hydration and contact the care team if vomiting or poor intake continues.",
            "Rule 4: vomiting or poor oral intake",
        )

    if "mouth sore" in symptoms and "reduced appetite" in symptoms:
        return (
            "medium",
            "Monitor eating and drinking and report worsening mouth sores to the care team.",
            "Rule 5: mouth sores with poor intake",
        )

    if any(symptom in symptoms for symptom in ["nausea", "pain", "reduced activity"]):
        return (
            "medium",
            "Continue monitoring and contact the care team if symptoms worsen.",
            "Rule 3: mild or moderate symptoms without fever",
        )

    return (
        "low",
        "Continue daily monitoring and record symptoms.",
        "Rule 6: stable or mild symptoms",
    )


def make_caregiver_note(record):
    symptoms = ", ".join(record["symptoms"])
    temperature = record["body_temperature_celsius"]

    if record["risk_level"] == "high":
        return (
            f"Caregiver reports {symptoms}. Body temperature is {temperature} °C. "
            "The child appears weaker than usual."
        )

    if record["risk_level"] == "medium":
        return (
            f"Caregiver reports {symptoms}. The child is able to stay at home, "
            "but symptoms require closer monitoring."
        )

    return (
        f"Caregiver reports {symptoms}. The child is stable and no fever is reported."
    )


def build_content(record):
    symptoms = ", ".join(record["symptoms"])
    medications = ", ".join(record["medication_intake"])
    labs = ", ".join(
        f"{key}: {value}" for key, value in record["lab_results"].items()
    )

    return (
        f"Patient timeline record for {record['patient_id']}. "
        f"The patient is a {record['age']}-year-old {record['sex']} child with {record['cancer_type']}. "
        f"Treatment phase: {record['treatment_phase']}. "
        f"Timeline day: {record['day']}. "
        f"Reported symptoms: {symptoms}. "
        f"Body temperature: {record['body_temperature_celsius']} °C. "
        f"Medication intake: {medications}. "
        f"Laboratory results: {labs}. "
        f"Caregiver note: {record['caregiver_note']} "
        f"Risk level: {record['risk_level']}. "
        f"Risk rule applied: {record['risk_rule_applied']}. "
        f"Recommended action: {record['recommended_action']}"
    )


def generate_records():
    records = []
    patients = create_patients()

    for patient in patients:
        for day in range(1, TIMELINE_DAYS + 1):
            pattern = choose_daily_pattern(day)

            record = {
                "patient_id": patient["patient_id"],
                "age": patient["age"],
                "sex": patient["sex"],
                "cancer_type": patient["cancer_type"],
                "treatment_phase": patient["treatment_phase"],
                "day": day,
                "symptoms": make_symptoms(pattern),
                "body_temperature_celsius": make_temperature(pattern),
                "medication_intake": patient["base_medications"],
                "lab_results": make_labs(pattern),
                "category": "pediatric_patient_timeline",
                "document_source": "Semi-Synthetic Pediatric Oncology Treatment Timeline Dataset",
                "document_url": f"synthetic_timeline:{patient['patient_id']}:day_{day}",
                "source_design": (
                    "Cancer type and treatment concepts are informed by pediatric oncology resources. "
                    "Symptom fields are informed by pediatric oncology symptom-reporting concepts. "
                    "Daily timeline values are synthetic and rule-based."
                ),
            }

            risk_level, recommended_action, risk_rule = classify_risk(record)

            record["risk_level"] = risk_level
            record["recommended_action"] = recommended_action
            record["risk_rule_applied"] = risk_rule
            record["caregiver_note"] = make_caregiver_note(record)
            record["content"] = build_content(record)

            records.append(record)

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = generate_records()

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    risk_counts = {}
    cancer_counts = {}

    for record in records:
        risk = record["risk_level"]
        cancer = record["cancer_type"]

        risk_counts[risk] = risk_counts.get(risk, 0) + 1
        cancer_counts[cancer] = cancer_counts.get(cancer, 0) + 1

    print(f"Created: {OUTPUT_FILE}")
    print(f"Patients: {PATIENT_COUNT}")
    print(f"Timeline days per patient: {TIMELINE_DAYS}")
    print(f"Total records: {len(records)}")
    print("Risk distribution:", risk_counts)
    print("Cancer type distribution:", cancer_counts)


if __name__ == "__main__":
    main()
