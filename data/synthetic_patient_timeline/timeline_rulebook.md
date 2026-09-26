# Rulebook for Semi-Synthetic Pediatric Oncology Treatment Timeline Dataset

## Purpose

This rulebook explains how the semi-synthetic pediatric oncology treatment timeline records are generated.

The records do not represent real patients. However, the timeline events are designed to be medically plausible and source-informed.

The purpose of this dataset is to evaluate a prototype multi-agent RAG chatbot for treatment progress monitoring, not to validate a clinical decision-making system.

## General Design Principle

Each timeline record represents one daily observation for a pediatric oncology patient.

Each record may include:

- patient identifier
- cancer type
- treatment phase
- timeline day
- symptoms
- body temperature
- medication intake
- laboratory status
- caregiver note
- risk level
- recommended action

## Risk Level Rules

### Rule 1: Fever during chemotherapy

If the patient is in chemotherapy and body temperature is 38.0 °C or higher, the risk level should be high.

Recommended action:
Contact the oncology care team immediately.

Reason:
Fever during chemotherapy can indicate a potentially serious infection risk, especially when immune function is low.

### Rule 2: Fever with very low ANC

If the patient has fever and ANC is very low, the risk level should be high.

Recommended action:
Seek urgent medical advice or contact the oncology care team immediately.

Reason:
Fever with very low neutrophil count may indicate febrile neutropenia risk.

### Rule 3: Mild symptoms without fever

If the patient has mild fatigue, reduced appetite, or mild nausea, and body temperature is below 38.0 °C, the risk level can be low or medium.

Recommended action:
Continue monitoring and record symptoms.

### Rule 4: Vomiting or poor oral intake

If the patient has vomiting or poor oral intake, the risk level should be medium.

Recommended action:
Monitor hydration and contact the care team if symptoms continue or worsen.

### Rule 5: Mouth sores with poor intake

If the patient has mouth sores and reduced oral intake, the risk level should be medium.

Recommended action:
Monitor eating and drinking, and report worsening mouth sores to the care team.

### Rule 6: Stable follow-up condition

If the patient is in follow-up, has normal temperature, stable labs, and only mild fatigue, the risk level should be low.

Recommended action:
Continue routine follow-up and daily monitoring.

### Rule 7: Persistent symptoms over multiple days

If symptoms such as fatigue, poor appetite, pain, or nausea continue across several timeline days, the risk level can increase from low to medium.

Recommended action:
Share symptom trends with the care team during follow-up.

## Symptom Groups

The following symptom groups can be used in the dataset:

- fever
- chills
- fatigue
- nausea
- vomiting
- pain
- mouth sore
- poor appetite
- weakness
- reduced fluid intake

## Laboratory Monitoring Fields

The dataset can include simplified laboratory monitoring fields:

- WBC
- ANC
- hemoglobin
- platelets

For the prototype dataset, these values may be represented as categorical status values such as:

- normal
- slightly low
- low
- very low
- stable

## Treatment Phases

The dataset can include treatment phases such as:

- induction chemotherapy
- maintenance therapy
- post-chemotherapy monitoring
- chemotherapy cycle monitoring
- follow-up after chemotherapy

## Important Limitation

This rulebook is designed for prototype data generation.

The generated records are medically plausible examples, but they are not real patient records and must not be used for real clinical decision-making.

A future clinical version of the system would require real patient data, expert review, and clinical validation.
