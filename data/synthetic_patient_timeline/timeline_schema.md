# Synthetic Pediatric Oncology Patient Timeline Dataset - Schema

## Purpose

This dataset is designed for a prototype multi-agent RAG chatbot for monitoring treatment progress in pediatric oncology patients.

The dataset is semi-synthetic. The structure of the dataset is informed by pediatric oncology knowledge sources and symptom-reporting instruments, but the individual patient timeline records are synthetic and do not represent real patients.

## Data Sources Used for Schema Design

1. NCI PDQ Pediatric Cancer Treatment Summaries
Used to define pediatric cancer types and treatment-related concepts.

2. Ped-PRO-CTCAE / Ped-PRO-CTCAE Caregiver
Used to guide symptom-related fields such as fatigue, nausea, pain, vomiting, appetite change, and other patient/caregiver-reported symptoms.

3. NCI TARGET pediatric cancer resources
Used to guide the selection of pediatric cancer types such as ALL, AML, neuroblastoma, osteosarcoma, and Wilms tumor.

## Dataset Columns

patient_id:
Synthetic patient identifier, for example PONC-001.

age:
Synthetic patient age within a pediatric range.

cancer_type:
Pediatric cancer type, such as Acute Lymphoblastic Leukemia, Acute Myeloid Leukemia, Neuroblastoma, Osteosarcoma, or Wilms Tumor.

treatment_phase:
Current phase of treatment or monitoring, such as induction chemotherapy, maintenance therapy, post-chemotherapy monitoring, or follow-up.

day:
Timeline day for the patient record.

symptoms:
Caregiver-reported or patient-reported symptoms, such as fever, nausea, fatigue, pain, vomiting, mouth sore, poor appetite, or weakness.

body_temperature_celsius:
Home-reported body temperature in Celsius.

medication_intake:
Reported medication intake or supportive medication use.

lab_results:
Structured laboratory-related status, such as WBC, ANC, hemoglobin, and platelets.

caregiver_note:
Synthetic short note describing the caregiver's daily observation.

risk_level:
Conservative risk category: low, medium, or high.

recommended_action:
Suggested non-diagnostic action, such as continue monitoring, contact care team, or seek urgent medical advice.

category:
Fixed category used for retrieval: pediatric_patient_timeline.

document_source:
Fixed source label: Synthetic Pediatric Oncology Patient Timeline Dataset.

document_url:
Synthetic document identifier, for example synthetic_timeline:PONC-001:day_7.

content:
Text version of the structured record for RAG retrieval.

## Important Limitation

This dataset is synthetic and must not be used for real clinical decision-making. It is only used to evaluate the design and behavior of a prototype system.
