# Data

The datasets used by this project are not distributed directly with the repository.
This directory is excluded from version control except for this README.

The instructions below describe the data sources used in the study and how they
can be obtained or regenerated.

## Knowledge-base sources

### NCI PDQ Pediatric Cancer Treatment Summaries

Pediatric oncology knowledge was derived from the National Cancer Institute
(NCI) PDQ Pediatric Cancer Treatment Summaries, using the Health Professional
versions.

Source:
National Cancer Institute, PDQ Pediatric Cancer Treatment Summaries.

The downloaded documents were converted to text and divided into overlapping
chunks before indexing in Weaviate. The preprocessing configuration used in the
reported experiments is documented in the main project README and source code.

### NCI Drug Dictionary

Drug and therapeutic-agent information was obtained from the National Cancer
Institute Drug Dictionary.

Source:
National Cancer Institute, NCI Drug Dictionary.

The repository does not redistribute a full copy of the NCI resource.

### MedQuAD

General medical question-answer material was obtained from MedQuAD:

Asma Ben Abacha and Dina Demner-Fushman,
"A Question-Entailment Approach to Question Answering,"
BMC Bioinformatics, 2019.

Repository:
`abachaa/MedQuAD`

MedQuAD is distributed by its authors under the Creative Commons Attribution
4.0 International license. Users should obtain the original dataset from the
upstream repository and comply with its license and attribution requirements.

### EHR-DS-QA

Clinical EHR question-answer examples were obtained from:

Konstantin Kotschenreuther,
"EHR-DS-QA: A Synthetic QA Dataset Derived from Medical Discharge Summaries
for Enhanced Medical Information Retrieval Systems,"
PhysioNet, version 1.0.0, 2024.

DOI: `10.13026/25fx-f706`

EHR-DS-QA is a credentialed-access PhysioNet resource. It is therefore not
redistributed in this repository.

Researchers wishing to reproduce this part of the study must obtain their own
authorized access through PhysioNet and complete the required training and data
access procedures.

Only data obtained under the researcher's own PhysioNet authorization should be
used.

## Pediatric oncology timeline

The longitudinal pediatric oncology timeline used in this project is
semi-synthetic and rule-based.

Relevant project files include:

- `synthetic_patient_timeline/timeline_schema.md`
- `synthetic_patient_timeline/timeline_rulebook.md`

Generation utilities are available under:

`../scripts/synthetic_timeline/`

The timeline should be regenerated locally rather than interpreted as real
patient data. It does not represent clinical records from actual pediatric
patients.

## Internal evaluation data

The final internal evaluation contains four task groups:

- EHR clinical question answering
- Pediatric oncology question answering
- Day-specific timeline question answering
- Longitudinal timeline summarization

The paper reports 200 questions from each group, for a total of 800 internal
evaluation questions.

The local working directory contains intermediate, development, audit, and
historical evaluation files in addition to the final evaluation set. These
intermediate files are intentionally excluded from version control.

Evaluation-generation and preparation utilities are located under:

`../scripts/evaluation/`

and

`../scripts/synthetic_timeline/`

## External evaluation data

The external evaluation was constructed from two independent medical QA
resources:

### TREC 2017 LiveQA Medical Task

The medical questions and reference answers are available from the official
TREC 2017 LiveQA Medical Task resources maintained by NIST.

The original source data should be obtained from the official TREC/NIST
distribution rather than from this repository.

### MedicationQA

MedicationQA is the medication question-answering corpus introduced in:

Asma Ben Abacha et al.,
"Bridging the Gap between Consumers' Medication Questions and Trusted Answers,"
MEDINFO 2019.

Repository:
`abachaa/Medication_QA_MedInfo2019`

The upstream dataset is distributed under the Creative Commons Attribution 4.0
International license.

The study's cleaned external evaluation set was produced locally from the source
datasets using the evaluation-preparation scripts. Intermediate filtering,
duplicate handling, and audit files are not required for running the released
code.

## Local directory structure

After obtaining or generating the required resources, data are stored locally
under subdirectories such as:

```text
data/
├── evaluation/
├── external_evaluation/
├── nci_thesaurus/
└── synthetic_patient_timeline/

## Data privacy and redistribution

No API credentials, environment files, private keys, or authentication tokens
should be placed in this directory.

Credentialed PhysioNet data must not be committed to the public repository.

Researchers are responsible for complying with the licenses, access agreements,
and attribution requirements of each upstream dataset.
