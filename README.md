# Route-Adaptive Multi-Agent RAG for Medical Question Answering

This repository contains the implementation and evaluation code for a research prototype that combines retrieval-augmented generation (RAG) with a route-adaptive multi-agent workflow for medical question answering.

The framework is designed to process heterogeneous information needs through five specialized routes:

- `general_medical_rag`
- `pediatric_oncology`
- `clinical_qa`
- `patient_timeline`
- `patient_timeline_summary`

The system integrates route-specific retrieval, ColBERTv2 reranking, evidence-sufficiency assessment, supplementary retrieval, answer generation, rule-based verification, and rule-based safety flagging.

> This repository implements a research prototype. It is not a clinically validated decision-support system and must not be used for patient-care decisions.

## System Overview

The framework uses different retrieval strategies depending on the structure of the information required by the query.

General medical, pediatric oncology, and clinical QA queries use semantic vector retrieval. Candidate documents are retrieved from Weaviate and reranked using ColBERTv2.

Patient timeline queries use structured retrieval based on patient identifiers and treatment day. Longitudinal summary queries aggregate the complete set of available daily records for the selected patient.

The main large language model used in the reported experiments was:

`meta-llama/Llama-3.3-70B-Instruct-Turbo`

served through Together AI.

Embedding model:

`BAAI/bge-small-en-v1.5`

Reranking model:

`colbert-ir/colbertv2.0`

Vector database:

Weaviate Cloud (`MedicalDocs` collection)

## Environment

The experiments were conducted using Python 3.12.

Key package versions:

```text
fastapi==0.141.1
uvicorn==0.52.1
streamlit==1.60.0
requests==2.34.2
python-dotenv==1.2.2

weaviate-client==4.22.0
sentence-transformers==5.6.1

torch==2.13.0
transformers==4.49.0
tokenizers==0.21.4
huggingface-hub==0.36.2

ragatouille==0.0.9.post2
psutil==7.2.2

langchain==0.1.20
langchain-core==0.1.53
langchain-community==0.0.38
langchain-text-splitters==0.0.2

numpy==1.26.4
pandas==3.0.5
pyarrow==24.0.0
protobuf==6.33.6
tqdm==4.70.0

openai==2.52.0
together==2.28.0
setuptools==83.0.0
```

## Installation

Create and activate a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Knowledge Sources

The system uses multiple medical information sources, including:

- NCI PDQ Pediatric Cancer Information Summaries
- NCI Drug Dictionary
- MedQuAD
- EHR-DS-QA from PhysioNet
- A rule-based semi-synthetic pediatric oncology treatment timeline

Some datasets cannot be redistributed directly because of their original access conditions or licenses. Instructions for obtaining and preparing these datasets are provided separately in the repository.

The semi-synthetic longitudinal data are intended for methodological evaluation only and do not represent real patient records.

## Evaluation

The paper reports an internal evaluation consisting of 800 questions across four groups:

- 200 EHR clinical QA questions
- 200 pediatric oncology questions
- 200 patient-day timeline questions
- 200 longitudinal timeline-summary questions

External evaluation uses:

- TREC 2017 LiveQA Medical
- MedicationQA

After preprocessing, the external evaluation set contains 721 questions.

A stratified sample of 150 external questions was additionally evaluated manually.

Evaluation, baseline, ablation, metric-calculation, and confidence-interval scripts are located under:

```text
scripts/evaluation/
```

## Experimental Configurations

Four configurations were evaluated:

1. Basic RAG
2. Routed RAG
3. Routed RAG + ColBERT
4. Full System

The Full System includes route selection, route-specific retrieval, ColBERT reranking where applicable, evidence-sufficiency assessment, supplementary retrieval, answer generation, and rule-based verification.

## Prompts and Agent Logic

The exact prompt templates used in the reported system are provided in:

- `prompts/router.txt`
- `prompts/evidence_sufficiency.txt`
- `prompts/generation.txt`

Query refinement is produced by the Evidence Sufficiency Agent and is not implemented as a separate LLM call.

The Verification and Safety Triage components are deterministic and rule-based rather than prompt-based. Their operational rules are documented in:

- `prompts/operational_rules.md`

## Data and Privacy

No real pediatric patient records are included in the semi-synthetic timeline dataset.

Datasets subject to access or redistribution restrictions must be obtained from their original providers and used according to their respective licenses and access requirements.

API keys, credentials, local environment files, and private configuration values are not included in this repository.

Detailed dataset acquisition, preparation, access, and redistribution information is provided in:

- `data/README.md`

## Reproducibility

The repository contains the implementation used for routing, retrieval, reranking, generation, internal and external evaluation, baseline experiments, ablation experiments, and metric calculation.

Exact commands for reproducing the main experiments and reported tables are documented alongside the corresponding evaluation scripts.


### Main evaluation commands

Run the four experimental configurations used in the paper:

```bash
# Basic RAG
python scripts/evaluation/run_article_baselines_800.py \
  --mode simple_rag \
  --max-new 800

# Routed RAG
python scripts/evaluation/run_article_routed_800.py \
  --mode routed_rag \
  --max-new 800

# Routed RAG + ColBERT
python scripts/evaluation/run_article_routed_800.py \
  --mode routed_rag_rerank \
  --max-new 800

# Full System
python scripts/evaluation/run_article_internal_800_full.py
```

Run the external evaluation:

```bash
python scripts/evaluation/run_external_evaluation_721.py
```

Calculate metrics from an evaluation result file:

```bash
python scripts/evaluation/calculate_article_metrics.py \
  --input <RESULT_JSONL> \
  --output <METRICS_JSON>
```

Optional smoke-test flags are available in the evaluation scripts to verify the environment before running the complete evaluation.

## Clinical Use Disclaimer

This work evaluates information retrieval and answer generation in a medical QA research setting.

The Safety Triage component is rule-based and has not been clinically validated. The framework does not establish clinical safety or clinical effectiveness.

Prospective evaluation using real-world clinical data and independent clinical validation would be required before any deployment in patient care.
