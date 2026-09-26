# Operational Rules

This file documents the deterministic rule-based components used in the reported experiments. These components do not use separate LLM prompts.

## Safety Triage

Safety Triage is applied only to:

- `patient_timeline`
- `patient_timeline_summary`

For all other routes, the safety level is reported as `not_applicable`.

The component scans the retrieved timeline evidence for predefined text indicators.

### High-risk indicators

The following indicators trigger the `high` level:

- `risk level: high`
- `fever during chemotherapy`
- `body temperature: 38`
- `body temperature: 39`
- `very low anc`
- `anc: very low`
- `seek urgent medical advice`
- `contact the oncology care team immediately`
- `the child appears weaker than usual`

When at least one high-risk indicator is detected, the associated recommendation is:

`Contact the oncology care team immediately. This prototype does not replace clinical judgment.`

### Medium-risk indicators

The following indicators trigger the `medium` level:

- `risk level: medium`
- `vomiting`
- `poor oral intake`
- `mouth sore`
- `reduced fluid intake`
- `symptoms require closer monitoring`
- `continue monitoring and contact the care team if symptoms worsen`
- `reduced activity`

The associated recommendation is:

`Continue close monitoring and contact the care team if symptoms worsen or persist.`

High-risk indicators take precedence over medium-risk indicators.

If neither high- nor medium-risk indicators are detected, the level remains `low`.

For medium- and high-level cases, a safety note is appended to the generated answer.

The Safety Triage component is rule-based and has not been clinically validated. It should not be interpreted as a clinical risk-assessment or decision-support tool.

## Verification

Verification is deterministic and begins with checks that apply across routes.

A response fails verification when:

1. No retrieved documents are available.
2. The generated answer is empty.
3. The expected source category is absent from the retrieved documents.
4. For patient-context queries, the retrieved source identifiers do not match the requested `subject_id` or `hadm_id`.
5. The answer explicitly reports insufficient evidence using one of the following phrases:
   - `retrieved context is insufficient`
   - `context is insufficient`
   - `insufficient to confirm`
   - `insufficient to determine`
   - `insufficient to answer`

For routes other than `pediatric_oncology`, a response that passes these checks is marked as verified.

## Additional Pediatric Oncology Verification

The `pediatric_oncology` route uses additional deterministic grounding checks.

The answer fails verification if any of the following conditions are met:

- The answer contains more than 180 words.
- Fewer than two content terms overlap between the question and answer **and simultaneously** `query_coverage < 0.30`.
- `grounding_coverage < 0.35`.
- The answer contains any numeric value that does not occur in the retrieved source evidence.

`query_coverage` is calculated from the overlap between normalized question and answer terms.

`grounding_coverage` is calculated from the overlap between normalized answer terms and retrieved-source terms.

Numeric values are extracted separately from the generated answer and retrieved evidence; any answer-side numeric value absent from the evidence causes verification to fail.

These verification rules are heuristic safeguards used for experimental evaluation. They do not establish clinical correctness, safety, or effectiveness.
