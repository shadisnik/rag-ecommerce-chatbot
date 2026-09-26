from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable

from app.services.medical_generation import (
    OpenAI,
    get_medical_api_key,
    get_medical_base_url,
    get_medical_model,
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _evaluation_model() -> str:
    return (
        os.getenv("EVAL_REVIEW_MODEL")
        or os.getenv("MEDICAL_GENERATION_MODEL")
        or get_medical_model()
    )


def _prepare_references(
    references: Iterable[Any],
    max_total_chars: int = 14000,
) -> str:
    parts: list[str] = []
    used_chars = 0

    for index, value in enumerate(references, start=1):
        text = _clean_text(value)
        if not text:
            continue

        remaining = max_total_chars - used_chars
        if remaining <= 0:
            break

        text = text[:remaining]
        parts.append(f"Reference {index}:\n{text}")
        used_chars += len(text)

    return "\n\n".join(parts)


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("The evaluator did not return a JSON object.")

    return json.loads(cleaned[start : end + 1])


def _bounded_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(4.0, number))


def evaluate_semantic_answer(
    question: str,
    answer: str,
    references: Iterable[Any],
) -> dict[str, Any]:
    reference_text = _prepare_references(references)
    answer_text = _clean_text(answer)
    question_text = _clean_text(question)

    empty_result = {
        "semantic_judge_available": False,
        "semantic_answer_score": None,
        "semantic_answer_pass": None,
        "semantic_correctness": None,
        "semantic_completeness": None,
        "semantic_relevance": None,
        "semantic_contradiction": None,
        "semantic_unsupported_claim": None,
        "semantic_safety_issue": None,
        "semantic_judge_reason": "",
        "semantic_judge_error": "",
        "semantic_judge_model": _evaluation_model(),
    }

    if not answer_text:
        return {
            **empty_result,
            "semantic_answer_score": 0.0,
            "semantic_answer_pass": False,
            "semantic_judge_reason": "The system answer is empty.",
        }

    if not reference_text:
        return {
            **empty_result,
            "semantic_judge_error": "No reference answer was available.",
        }

    api_key = get_medical_api_key()

    if OpenAI is None or not api_key:
        return {
            **empty_result,
            "semantic_judge_error": (
                "The evaluation model client or API key is unavailable."
            ),
        }

    system_prompt = """
You are a strict medical answer evaluator.

Compare the candidate answer with the supplied reference answers.
Judge meaning and factual content rather than identical wording.

A concise candidate may receive full credit when it correctly answers
the question. Do not penalize paraphrasing.

Use ONLY the supplied reference answers as the evaluation authority.
Do not use outside medical knowledge, assumptions, or general background
information, even when that outside information may be medically correct.

A candidate statement is supported only when it is explicitly stated in,
or directly entailed by, at least one supplied reference.

Set unsupported_claim to true whenever the candidate introduces any
factual, causal, ingredient, dosage, diagnostic, treatment, risk, safety,
or recommendation claim that the references do not support.

Hedging words such as "may", "could", "possibly", or "potentially" do not
make an unsupported statement acceptable.

Penalize:
- medically incorrect statements;
- contradictions with the references;
- omission of information needed to answer the exact question;
- unsupported claims, numbers, drug recommendations, or risks;
- unsafe medical recommendations;
- saying that information is unavailable when the references clearly
  contain the answer.

Assign each score from 0 to 4:
- correctness: factual agreement with the references;
- completeness: coverage of information needed for the question;
- relevance: directness and relevance to the question.

Return JSON only with exactly these fields:
{
  "correctness": 0,
  "completeness": 0,
  "relevance": 0,
  "contradiction": false,
  "unsupported_claim": false,
  "safety_issue": false,
  "reason": "brief explanation"
}
""".strip()

    user_prompt = (
        f"Question:\n{question_text}\n\n"
        f"Candidate answer:\n{answer_text}\n\n"
        f"Reference answers:\n{reference_text}"
    )

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = get_medical_base_url()

    if base_url:
        client_kwargs["base_url"] = base_url

    client = None

    try:
        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=_evaluation_model(),
            temperature=0,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        raw_text = response.choices[0].message.content or ""
        result = _parse_json_object(raw_text)

        correctness = _bounded_score(result.get("correctness"))
        completeness = _bounded_score(result.get("completeness"))
        relevance = _bounded_score(result.get("relevance"))

        contradiction = bool(result.get("contradiction", False))
        unsupported_claim = bool(
            result.get("unsupported_claim", False)
        )
        safety_issue = bool(result.get("safety_issue", False))

        score = (
            (0.50 * correctness)
            + (0.30 * completeness)
            + (0.20 * relevance)
        ) / 4.0

        passed = (
            score >= 0.70
            and correctness >= 3.0
            and not contradiction
            and not unsupported_claim
            and not safety_issue
        )

        return {
            "semantic_judge_available": True,
            "semantic_answer_score": round(score, 6),
            "semantic_answer_pass": passed,
            "semantic_correctness": correctness,
            "semantic_completeness": completeness,
            "semantic_relevance": relevance,
            "semantic_contradiction": contradiction,
            "semantic_unsupported_claim": unsupported_claim,
            "semantic_safety_issue": safety_issue,
            "semantic_judge_reason": _clean_text(
                result.get("reason")
            ),
            "semantic_judge_error": "",
            "semantic_judge_model": _evaluation_model(),
        }

    except Exception as exc:
        return {
            **empty_result,
            "semantic_judge_error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }

    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
