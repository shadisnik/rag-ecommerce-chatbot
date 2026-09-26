import json
import os
import re
from typing import Any, Dict, List, Optional

from app.services.medical_retrieval import retrieve_medical_context
from app.services.medical_generation import (
    OpenAI,
    generate_medical_answer,
    get_medical_api_key,
    get_medical_base_url,
    get_medical_model,
)
from app.services.trusted_medical_web import retrieve_trusted_medical_web


VERIFICATION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "what", "when", "which", "who", "with",
    "patients", "patient", "children", "child", "rate", "effect", "level",
}


def verification_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", text.lower())
        if len(token) > 2 and token not in VERIFICATION_STOPWORDS
    }


def verification_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?%?", text.lower()))


def refine_routing_from_retrieval(
    routing: Dict[str, Any],
    docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Keep the route selected from the question itself.

    Retrieved categories must not change a general medical question
    into a pediatric oncology question merely because an oncology
    document appeared among the retrieval results.
    """
    return routing



def semantic_knowledge_router(
    query: str,
    subject_id: Optional[str] = None,
    hadm_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classify every incoming question into one of the five
    workflow routes.
    """
    query_lower = query.lower().strip()

    normalized_subject_id = (
        str(subject_id).strip()
        if subject_id is not None
        else ""
    )
    normalized_hadm_id = (
        str(hadm_id).strip()
        if hadm_id is not None
        else ""
    )

    patient_match = re.search(
        r"\b(PONC-\d{3})\b",
        query,
        re.IGNORECASE,
    )
    patient_id = (
        patient_match.group(1).upper()
        if patient_match
        else None
    )

    day_match = re.search(
        r"\b(?:treatment\s+)?day\s*(\d{1,3})\b",
        query_lower,
    )
    day_number = day_match.group(1) if day_match else None

    # Safe fallback if the routing model is unavailable.
    if normalized_subject_id or normalized_hadm_id:
        fallback_route = "clinical_qa"
    elif patient_id and day_number:
        fallback_route = "patient_timeline"
    elif patient_id:
        fallback_route = "patient_timeline_summary"
    else:
        fallback_route = "general_medical_rag"

    category_map = {
        "clinical_qa": "ehr_qa",
        "patient_timeline": "pediatric_patient_timeline",
        "patient_timeline_summary": "pediatric_patient_timeline",
        "pediatric_oncology": "pediatric_oncology",
        "general_medical_rag": "general_medical_qa",
    }

    default_result = {
        "route": fallback_route,
        "expected_category": category_map[fallback_route],
        "document_keyword": None,
        "routing_method": "five_route_fallback",
        "routing_confidence": None,
        "routing_reason": (
            "The semantic routing model was unavailable; "
            "structured evidence was used."
        ),
        "missing_information": [],
    }

    api_key = get_medical_api_key()
    if OpenAI is None or not api_key:
        return default_result

    metadata = (
        f"subject_id: {normalized_subject_id or 'not provided'}\n"
        f"hadm_id: {normalized_hadm_id or 'not provided'}\n"
        f"detected timeline patient id: {patient_id or 'not detected'}\n"
        f"detected treatment day: {day_number or 'not detected'}"
    )

    try:
        client_kwargs = {"api_key": api_key}
        base_url = get_medical_base_url()

        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=(
                os.getenv("MEDICAL_ROUTER_MODEL")
                or get_medical_model()
            ),
            temperature=0,
            max_tokens=220,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the routing agent for a medical "
                        "multi-agent RAG system. Do not answer the "
                        "medical question. Classify every request into "
                        "exactly one of five routes.\n\n"

                        "clinical_qa: Questions about a particular "
                        "clinical record, admission, discharge summary, "
                        "diagnoses, procedures, medications, or results "
                        "from an EHR encounter.\n\n"

                        "patient_timeline: Questions about a specific "
                        "day or event in a pediatric oncology treatment "
                        "timeline.\n\n"

                        "patient_timeline_summary: Questions asking for "
                        "overall progress, changes, trends, or a summary "
                        "across a patient's treatment timeline.\n\n"

                        "pediatric_oncology: General oncology knowledge, "
                        "including cancer diagnosis, staging, prognosis, "
                        "tumors, hematologic malignancies, chemotherapy "
                        "protocols, relapse, and remission, when the "
                        "question is not about one specific patient "
                        "timeline or EHR encounter.\n\n"

                        "general_medical_rag: General diseases, symptoms, "
                        "routine medical information, and general drug "
                        "questions such as dose, usage, interaction, "
                        "side effects, storage, or appearance.\n\n"

                        "Identifiers and metadata are strong evidence, "
                        "but the question's intent must also be considered. "
                        "If the selected route requires a patient ID, "
                        "admission ID, or treatment day that was not "
                        "provided, list it in missing_information.\n\n"

                        "Return JSON only:\n"
                        '{"route":"one allowed route",'
                        '"confidence":0.0,'
                        '"reason":"brief reason",'
                        '"missing_information":[]}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{query}\n\n"
                        f"Available metadata:\n{metadata}"
                    ),
                },
            ],
        )

        raw_output = (
            response.choices[0].message.content or ""
        ).strip()

        json_match = re.search(
            r"\{.*\}",
            raw_output,
            re.DOTALL,
        )
        if not json_match:
            raise ValueError("No JSON object returned.")

        result = json.loads(json_match.group(0))
        route = str(result.get("route", "")).strip()

        allowed_routes = set(category_map)
        if route not in allowed_routes:
            raise ValueError(f"Unsupported route: {route}")

        try:
            confidence = float(result.get("confidence"))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = None

        # Exact structured identifiers remain authoritative evidence.
        if normalized_subject_id or normalized_hadm_id:
            route = "clinical_qa"
        elif patient_id and day_number:
            route = "patient_timeline"
        elif patient_id:
            route = "patient_timeline_summary"

        # Determine required metadata from the selected route.
        # Do not accept inconsistent missing fields suggested by the model.
        if route == "clinical_qa":
            missing = (
                []
                if normalized_subject_id or normalized_hadm_id
                else ["subject_id or hadm_id"]
            )

        elif route == "patient_timeline":
            missing = []
            if not patient_id:
                missing.append("patient_id")
            if not day_number:
                missing.append("treatment_day")

        elif route == "patient_timeline_summary":
            missing = (
                []
                if patient_id
                else ["patient_id"]
            )

        else:
            missing = []

        return {
            "route": route,
            "expected_category": category_map[route],
            "document_keyword": None,
            "routing_method": "five_route_semantic_llm",
            "routing_confidence": confidence,
            "routing_reason": str(
                result.get("reason", "")
            ).strip(),
            "missing_information": sorted(set(missing)),
        }

    except Exception as exc:
        failed = dict(default_result)
        failed["routing_method"] = "five_route_error_fallback"
        failed["routing_reason"] = (
            f"Semantic router error: {type(exc).__name__}"
        )
        return failed


def router_agent(
    query: str,
    subject_id: Optional[str] = None,
    hadm_id: Optional[str] = None,
) -> Dict[str, Any]:
    routing = semantic_knowledge_router(
        query=query,
        subject_id=subject_id,
        hadm_id=hadm_id,
    )

    query_lower = query.lower().strip()

    patient_match = re.search(
        r"\b(PONC-\d{3})\b",
        query,
        re.IGNORECASE,
    )
    patient_id = (
        patient_match.group(1).upper()
        if patient_match
        else None
    )

    day_match = re.search(
        r"\b(?:treatment\s+)?day\s*(\d{1,3})\b",
        query_lower,
    )
    day_number = day_match.group(1) if day_match else None

    route = routing.get("route")

    if route == "patient_timeline" and patient_id and day_number:
        routing["document_keyword"] = (
            f"synthetic_timeline:{patient_id}:day_{day_number}"
        )

    elif route == "patient_timeline_summary" and patient_id:
        routing["document_keyword"] = (
            f"synthetic_timeline_patient:{patient_id}"
        )

    elif (
        route == "pediatric_oncology"
        and (
            "acute lymphoblastic leukemia" in query_lower
            or "acute lymphoblastic leukaemia" in query_lower
            or "childhood all" in query_lower
        )
    ):
        routing["document_keyword"] = (
            "Childhood Acute Lymphoblastic Leukemia Treatment"
        )

    return routing

def patient_context_agent(
    routing: Dict[str, Any],
    subject_id: Optional[str] = None,
    hadm_id: Optional[str] = None,
) -> Dict[str, Any]:
    subject_id = str(subject_id).strip() if subject_id else None
    hadm_id = str(hadm_id).strip() if hadm_id else None

    if subject_id == "":
        subject_id = None

    if hadm_id == "":
        hadm_id = None

    has_patient_context = bool(subject_id or hadm_id)
    applies_to_patient_context = routing.get("route") == "clinical_qa"

    warning = ""

    if applies_to_patient_context and not has_patient_context:
        warning = (
            "No subject_id or hadm_id was provided. "
            "Clinical retrieval may include information from multiple patients."
        )

    return {
        "has_patient_context": has_patient_context,
        "applies_to_patient_context": applies_to_patient_context,
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "warning": warning,
    }


def retrieval_agent(
    query: str,
    routing: Dict[str, Any],
    patient_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    subject_id = None
    hadm_id = None

    if patient_context.get("applies_to_patient_context"):
        subject_id = patient_context.get("subject_id")
        hadm_id = patient_context.get("hadm_id")

    if routing.get("route") == "patient_timeline_summary":
        top_k = 90
        candidate_limit = 100
    else:
        top_k = 5
        candidate_limit = 50

    # General medical questions must not retrieve pediatric
    # oncology documents. Pediatric oncology drug information
    # remains available through the pediatric oncology route.
    if routing.get("route") == "general_medical_rag":
        return retrieve_medical_context(
            query=query,
            top_k=top_k,
            candidate_limit=candidate_limit,
            document_keyword=None,
            subject_id=subject_id,
            hadm_id=hadm_id,
            expected_category="general_medical_qa",
        )

    docs = retrieve_medical_context(
        query=query,
        top_k=top_k,
        candidate_limit=candidate_limit,
        document_keyword=routing.get("document_keyword"),
        subject_id=subject_id,
        hadm_id=hadm_id,
        expected_category=routing.get("expected_category"),
    )

    return docs


def evidence_sufficiency_agent(
    query: str,
    docs: List[Dict[str, Any]],
    route: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Determine whether the retrieved evidence is sufficient to answer
    the user's question. Timeline routes use deterministic checks;
    other routes use semantic LLM-based evidence assessment.
    """

    if not docs:
        return {
            "sufficient": False,
            "assessment_method": "deterministic",
            "reason": "No supporting documents were retrieved.",
            "missing_information": ["supporting evidence"],
            "refined_query": query,
        }

    # Exact timeline-day retrieval.
    if route == "patient_timeline":
        return {
            "sufficient": True,
            "assessment_method": "deterministic",
            "reason": "The exact requested patient timeline record was retrieved.",
            "missing_information": [],
            "refined_query": "",
        }

    # Timeline summaries are expected to contain all 90 daily records.
    if route == "patient_timeline_summary":
        record_count = len(docs)
        sufficient = record_count >= 90

        return {
            "sufficient": sufficient,
            "assessment_method": "deterministic",
            "record_count": record_count,
            "expected_record_count": 90,
            "reason": (
                f"{record_count} timeline records were retrieved "
                f"out of 90 expected records."
            ),
            "missing_information": (
                []
                if sufficient
                else ["missing patient timeline records"]
            ),
            "refined_query": "",
        }

    # Semantic assessment for general medical, pediatric oncology,
    # and EHR clinical QA routes.
    api_key = get_medical_api_key()

    if OpenAI is None or not api_key:
        return {
            "sufficient": False,
            "assessment_method": "unavailable_fallback",
            "reason": (
                "Semantic sufficiency assessment was unavailable. "
                "Answer generation was blocked because evidence "
                "sufficiency could not be confirmed."
            ),
            "missing_information": [
                "evidence sufficiency could not be confirmed"
            ],
            "refined_query": "",
        }

    evidence_blocks = []

    web_evidence = [
        doc
        for doc in docs
        if doc.get("evidence_origin") == "trusted_web"
    ]

    kb_evidence = [
        doc
        for doc in docs
        if doc.get("evidence_origin") != "trusted_web"
    ]

    # Inspect evidence from both the initial and refined KB retrieval.
    # Non-summary routes retrieve up to five documents per retrieval pass,
    # so up to ten KB documents may be available after refinement.
    selected_docs = kb_evidence[:10]

    # When trusted web fallback has been used, inspect that evidence too.
    if web_evidence:
        selected_docs += web_evidence[:5]

    for index, doc in enumerate(selected_docs, start=1):
        content = str(
            doc.get("content", "")
        ).strip()

        if content:
            source = (
                doc.get("document_source")
                or doc.get("source")
                or doc.get("evidence_origin")
                or "unknown"
            )

            evidence_blocks.append(
                f"[Evidence {index} | Source: {source}]\n"
                f"{content[:3000]}"
            )

    evidence_text = "\n\n".join(evidence_blocks)

    if not evidence_text:
        return {
            "sufficient": False,
            "assessment_method": "deterministic",
            "reason": "Retrieved documents contained no usable text.",
            "missing_information": ["supporting evidence"],
            "refined_query": query,
        }

    try:
        client_kwargs = {"api_key": api_key}
        base_url = get_medical_base_url()

        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=(
                os.getenv("MEDICAL_SUFFICIENCY_MODEL")
                or get_medical_model()
            ),
            temperature=0,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an Evidence Sufficiency Agent in a medical "
                        "question-answering system. Your task is NOT to answer "
                        "the user's medical question. Your task is only to determine "
                        "whether the retrieved evidence contains enough factual "
                        "information to answer all parts of the user's question. "
                        "Do not use outside medical knowledge.\n\n"

                        "Judge only whether the retrieved evidence contains the factual "
                        "information needed to answer the user's question. The evidence "
                        "does not need to contain a ready-made, concise, or well-structured "
                        "answer. Information may be distributed across multiple documents "
                        "and should be considered collectively. Do not mark evidence "
                        "insufficient merely because summarization, synthesis, or organization "
                        "is still required; those are responsibilities of the generation step. "
                        "Mark evidence insufficient only when factual information required "
                        "by one or more parts of the question is genuinely absent from the "
                        "retrieved evidence.\n\n"

                        "Synonyms, paraphrases, and semantically equivalent medical "
                        "expressions count as evidence. Do not require exact lexical overlap "
                        "with the user's wording.\n\n"

                        "If evidence is incomplete, identify only the information that is "
                        "genuinely missing and create a concise retrieval-oriented refined "
                        "query. Preserve the original medical entity, but make the refined "
                        "query focus only on the information that is still missing. Do not "
                        "repeat question aspects that are already supported by the retrieved "
                        "evidence.\n\n"

                        "When creating refined_query, include only the original medical "
                        "entity and the genuinely missing information. Do not add "
                        "already-supported question parts. Do not use Boolean operators "
                        "such as AND or OR. Do not introduce additional concepts unless "
                        "they are direct synonyms of the missing concept. Use keyword-style "
                        "search wording rather than simply repeating the user's question.\n\n"

                        "Return JSON only in this format:\n"
                        "{"
                        "\"sufficient\": true,"
                        "\"reason\": \"brief explanation\","
                        "\"missing_information\": [],"
                        "\"refined_query\": \"\""
                        "}\n"
                        "sufficient must be true or false. If sufficient is true, "
                        "missing_information must be an empty list and refined_query "
                        "must be an empty string."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Route: {route}\n\n"
                        f"Question:\n{query}\n\n"
                        f"Retrieved evidence:\n{evidence_text}"
                    ),
                },
            ],
        )

        raw_output = (
            response.choices[0].message.content or ""
        ).strip()

        json_match = re.search(
            r"\{.*\}",
            raw_output,
            re.DOTALL,
        )

        if not json_match:
            raise ValueError(
                "No JSON object returned by sufficiency agent."
            )

        result = json.loads(json_match.group(0))

        sufficient = bool(result.get("sufficient", False))

        missing_information = result.get(
            "missing_information",
            [],
        )

        if not isinstance(missing_information, list):
            missing_information = [
                str(missing_information)
            ]

        refined_query = str(
            result.get("refined_query", "")
        ).strip()

        if not sufficient and not refined_query:
            refined_query = query

        return {
            "sufficient": sufficient,
            "assessment_method": "semantic_llm",
            "reason": str(
                result.get("reason", "")
            ).strip(),
            "missing_information": missing_information,
            "refined_query": refined_query,
        }

    except Exception as exc:
        return {
            "sufficient": False,
            "assessment_method": "error_fallback",
            "reason": (
                "Semantic sufficiency assessment failed; "
                f"{type(exc).__name__}. Answer generation was blocked "
                "because evidence sufficiency could not be confirmed."
            ),
            "missing_information": [
                "evidence sufficiency could not be confirmed"
            ],
            "refined_query": "",
        }

def safety_triage_agent(
    docs: List[Dict[str, Any]],
    route: Optional[str] = None,
) -> Dict[str, Any]:
    timeline_routes = {
        "patient_timeline",
        "patient_timeline_summary",
    }

    if route not in timeline_routes:
        return {
            "safety_level": "not_applicable",
            "safety_reason": (
                "Safety triage was not applied because the query "
                "does not use patient timeline evidence."
            ),
            "recommended_action": "",
            "description": (
                "This agent is applied only to patient timeline queries."
            ),
        }

    combined_text = ""

    for doc in docs:
        combined_text += " "
        combined_text += doc.get("content", "").lower()

    safety_level = "low"
    safety_reason = "No high-risk warning signs were detected in the retrieved context."
    recommended_action = "Continue monitoring and record symptoms."

    high_risk_signs = [
        "risk level: high",
        "fever during chemotherapy",
        "body temperature: 38",
        "body temperature: 39",
        "very low anc",
        "anc: very low",
        "seek urgent medical advice",
        "contact the oncology care team immediately",
        "the child appears weaker than usual",
    ]

    medium_risk_signs = [
        "risk level: medium",
        "vomiting",
        "poor oral intake",
        "mouth sore",
        "reduced fluid intake",
        "symptoms require closer monitoring",
        "continue monitoring and contact the care team if symptoms worsen",
        "reduced activity",
    ]

    if any(sign in combined_text for sign in high_risk_signs):
        safety_level = "high"
        safety_reason = (
            "High-risk signs were found in the retrieved patient timeline, "
            "such as fever during chemotherapy and/or very low ANC."
        )
        recommended_action = (
            "Contact the oncology care team immediately. "
            "This prototype does not replace clinical judgment."
        )

    elif any(sign in combined_text for sign in medium_risk_signs):
        safety_level = "medium"
        safety_reason = (
            "Moderate-risk signs were found in the retrieved patient timeline. "
            "These signs may require closer monitoring."
        )
        recommended_action = (
            "Continue close monitoring and contact the care team if symptoms worsen "
            "or persist."
        )

    return {
        "safety_level": safety_level,
        "safety_reason": safety_reason,
        "recommended_action": recommended_action,
        "description": (
            "This agent performs a conservative safety check based on retrieved "
            "patient timeline evidence."
        ),
    }


def generation_agent(query: str, docs: List[Dict[str, Any]]) -> str:
    return generate_medical_answer(query, docs)


def add_safety_message_to_answer(
    answer: str,
    safety_triage: Dict[str, Any],
) -> str:
    safety_level = safety_triage.get("safety_level", "low")
    recommended_action = safety_triage.get("recommended_action", "")

    if safety_level == "high":
        return (
            f"{answer}\n\n"
            f"Safety note: High-risk situation detected. {recommended_action}"
        )

    if safety_level == "medium":
        return (
            f"{answer}\n\n"
            f"Safety note: Medium-risk situation detected. {recommended_action}"
        )

    return answer


def verification_agent(
    query: str,
    answer: str,
    docs: List[Dict[str, Any]],
    routing: Dict[str, Any],
    patient_context: Dict[str, Any],
) -> Dict[str, Any]:
    if not docs:
        return {
            "verified": False,
            "warning": "No retrieved medical sources were found.",
        }

    if not answer or not answer.strip():
        return {
            "verified": False,
            "warning": "The generated answer is empty.",
        }

    retrieved_categories = {
        doc.get("category", "")
        for doc in docs
        if doc.get("category")
    }
    expected_category = routing.get("expected_category")

    if expected_category and expected_category not in retrieved_categories:
        return {
            "verified": False,
            "warning": (
                f"Expected category '{expected_category}' was not found "
                "in retrieved sources."
            ),
        }

    if (
        patient_context.get("applies_to_patient_context")
        and patient_context.get("has_patient_context")
    ):
        subject_id = patient_context.get("subject_id")
        hadm_id = patient_context.get("hadm_id")

        for doc in docs:
            document_url = doc.get("document_url", "").lower()
            if subject_id and f"subject_id={subject_id.lower()}" not in document_url:
                return {
                    "verified": False,
                    "warning": "A retrieved source does not match the requested subject_id.",
                }
            if hadm_id and f"hadm_id={hadm_id.lower()}" not in document_url:
                return {
                    "verified": False,
                    "warning": "A retrieved source does not match the requested hadm_id.",
                }

    # Insufficient-context answers must never be marked as verified,
    # regardless of the route.
    answer_lower = answer.lower()

    insufficient_phrases = (
        "retrieved context is insufficient",
        "context is insufficient",
        "insufficient to confirm",
        "insufficient to determine",
        "insufficient to answer",
    )

    if any(
        phrase in answer_lower
        for phrase in insufficient_phrases
    ):
        return {
            "verified": False,
            "warning": (
                "The generation agent reported insufficient "
                "supporting context."
            ),
        }

    # Keep the established deterministic checks for timeline and EHR routes.
    # Apply stronger semantic grounding checks to NCI pediatric-oncology QA.
    if routing.get("route") != "pediatric_oncology":
        return {
            "verified": True,
            "warning": patient_context.get("warning", ""),
        }

    source_text = " ".join(
        doc.get("content", "")
        for doc in docs
        if doc.get("content")
    )

    query_terms = verification_tokens(query)
    answer_terms = verification_tokens(answer)
    source_terms = verification_tokens(source_text)

    query_answer_overlap = query_terms & answer_terms
    answer_source_overlap = answer_terms & source_terms

    query_coverage = (
        len(query_answer_overlap) / len(query_terms)
        if query_terms
        else 1.0
    )
    grounding_coverage = (
        len(answer_source_overlap) / len(answer_terms)
        if answer_terms
        else 0.0
    )

    answer_numbers = verification_numbers(answer)
    source_numbers = verification_numbers(source_text)
    unsupported_numbers = sorted(answer_numbers - source_numbers)

    if len(answer.split()) > 180:
        return {
            "verified": False,
            "warning": "The answer is too long and may contain unrelated retrieved text.",
            "query_coverage": round(query_coverage, 3),
            "grounding_coverage": round(grounding_coverage, 3),
        }

    if len(query_answer_overlap) < 2 and query_coverage < 0.30:
        return {
            "verified": False,
            "warning": "The answer does not directly address enough of the question.",
            "query_coverage": round(query_coverage, 3),
            "grounding_coverage": round(grounding_coverage, 3),
        }

    if grounding_coverage < 0.35:
        return {
            "verified": False,
            "warning": "Too much of the answer is not grounded in the retrieved sources.",
            "query_coverage": round(query_coverage, 3),
            "grounding_coverage": round(grounding_coverage, 3),
        }

    if unsupported_numbers:
        return {
            "verified": False,
            "warning": (
                "The answer contains numeric values that were not found in the "
                f"retrieved sources: {', '.join(unsupported_numbers)}"
            ),
            "query_coverage": round(query_coverage, 3),
            "grounding_coverage": round(grounding_coverage, 3),
        }

    return {
        "verified": True,
        "warning": "",
        "query_coverage": round(query_coverage, 3),
        "grounding_coverage": round(grounding_coverage, 3),
        "unsupported_numbers": [],
    }


def build_sources(
    docs: List[Dict[str, Any]],
    max_sources: int = 10,
) -> List[Dict[str, str]]:
    sources = []

    for doc in docs[:max_sources]:
        sources.append(
            {
                "document_source": doc.get("document_source", ""),
                "document_url": doc.get("document_url", ""),
                "category": doc.get("category", ""),
                "content": doc.get("content", "")[:700],
            }
        )

    return sources



def _web_fallback_allowed(
    routing: Dict[str, Any],
    subject_id: Optional[str],
    hadm_id: Optional[str],
    query: str,
) -> bool:
    enabled = os.getenv(
        "ENABLE_TRUSTED_WEB_FALLBACK",
        "false",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if not enabled:
        return False

    route = routing.get("route")

    if route not in {
        "general_medical_rag",
        "pediatric_oncology",
    }:
        return False

    # Patient-specific information must never go to the web.
    if subject_id or hadm_id:
        return False

    if re.search(
        r"\bPONC-\d{3}\b",
        query,
        re.IGNORECASE,
    ):
        return False

    return True


def _prepare_web_docs(
    web_docs: List[Dict[str, Any]],
    routing: Dict[str, Any],
) -> List[Dict[str, Any]]:
    expected_category = (
        routing.get("expected_category")
        or "general_medical_qa"
    )

    prepared = []

    for item in web_docs:
        doc = dict(item)
        doc["web_original_category"] = doc.get(
            "category",
            "",
        )
        doc["category"] = expected_category
        prepared.append(doc)

    return prepared

def run_medical_workflow(
    query: str,
    session_id: str = "default",
    subject_id: Optional[str] = None,
    hadm_id: Optional[str] = None,
) -> Dict[str, Any]:
    query = query.strip()

    routing = router_agent(
        query=query,
        subject_id=subject_id,
        hadm_id=hadm_id,
    )

    patient_context = patient_context_agent(
        routing=routing,
        subject_id=subject_id,
        hadm_id=hadm_id,
    )

    # Stop before retrieval when the selected route requires
    # patient-specific information that was not provided.
    missing_information = routing.get("missing_information") or []

    if missing_information:
        missing_text = ", ".join(
            str(item) for item in missing_information
        )

        return {
            "route": routing["route"],
            "found": False,
            "message": (
                "Additional information is required before this question "
                f"can be answered safely: {missing_text}."
            ),
            "products": [],
            "sources": [],
            "agent_trace": {
                "router_agent": routing,
                "patient_context_agent": patient_context,
                "retrieval_agent": {
                    "retrieved_documents": 0,
                    "knowledge_base_documents": len(kb_docs),
                    "refinement_attempted": retrieval_refinement_attempted,
                    "refinement_query": retrieval_refinement_query,
                    "additional_documents": additional_docs_count,
                    "web_fallback_allowed": fallback_allowed,
                    "web_fallback_attempted": web_fallback_attempted,
                    "web_fallback_used": web_fallback_used,
                    "web_fallback_reason": web_fallback_reason,
                    "web_documents": len(web_docs),
                    "retrieved_categories": [],
                },
                "evidence_sufficiency_agent": {
                    "sufficient": False,
                    "assessment_method": "missing_information_guard",
                    "reason": (
                        "Required patient-specific routing information "
                        "was not provided."
                    ),
                    "missing_information": missing_information,
                    "refined_query": "",
                },
                "safety_triage_agent": {
                    "safety_level": "not_applicable",
                    "safety_reason": (
                        "Retrieval was not performed because required "
                        "patient information was missing."
                    ),
                    "recommended_action": "",
                    "description": "",
                },
                "generation_agent": {
                    "answer_generated": False,
                },
                "verification_agent": {
                    "verified": False,
                    "warning": (
                        "Required patient-specific information was missing."
                    ),
                },
            },
        }

    docs = retrieval_agent(
        query=query,
        routing=routing,
        patient_context=patient_context,
    )

    routing = refine_routing_from_retrieval(
        routing=routing,
        docs=docs,
    )
    patient_context = patient_context_agent(
        routing=routing,
        subject_id=subject_id,
        hadm_id=hadm_id,
    )

    evidence_sufficiency = evidence_sufficiency_agent(
        query=query,
        docs=docs,
        route=routing.get("route"),
    )

    retrieval_refinement_attempted = False
    retrieval_refinement_query = ""
    additional_docs_count = 0

    if (
        not evidence_sufficiency.get("sufficient", True)
        and routing.get("route") in {
            "general_medical_rag",
            "pediatric_oncology",
            "clinical_qa",
        }
    ):
        refined_query = evidence_sufficiency.get(
            "refined_query",
            "",
        ).strip()

        if refined_query:
            retrieval_refinement_attempted = True
            retrieval_refinement_query = refined_query

            refined_docs = retrieval_agent(
                query=refined_query,
                routing=routing,
                patient_context=patient_context,
            )

            additional_docs_count = len(refined_docs)

            # Combine original and refined retrieval results
            # while avoiding duplicate documents.
            combined_docs = []
            seen = set()

            for doc in docs + refined_docs:
                key = (
                    doc.get("document_url", ""),
                    doc.get("content", ""),
                )

                if key not in seen:
                    seen.add(key)
                    combined_docs.append(doc)

            docs = combined_docs

            # Re-check sufficiency against the ORIGINAL user question.
            evidence_sufficiency = evidence_sufficiency_agent(
                query=query,
                docs=docs,
                route=routing.get("route"),
            )

    safety_triage = safety_triage_agent(
        docs=docs,
        route=routing.get("route"),
    )

    kb_docs = list(docs)
    web_docs: List[Dict[str, Any]] = []
    web_fallback_used = False
    web_fallback_attempted = False
    web_fallback_reason = ""



    fallback_allowed = _web_fallback_allowed(
        routing=routing,
        subject_id=subject_id,
        hadm_id=hadm_id,
        query=query,
    )

    # Retrieved KB evidence exists, but it is still insufficient
    # even after the optional refined retrieval.
    if (
        docs
        and not evidence_sufficiency.get("sufficient", True)
        and fallback_allowed
    ):
        web_fallback_attempted = True

        fallback_query = (
            evidence_sufficiency.get("refined_query")
            or retrieval_refinement_query
            or query
        )

        web_docs = retrieve_trusted_medical_web(
            query=fallback_query,
            top_k=5,
        )

        if web_docs:
            prepared_web_docs = _prepare_web_docs(
                web_docs=web_docs,
                routing=routing,
            )

            combined_docs = []
            seen = set()

            for doc in docs + prepared_web_docs:
                key = (
                    doc.get("document_url", ""),
                    doc.get("content", ""),
                )

                if key not in seen:
                    seen.add(key)
                    combined_docs.append(doc)

            docs = combined_docs

            web_fallback_used = True
            web_fallback_reason = (
                "evidence_sufficiency_insufficient"
            )

            # Re-check against the ORIGINAL user question.
            evidence_sufficiency = evidence_sufficiency_agent(
                query=query,
                docs=docs,
                route=routing.get("route"),
            )


    # No KB evidence at all -> try trusted medical web sources.
    if not docs and fallback_allowed:
        web_fallback_attempted = True

        web_docs = retrieve_trusted_medical_web(
            query=query,
            top_k=5,
        )

        if web_docs:
            docs = _prepare_web_docs(
                web_docs=web_docs,
                routing=routing,
            )
            web_fallback_used = True
            web_fallback_reason = "no_kb_documents"

            # Re-check the web evidence against the original question.
            evidence_sufficiency = evidence_sufficiency_agent(
                query=query,
                docs=docs,
                route=routing.get("route"),
            )

    if not docs:
        return {
            "route": routing["route"],
            "found": False,
            "message": "I could not find enough information in the medical knowledge base.",
            "products": [],
            "sources": [],
            "agent_trace": {
                "router_agent": routing,
                "patient_context_agent": patient_context,
                "retrieval_agent": {
                    "retrieved_documents": 0,
                    "retrieved_categories": [],
                },
                "evidence_sufficiency_agent": evidence_sufficiency,
                "safety_triage_agent": safety_triage,
                "generation_agent": {
                    "answer_generated": False,
                },
                "verification_agent": {
                    "verified": False,
                    "warning": "No documents retrieved.",
                },
            },
        }


    # Do not generate an answer when the available evidence
    # is still insufficient after all permitted retrieval attempts.
    if not evidence_sufficiency.get("sufficient", True):
        return {
            "route": routing["route"],
            "found": False,
            "message": (
                "I could not find enough supporting information "
                "to answer the question reliably."
            ),
            "products": [],
            "sources": build_sources(docs),
            "agent_trace": {
                "router_agent": routing,
                "patient_context_agent": patient_context,
                "retrieval_agent": {
                    "retrieved_documents": len(docs),
                    "knowledge_base_documents": len(kb_docs),
                    "refinement_attempted": retrieval_refinement_attempted,
                    "refinement_query": retrieval_refinement_query,
                    "additional_documents": additional_docs_count,
                    "web_fallback_allowed": fallback_allowed,
                    "web_fallback_attempted": web_fallback_attempted,
                    "web_fallback_used": web_fallback_used,
                    "web_fallback_reason": web_fallback_reason,
                    "web_documents": len(web_docs),
                    "retrieved_categories": list(
                        {
                            doc.get("category", "")
                            for doc in docs
                            if doc.get("category")
                        }
                    ),
                },
                "evidence_sufficiency_agent": evidence_sufficiency,
                "safety_triage_agent": safety_triage,
                "generation_agent": {
                    "answer_generated": False,
                },
                "verification_agent": {
                    "verified": False,
                    "warning": (
                        "Available evidence remained insufficient "
                        "after retrieval refinement."
                    ),
                },
            },
        }

    answer = generation_agent(query, docs)

    answer = add_safety_message_to_answer(
        answer=answer,
        safety_triage=safety_triage,
    )

    verification = verification_agent(
        query=query,
        answer=answer,
        docs=docs,
        routing=routing,
        patient_context=patient_context,
    )

    return {
        "route": routing["route"],
        "found": True,
        "message": answer,
        "products": [],
        "sources": build_sources(docs),
        "agent_trace": {
            "router_agent": routing,
            "patient_context_agent": patient_context,
            "retrieval_agent": {
                "retrieved_documents": len(docs),
                "knowledge_base_documents": len(kb_docs),
                "web_fallback_allowed": fallback_allowed,
                "web_fallback_attempted": web_fallback_attempted,
                "web_fallback_used": web_fallback_used,
                "web_fallback_reason": web_fallback_reason,
                "web_documents": len(web_docs),
                "refinement_attempted": retrieval_refinement_attempted,
                "refinement_query": retrieval_refinement_query,
                "additional_documents": additional_docs_count,
                "retrieved_categories": list(
                    {
                        doc.get("category", "")
                        for doc in docs
                        if doc.get("category")
                    }
                ),
            },
            "evidence_sufficiency_agent": evidence_sufficiency,
            "safety_triage_agent": safety_triage,
            "generation_agent": {
                "answer_generated": True,
            },
            "verification_agent": verification,
        },
    }
