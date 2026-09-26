import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv


load_dotenv(".env.cloud")


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def get_medical_api_key() -> str:
    return (
        os.getenv("TOGETHER_API_KEY")
        or os.getenv("TOGETHER_API_TOKEN")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )


def get_medical_base_url() -> Optional[str]:
    if (
        os.getenv("TOGETHER_API_KEY")
        or os.getenv("TOGETHER_API_TOKEN")
    ):
        return (
            os.getenv("TOGETHER_BASE_URL")
            or "https://api.together.ai/v1"
        )

    return os.getenv("OPENAI_BASE_URL") or None


def get_medical_model() -> str:
    configured = (
        os.getenv("MEDICAL_GENERATION_MODEL")
        or os.getenv("EVAL_REVIEW_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("TOGETHER_MODEL")
    )
    if configured:
        return configured

    base_url = (get_medical_base_url() or "").lower()
    if "together" in base_url or os.getenv("TOGETHER_API_KEY"):
        return "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    return "gpt-4o-mini"


def extract_text(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        return default

    return match.group(1).strip()


def get_first_timeline_doc(docs: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    for doc in docs:
        if doc.get("category") == "pediatric_patient_timeline":
            return doc

    return None


def has_category(docs: List[Dict[str, str]], category: str) -> bool:
    return any(doc.get("category") == category for doc in docs)


def combine_doc_content(docs: List[Dict[str, str]]) -> str:
    content_parts = []

    for doc in docs:
        content = doc.get("content", "")
        if content:
            content_parts.append(content)

    return " ".join(content_parts)


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]


def generate_pediatric_oncology_answer(
    query: str,
    docs: List[Dict[str, str]],
) -> str:
    """Generate a direct, source-grounded answer for NCI PDQ questions."""
    oncology_docs = [
        doc
        for doc in docs
        if doc.get("category") == "pediatric_oncology"
    ]

    return generate_llm_answer(
        query=query,
        docs=oncology_docs or docs,
        domain="pediatric oncology",
    )


def generate_timeline_risk_answer(
    query: str,
    docs: List[Dict[str, str]],
) -> str:
    timeline_doc = get_first_timeline_doc(docs)

    if not timeline_doc:
        return "I could not find a matching patient timeline record."

    content = timeline_doc.get("content", "")
    query_lower = query.lower()

    patient_id = extract_text(
        r"Patient timeline record for\s+([A-Z0-9\-]+)",
        content,
        default="the patient",
    )

    day = extract_text(
        r"Timeline day:\s*(\d+)",
        content,
        default="the requested day",
    )

    risk_level = extract_text(
        r"Risk level:\s*([^\.]+)",
        content,
        default="not specified",
    )

    treatment_phase = extract_text(
        r"Treatment phase:\s*([^\.]+)",
        content,
        default="not specified",
    )

    symptoms = extract_text(
        r"Reported symptoms:\s*([^\.]+)",
        content,
        default="not specified",
    )

    temperature = extract_text(
        r"Body temperature:\s*([0-9\.]+)\s*°C",
        content,
        default="not specified",
    )

    wbc = extract_text(
        r"wbc:\s*([^,\.]+)",
        content,
        default="not specified",
    )

    anc = extract_text(
        r"anc:\s*([^,\.]+)",
        content,
        default="not specified",
    )

    hemoglobin = extract_text(
        r"hemoglobin:\s*([^,\.]+)",
        content,
        default="not specified",
    )

    platelets = extract_text(
        r"platelets:\s*([^,\.]+)",
        content,
        default="not specified",
    )

    recommended_action = extract_text(
        r"Recommended action:\s*([^\.]+)",
        content,
        default="not specified",
    )

    laboratory_keywords = [
        "laboratory",
        "lab result",
        "lab finding",
        "blood test",
        "wbc",
        "white blood",
        "anc",
        "neutrophil",
        "hemoglobin",
        "haemoglobin",
        "platelet",
    ]

    symptom_keywords = [
        "symptom",
        "symptoms",
        "signs",
        "reported",
    ]

    temperature_keywords = [
        "temperature",
        "fever",
        "how high",
    ]

    action_keywords = [
        "recommended action",
        "recommendation",
        "recommended",
        "advised",
        "follow-up",
        "follow up",
        "what should",
        "what to do",
        "care action",
        "clinical action",
        "response",
        "step",
    ]

    risk_keywords = [
        "risk",
        "serious",
        "severity",
        "safety status",
        "how severe",
    ]

    if any(
        keyword in query_lower
        for keyword in laboratory_keywords
    ):
        return (
            f"The laboratory results for patient {patient_id} "
            f"on day {day} were: "
            f"WBC: {wbc}; "
            f"ANC: {anc}; "
            f"hemoglobin: {hemoglobin}; "
            f"platelets: {platelets}."
        )

    if any(
        keyword in query_lower
        for keyword in symptom_keywords
    ):
        return (
            f"The symptoms reported for patient {patient_id} "
            f"on day {day} were: {symptoms}."
        )

    if any(
        keyword in query_lower
        for keyword in temperature_keywords
    ):
        return (
            f"The recorded body temperature for patient "
            f"{patient_id} on day {day} was "
            f"{temperature} °C."
        )

    if any(
        keyword in query_lower
        for keyword in action_keywords
    ):
        return (
            f"The recommended action for patient {patient_id} "
            f"on day {day} was: {recommended_action}."
        )

    if any(
        keyword in query_lower
        for keyword in risk_keywords
    ):
        return (
            f"The recorded risk level for patient {patient_id} "
            f"on day {day} was {risk_level} risk."
        )

    return (
        f"On day {day}, patient {patient_id} was in the "
        f"{treatment_phase} phase. "
        f"The reported symptoms were {symptoms}. "
        f"The body temperature was {temperature} °C. "
        f"The risk level was {risk_level} risk. "
        f"The recommended action was: {recommended_action}."
    )


def generate_timeline_summary_answer(query: str, docs: List[Dict[str, str]]) -> str:
    timeline_docs = [
        doc for doc in docs
        if doc.get("category") == "pediatric_patient_timeline"
    ]

    if not timeline_docs:
        return "I could not find enough patient timeline records to summarize."

    first_content = timeline_docs[0].get("content", "")

    patient_id = extract_text(
        r"Patient timeline record for\s+([A-Z0-9\-]+)",
        first_content,
        default="the patient",
    )

    cancer_type = extract_text(
        r"child with\s+([^\.]+)",
        first_content,
    )

    treatment_phase = extract_text(
        r"Treatment phase:\s*([^\.]+)",
        first_content,
    )

    risk_counts = {}
    days = []
    high_risk_days = []
    medium_risk_days = []
    fever_days = []
    very_low_anc_days = []
    temperatures = []

    for doc in timeline_docs:
        content = doc.get("content", "")

        day_text = extract_text(
            r"Timeline day:\s*(\d+)",
            content,
        )

        if not day_text:
            continue

        day = int(day_text)
        days.append(day)

        risk_level = extract_text(
            r"Risk level:\s*([^\.]+)",
            content,
            default="not specified",
        ).lower()

        risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1

        if risk_level == "high":
            high_risk_days.append(day)

        if risk_level == "medium":
            medium_risk_days.append(day)

        temperature_text = extract_text(
            r"Body temperature:\s*([0-9\.]+)\s*°C",
            content,
        )

        if temperature_text:
            temperature_value = float(temperature_text)
            temperatures.append(temperature_value)

            if temperature_value >= 38.0:
                fever_days.append(day)

        if "anc: very low" in content.lower():
            very_low_anc_days.append(day)

    days = sorted(days)
    high_risk_days = sorted(high_risk_days)
    medium_risk_days = sorted(medium_risk_days)
    fever_days = sorted(fever_days)
    very_low_anc_days = sorted(very_low_anc_days)

    total_records = len(timeline_docs)
    start_day = days[0] if days else "unknown"
    end_day = days[-1] if days else "unknown"

    low_count = risk_counts.get("low", 0)
    medium_count = risk_counts.get("medium", 0)
    high_count = risk_counts.get("high", 0)

    answer = (
        f"Based on {total_records} retrieved daily timeline records, "
        f"patient {patient_id}"
    )

    if cancer_type:
        answer += f" is a child with {cancer_type}"

    answer += f". The retrieved monitoring period covers days {start_day} to {end_day}."

    if treatment_phase:
        answer += f" The treatment phase in the retrieved records is {treatment_phase}."

    answer += (
        f" Risk distribution across the retrieved timeline is: "
        f"{low_count} low-risk day(s), "
        f"{medium_count} medium-risk day(s), and "
        f"{high_count} high-risk day(s)."
    )

    if temperatures:
        answer += f" The maximum recorded temperature was {max(temperatures):.1f} °C."

    if high_risk_days:
        shown_days = high_risk_days[:10]
        answer += (
            f" High-risk episodes were found on day(s): "
            f"{', '.join(str(day) for day in shown_days)}"
        )

        if len(high_risk_days) > 10:
            answer += f", and {len(high_risk_days) - 10} more day(s)"

        answer += "."

    if fever_days:
        shown_days = fever_days[:10]
        answer += (
            f" Fever-level temperatures were recorded on day(s): "
            f"{', '.join(str(day) for day in shown_days)}"
        )

        if len(fever_days) > 10:
            answer += f", and {len(fever_days) - 10} more day(s)"

        answer += "."

    if very_low_anc_days:
        shown_days = very_low_anc_days[:10]
        answer += (
            f" Very low ANC was recorded on day(s): "
            f"{', '.join(str(day) for day in shown_days)}"
        )

        if len(very_low_anc_days) > 10:
            answer += f", and {len(very_low_anc_days) - 10} more day(s)"

        answer += "."

    if high_count > 0:
        answer += (
            " Overall, the patient timeline shows repeated high-risk episodes, "
            "so this case requires careful monitoring and clinical review."
        )

    elif medium_count > 0:
        answer += (
            " Overall, the patient timeline shows moderate symptoms without "
            "high-risk episodes in the retrieved records."
        )

    else:
        answer += (
            " Overall, the retrieved timeline shows mostly stable low-risk records."
        )

    return answer


def fallback_context_answer(query: str, docs: List[Dict[str, str]]) -> str:
    if not docs:
        return "I could not find enough information in the medical knowledge base."

    first_doc = docs[0]
    content = first_doc.get("content", "")

    if not content:
        return "I found a source, but it did not contain enough readable content."

    useful_sentences = split_sentences(content)

    selected = useful_sentences[:4]

    if not selected:
        return content[:700]

    return " ".join(selected)


def generate_llm_answer(
    query: str,
    docs: List[Dict[str, str]],
    domain: str = "general medical",
) -> str:
    if not docs:
        return "I could not find enough information in the medical knowledge base."

    context_blocks = []

    for index, doc in enumerate(docs, start=1):
        content = doc.get("content", "")
        source = doc.get("document_source", "")
        url = doc.get("document_url", "")
        category = doc.get("category", "")

        metadata = [
            f"Source {index}",
            f"Document source: {source}",
            f"Document URL: {url}",
            f"Category: {category}",
        ]

        if doc.get("evidence_origin"):
            metadata.append(
                f"Evidence origin: {doc.get('evidence_origin')}"
            )

        if doc.get("product_title"):
            metadata.append(
                f"Product title: {doc.get('product_title')}"
            )

        if doc.get("match_method"):
            metadata.append(
                f"Product match method: {doc.get('match_method')}"
            )

        if doc.get("requested_ndc"):
            metadata.append(
                f"Requested NDC: {doc.get('requested_ndc')}"
            )

        if doc.get("exact_product_confirmed") is not None:
            metadata.append(
                "Exact product confirmed: "
                f"{doc.get('exact_product_confirmed')}"
            )

        context_blocks.append(
            "\n".join(metadata)
            + f"\nContent:\n{content}"
        )

    context = "\n\n".join(context_blocks)
    api_key = get_medical_api_key()

    if OpenAI is None or not api_key:
        raise RuntimeError(
            "Medical generation API is unavailable."
        )

    try:
        client_kwargs = {"api_key": api_key}
        base_url = get_medical_base_url()
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=get_medical_model(),
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise medical RAG answer-generation agent. "
                        "Answer the exact user question using only the retrieved context. "
                        "Choose the source passage that directly answers the question and "
                        "ignore unrelated neighboring text. Do not copy a long passage. "
                        "Return a self-contained answer in one or two concise sentences. "
                        "Preserve exact numbers, percentages, stages, drug names, endpoints, "
                        "comparisons, and units from the context. Do not add unsupported "
                        "explanations or clinical recommendations. "
                        "For product-specific questions, respect the product-match metadata. "
                        "If an NDC was requested and Exact product confirmed is False, do not "
                        "claim that the retrieved label belongs to that exact NDC; clearly "
                        "state that only a drug-name or dosage-form match was found. "
                        "If the context does not clearly support an answer, state that the "
                        "retrieved context is insufficient."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Domain: {domain}\n"
                        f"Question:\n{query}\n\n"
                        f"Retrieved context:\n{context}"
                    ),
                },
            ],
        )

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError(
                "medical_llm_generation_failed: empty model response"
            )

        return answer.strip()

    except Exception as exc:
        raise RuntimeError(
            "medical_llm_generation_failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def generate_medical_answer(
    query: str,
    docs: List[Dict[str, str]],
) -> str:
    has_timeline_source = has_category(
        docs=docs,
        category="pediatric_patient_timeline",
    )

    has_pediatric_oncology_source = has_category(
        docs=docs,
        category="pediatric_oncology",
    )

    if has_timeline_source:
        timeline_docs = [
            doc
            for doc in docs
            if doc.get("category")
            == "pediatric_patient_timeline"
        ]

        if len(timeline_docs) > 1:
            return generate_timeline_summary_answer(
                query,
                timeline_docs,
            )

        return generate_timeline_risk_answer(
            query,
            timeline_docs,
        )

    if has_pediatric_oncology_source:
        return generate_pediatric_oncology_answer(
            query,
            docs,
        )

    return generate_llm_answer(
        query,
        docs,
    )