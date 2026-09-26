import os
from typing import Optional

from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.query import Filter
from sentence_transformers import SentenceTransformer
from app.services.colbert_reranker import rerank_candidates


load_dotenv(".env.cloud")

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

TIMELINE_DAYS = 90

_model = None


def get_embedding_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model


def is_reference_like(text: str) -> bool:
    text_lower = text.lower()

    reference_terms = [
        "j clin oncol",
        "j pediatr hematol oncol",
        "pediatr blood cancer",
        "arch dis child",
        "br j haematol",
        "lancet haematol",
        "leuk lymphoma",
        "blood adv",
        "bone marrow transplant",
        "ann oncol",
        "j cancer res ther",
        "int j radiat oncol biol phys",
        "medical research council",
        "working party",
        "working group",
        "berlin-frankfurt",
        "münster",
        "randomised trial",
        "randomized trial",
        "meta-analysis",
        "international agency for research on cancer",
        "et al",
    ]

    hits = sum(term in text_lower for term in reference_terms)

    fragment_starts = (
        "outcome of",
        "treatment of relapsed",
        "criteria.",
        "chemotherapy in",
        "acute lymphoblastic leukaemia",
        "with relapsed",
        "transplantation",
        "treatment of a pediatric patient",
        "available.",
        "and early treatment",
        "with relapsed/refractory",
    )

    starts_badly = text_lower.strip().startswith(fragment_starts)

    if hits >= 2:
        return True

    if starts_badly:
        return True

    return False


def matches_patient_context(
    document_url: str,
    subject_id: Optional[str] = None,
    hadm_id: Optional[str] = None,
) -> bool:
    if not subject_id and not hadm_id:
        return True

    document_url_lower = document_url.lower()

    if subject_id:
        expected_subject = f"subject_id={str(subject_id).lower()}"
        if expected_subject not in document_url_lower:
            return False

    if hadm_id:
        expected_hadm = f"hadm_id={str(hadm_id).lower()}"
        if expected_hadm not in document_url_lower:
            return False

    return True


def relevance_score(query: str, content: str) -> int:
    query_lower = query.lower()
    content_lower = content.lower()

    score = 0

    if query_lower in content_lower:
        score += 10

    stop_words = {
        "what",
        "when",
        "where",
        "which",
        "does",
        "did",
        "the",
        "was",
        "were",
        "with",
        "during",
        "patient",
        "given",
        "their",
        "herself",
        "himself",
        "this",
        "that",
        "from",
        "into",
        "about",
        "level",
    }

    important_terms = [
        word.strip("?.!,;:()[]{}")
        for word in query_lower.split()
        if len(word.strip("?.!,;:()[]{}")) > 3
        and word.strip("?.!,;:()[]{}").lower() not in stop_words
    ]

    for term in important_terms:
        if term in content_lower:
            score += 2

    medical_focus_terms = [
        "medication",
        "medications",
        "diagnosis",
        "diagnosed",
        "treatment",
        "therapy",
        "chemotherapy",
        "discharge",
        "follow-up",
        "follow up",
        "complication",
        "risk",
        "fever",
        "timeline",
        "symptoms",
    ]

    for term in medical_focus_terms:
        if term in query_lower and term in content_lower:
            score += 5

    return score


def fetch_exact_timeline_record(
    collection,
    document_keyword: str,
    top_k: int,
):
    response = collection.query.fetch_objects(
        filters=Filter.by_property("document_url").equal(document_keyword),
        limit=top_k,
        return_properties=[
            "content",
            "category",
            "document_source",
            "document_url",
        ],
    )

    results = []

    for obj in response.objects:
        props = obj.properties

        results.append(
            {
                "content": props.get("content", ""),
                "category": props.get("category", ""),
                "document_source": props.get("document_source", ""),
                "document_url": props.get("document_url", ""),
            }
        )

    return results


def fetch_patient_timeline_records(
    collection,
    patient_id: str,
):
    results = []

    for day in range(1, TIMELINE_DAYS + 1):
        document_url = f"synthetic_timeline:{patient_id}:day_{day}"

        daily_records = fetch_exact_timeline_record(
            collection=collection,
            document_keyword=document_url,
            top_k=1,
        )

        if daily_records:
            results.append(daily_records[0])

    return results


def retrieve_medical_context(
    query: str,
    top_k: int = 5,
    candidate_limit: int = 50,
    document_keyword: Optional[str] = None,
    subject_id: Optional[str] = None,
    hadm_id: Optional[str] = None,
    expected_category: Optional[str] = None,
    expected_categories: Optional[list[str]] = None,
):
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
    )

    try:
        collection = client.collections.get("MedicalDocs")

        if document_keyword and document_keyword.startswith("synthetic_timeline_patient:"):
            patient_id = document_keyword.replace("synthetic_timeline_patient:", "").strip().upper()

            return fetch_patient_timeline_records(
                collection=collection,
                patient_id=patient_id,
            )

        if document_keyword and document_keyword.startswith("synthetic_timeline:"):
            return fetch_exact_timeline_record(
                collection=collection,
                document_keyword=document_keyword,
                top_k=top_k,
            )

        model = get_embedding_model()

        query_for_embedding = "Represent this sentence for searching relevant passages: " + query
        query_vector = model.encode(query_for_embedding, normalize_embeddings=True).tolist()

        query_parameters = {
            "near_vector": query_vector,
            "limit": candidate_limit,
            "return_properties": [
                "content",
                "category",
                "document_source",
                "document_url",
            ],
        }

        categories = []

        if expected_category:
            categories.append(expected_category)

        if expected_categories:
            categories.extend(expected_categories)

        categories = list(
            dict.fromkeys(
                category
                for category in categories
                if category
            )
        )

        if len(categories) == 1:
            query_parameters["filters"] = (
                Filter.by_property("category").equal(
                    categories[0]
                )
            )

        elif len(categories) > 1:
            query_parameters["filters"] = Filter.any_of(
                [
                    Filter.by_property("category").equal(
                        category
                    )
                    for category in categories
                ]
            )

        response = collection.query.near_vector(
            **query_parameters
        )

        candidates = []

        for obj in response.objects:
            props = obj.properties

            content = props.get("content", "")
            category = props.get("category", "")
            document_source = props.get("document_source", "")
            document_url = props.get("document_url", "")

            if document_keyword:
                if document_keyword.lower() not in document_url.lower():
                    continue

            if not matches_patient_context(
                document_url=document_url,
                subject_id=subject_id,
                hadm_id=hadm_id,
            ):
                continue

            if is_reference_like(content):
                continue

            score = relevance_score(query=query, content=content)

            # Give a strong but targeted boost when a pediatric-oncology
            # query explicitly names a drug or therapeutic agent that matches
            # the Drug Dictionary record title.
            if (
                expected_category == "pediatric_oncology"
                and document_source == "NCI Drug Dictionary"
            ):
                first_line = content.splitlines()[0].lower() if content else ""

                query_terms = [
                    word.strip("?.!,;:()[]{}").lower()
                    for word in query.split()
                    if len(word.strip("?.!,;:()[]{}")) > 3
                ]

                if any(term in first_line for term in query_terms):
                    score += 20

            candidates.append(
                {
                    "content": content,
                    "category": category,
                    "document_source": document_source,
                    "document_url": document_url,
                    "relevance_score": score,
                }
            )

        # Remove duplicate documents before ColBERT reranking.
        # Documents with the same normalized content are treated as duplicates.
        unique_candidates = {}

        for item in candidates:
            normalized_content = " ".join(
                str(item.get("content", "")).lower().split()
            )

            if not normalized_content:
                continue

            existing = unique_candidates.get(normalized_content)

            if (
                existing is None
                or item.get("relevance_score", 0)
                > existing.get("relevance_score", 0)
            ):
                unique_candidates[normalized_content] = item

        candidates = list(unique_candidates.values())

        # Keep all unique dense-retrieval candidates available for
        # ColBERT reranking. Do not remove semantically relevant documents
        # using a lexical score before the semantic reranker can assess them.
        if not candidates:
            return []

        use_reranker = os.getenv(
            "USE_RERANKER",
            "true",
        ).strip().lower() in {"1", "true", "yes", "on"}

        if not use_reranker:
            return candidates[:top_k]

        return rerank_candidates(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

    finally:
        client.close()


if __name__ == "__main__":
    query = "Summarize the treatment progress of patient PONC-001."

    results = retrieve_medical_context(
        query=query,
        top_k=90,
        candidate_limit=100,
        document_keyword="synthetic_timeline_patient:PONC-001",
    )

    print("Query:", query)
    print("Retrieved records:", len(results))
    print("=" * 80)

    for result in results[:5]:
        print(result["document_url"])
        print(result["content"][:300])
        print("-" * 80)
