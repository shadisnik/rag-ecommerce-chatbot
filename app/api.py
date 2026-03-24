from fastapi import FastAPI
from pydantic import BaseModel
import json
import weaviate
from weaviate.classes.query import Filter
from ragatouille import RAGPretrainedModel

from scripts.query_parser import parse_query
from app.llm import generate_answer

app = FastAPI()

client = weaviate.connect_to_local(port=8079, grpc_port=50050)
collection = client.collections.get("StoreDocs")
colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


class ChatRequest(BaseModel):
    query: str


def clean_filters(filters: dict) -> dict:
    return {k: v for k, v in filters.items() if v is not None and v != ""}


def build_filters(filters: dict):
    filters = clean_filters(filters)

    where_filter = None
    for key, value in filters.items():
        current_filter = Filter.by_property(key).equal(value)
        where_filter = current_filter if where_filter is None else where_filter & current_filter

    return where_filter


def run_search(search_query: str, filters: dict, limit: int = 20):
    where_filter = build_filters(filters) if filters else None

    response = collection.query.hybrid(
        query=search_query,
        filters=where_filter,
        alpha=0.5,
        limit=limit
    )

    return response.objects


def run_staged_retrieval(parsed: dict, limit: int = 20):
    search_query = parsed["search_query"]

    strict_filters = clean_filters(parsed.get("strict_filters", {}))
    soft_filters = clean_filters(parsed.get("soft_filters", {}))

    stages = [
        ("strict", strict_filters),
        ("soft", soft_filters),
        ("none", {})
    ]

    for stage_name, filters_dict in stages:
        candidates = run_search(search_query, filters_dict, limit=limit)
        if candidates:
            return {
                "stage": stage_name,
                "filters_used": filters_dict,
                "candidates": candidates
            }

    return {
        "stage": "none",
        "filters_used": {},
        "candidates": []
    }


def build_context(top_docs: list, stage: str) -> str:
    context_parts = []

    if stage == "strict":
        retrieval_note = "Exact matches found based on the user's filters."
    elif stage == "soft":
        retrieval_note = "No exact matches found. These are the closest matches based on partial filters."
    else:
        retrieval_note = "No exact filtered matches found. These are the closest semantic matches."

    context_parts.append(f"Retrieval note: {retrieval_note}")

    for doc in top_docs:
        props = doc.properties

        context_parts.append(
            f"Product Name: {props.get('product_name', '')}\n"
            f"Category: {props.get('category', '')}\n"
            f"Sub Category: {props.get('sub_category', '')}\n"
            f"Gender: {props.get('gender', '')}\n"
            f"Article Type: {props.get('article_type', '')}\n"
            f"Color: {props.get('color', '')}\n"
            f"Usage: {props.get('usage', '')}\n"
            f"Season: {props.get('season', '')}\n"
            f"Year: {props.get('year', '')}\n"
            f"Image Path: {props.get('image_path', '')}\n"
            f"Description: {props.get('content', '')}\n"
        )

    return "\n\n".join(context_parts)


@app.get("/")
def root():
    return {"message": "E-commerce RAG API is running"}


@app.post("/chat")
def chat(req: ChatRequest):
    query = req.query.strip()
    parsed = parse_query(query)

    retrieval_result = run_staged_retrieval(parsed, limit=20)

    stage = retrieval_result["stage"]
    candidates = retrieval_result["candidates"]

    if not candidates:
        return {
            "query": query,
            "found": False,
            "message": "No relevant products found.",
            "products": [],
            "retrieved_products": [],
            "retrieval_stage": stage,
            "parsed_query": parsed
        }

    valid_candidates = [
        obj for obj in candidates
        if obj.properties.get("content")
    ]

    if not valid_candidates:
        return {
            "query": query,
            "found": False,
            "message": "No usable product content found.",
            "products": [],
            "retrieved_products": [],
            "retrieval_stage": stage,
            "parsed_query": parsed
        }

    documents = [obj.properties["content"] for obj in valid_candidates]

    results = colbert.rerank(
        query=query,
        documents=documents,
        k=min(5, len(documents))
    )

    top_docs = [valid_candidates[r["result_index"]] for r in results]

    retrieved_products = []
    for doc in top_docs:
        props = doc.properties
        retrieved_products.append({
            "product_name": props.get("product_name", ""),
            "category": props.get("category", ""),
            "gender": props.get("gender", ""),
            "color": props.get("color", ""),
            "usage": props.get("usage", ""),
            "article_type": props.get("article_type", ""),
            "image_path": props.get("image_path", ""),
            "content": props.get("content", ""),
            "link": f"https://www.google.com/search?q={props.get('product_name', '').replace(' ', '+')}"
        })

    context = build_context(top_docs, stage)
    answer = generate_answer(query, context)

    try:
        parsed_answer = json.loads(answer)
    except Exception:
        parsed_answer = {
            "found": False,
            "message": answer,
            "products": []
        }

    return {
        "query": query,
        **parsed_answer,
        "retrieval_stage": stage,
        "parsed_query": parsed,
        "retrieved_products": retrieved_products
    }