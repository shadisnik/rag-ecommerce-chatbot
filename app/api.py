from fastapi import FastAPI
from pydantic import BaseModel
import json
import os

from dotenv import load_dotenv

ENV = os.getenv("ENV", "local").lower()

if ENV == "cloud":
    load_dotenv(".env.cloud")
else:
    load_dotenv(".env.local")

print("STEP 1: env loaded")

import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.query import Filter
from sentence_transformers import SentenceTransformer

from scripts.query_parser import parse_query
from app.llm import generate_answer

WEAVIATE_MODE = os.getenv("WEAVIATE_MODE", "local").lower()
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_LOCAL_HTTP_PORT = int(os.getenv("WEAVIATE_LOCAL_HTTP_PORT", "8079"))
WEAVIATE_LOCAL_GRPC_PORT = int(os.getenv("WEAVIATE_LOCAL_GRPC_PORT", "50050"))
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"

print("STEP 2: config loaded")

app = FastAPI()
print("STEP 3: FastAPI created")

print("STEP 4: connecting to weaviate...")

if WEAVIATE_MODE == "cloud":
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
    )
else:
    client = weaviate.connect_to_local(
        port=WEAVIATE_LOCAL_HTTP_PORT,
        grpc_port=WEAVIATE_LOCAL_GRPC_PORT,
    )

print("STEP 5: weaviate connected")

print("STEP 6: loading collection...")
collection = client.collections.get("StoreDocs")
print("STEP 7: collection loaded")

# مدل‌ها را همان اول لود نکن
embedding_model = None
colbert_model = None

class ChatRequest(BaseModel):
    query: str


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("LOADING EMBEDDING MODEL...")
        embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        print("EMBEDDING MODEL LOADED")
    return embedding_model


def get_embedding(text: str):
    model = get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def get_colbert():
    global colbert_model
    if colbert_model is None:
        print("LOADING COLBERT...")
        from ragatouille import RAGPretrainedModel
        colbert_model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        print("COLBERT LOADED")
    return colbert_model


def clean_filters(filters: dict) -> dict:
    return {k: v for k, v in filters.items() if v is not None and v != ""}


def build_filters(filters: dict):
    filters = clean_filters(filters)

    where_filter = None
    for key, value in filters.items():
        current_filter = Filter.by_property(key).equal(value)
        where_filter = current_filter if where_filter is None else where_filter & current_filter

    return where_filter


def run_search(search_query, filters_dict, limit=8):
    where_filter = build_filters(filters_dict)

    query_vector = get_embedding(search_query)

    response = collection.query.hybrid(
        query=search_query,
        vector=query_vector,
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
    return {"message": f"E-commerce RAG API is running ({WEAVIATE_MODE})"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    print("CHAT STEP 1: request received")
    query = req.query.strip()

    print("CHAT STEP 2: parsing query")
    parsed = parse_query(query)

    print("CHAT STEP 3: staged retrieval start")
    retrieval_result = run_staged_retrieval(parsed, limit=8)
    print("CHAT STEP 4: staged retrieval done")

    stage = retrieval_result["stage"]
    candidates = retrieval_result["candidates"]

    if not candidates:
        print("CHAT STEP 4A: no candidates")
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

    print(f"CHAT STEP 5: valid candidates = {len(valid_candidates)}")

    if not valid_candidates:
        print("CHAT STEP 5A: no usable content")
        return {
            "query": query,
            "found": False,
            "message": "No usable product content found.",
            "products": [],
            "retrieved_products": [],
            "retrieval_stage": stage,
            "parsed_query": parsed
        }

    top_docs = valid_candidates[:3]

    if USE_RERANKER:
        print("CHAT STEP 6: rerank start")
        colbert = get_colbert()

        documents = [obj.properties["content"] for obj in valid_candidates]
        results = colbert.rerank(
            query=query,
            documents=documents,
            k=min(3, len(documents))
        )

        top_docs = [valid_candidates[r["result_index"]] for r in results]
        print("CHAT STEP 7: rerank done")
    else:
        print("CHAT STEP 6: reranker disabled")

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

    print("CHAT STEP 8: build context")
    context = build_context(top_docs, stage)

    print("CHAT STEP 9: generate answer start")
    answer = generate_answer(query, context)
    print("CHAT STEP 10: generate answer done")

    try:
        parsed_answer = json.loads(answer)
    except Exception:
        parsed_answer = {
            "found": False,
            "message": answer,
            "products": []
        }

    print("CHAT STEP 11: response ready")

    return {
        "query": query,
        **parsed_answer,
        "retrieval_stage": stage,
        "parsed_query": parsed,
        "retrieved_products": retrieved_products
    }