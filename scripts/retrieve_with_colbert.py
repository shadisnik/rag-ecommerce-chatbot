# scripts/retrieve_with_colbert.py

import weaviate
from weaviate.classes.query import Filter
from ragatouille import RAGPretrainedModel

from query_parser import parse_query
from app.llm import generate_answer


def clean_filters(filters: dict) -> dict:
    return {k: v for k, v in filters.items() if v is not None and v != ""}


def build_filters(filters: dict):
    filters = clean_filters(filters)

    where_filter = None
    for key, value in filters.items():
        current_filter = Filter.by_property(key).equal(value)

        if where_filter is None:
            where_filter = current_filter
        else:
            where_filter = where_filter & current_filter

    return where_filter


def run_staged_retrieval(collection, parsed, limit=20):
    search_query = parsed["search_query"]

    strict_filters = clean_filters(parsed.get("strict_filters", {}))
    soft_filters = clean_filters(parsed.get("soft_filters", {}))

    stages = [
        ("strict", strict_filters),
        ("soft", soft_filters),
        ("none", {})
    ]

    for stage_name, filters_dict in stages:
        where_filter = build_filters(filters_dict) if filters_dict else None

        response = collection.query.hybrid(
            query=search_query,
            filters=where_filter,
            alpha=0.5,
            limit=limit
        )

        candidates = response.objects

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


def print_candidates(title: str, candidates: list):
    print(f"\n{title}")
    for i, obj in enumerate(candidates, start=1):
        print(f"\nRank {i}")
        print("-" * 50)
        print(obj.properties)


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

        one_product_context = f"""
Product Name: {props.get("product_name", "")}
Category: {props.get("category", "")}
Sub Category: {props.get("sub_category", "")}
Gender: {props.get("gender", "")}
Article Type: {props.get("article_type", "")}
Color: {props.get("color", "")}
Usage: {props.get("usage", "")}
Season: {props.get("season", "")}
Year: {props.get("year", "")}
Image Path: {props.get("image_path", "")}
Description: {props.get("content", "")}
""".strip()

        context_parts.append(one_product_context)

    return "\n\n".join(context_parts)


def main():
    print("Loading ColBERT model ...")
    colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    print("ColBERT is ready.\n")

    client = weaviate.connect_to_local(port=8079, grpc_port=50050)
    collection = client.collections.get("StoreDocs")

    try:
        query = input("Enter your query: ").strip()

        parsed = parse_query(query)

        print("\nPARSED QUERY")
        print(parsed)

        retrieval_result = run_staged_retrieval(collection, parsed, limit=20)

        stage = retrieval_result["stage"]
        filters_used = retrieval_result["filters_used"]
        candidates = retrieval_result["candidates"]

        print("\nRETRIEVAL STAGE")
        print(stage)

        print("\nFILTERS USED")
        print(filters_used)

        print("\nWEAVIATE RESULTS")
        if not candidates:
            print("No candidates found")
            return

        print_candidates("WEAVIATE RESULTS", candidates)

        valid_candidates = [
            obj for obj in candidates
            if obj.properties.get("content")
        ]

        if not valid_candidates:
            print("\nNo valid candidates with content found for reranking.")
            return

        documents = [obj.properties["content"] for obj in valid_candidates]

        results = colbert.rerank(
            query=query,
            documents=documents,
            k=min(5, len(documents))
        )

        top_docs = []
        for r in results:
            obj = valid_candidates[r["result_index"]]
            top_docs.append(obj)

        print("\nCOLBERT RESULTS")
        for i, (r, doc) in enumerate(zip(results, top_docs), start=1):
            print(f"\nRank {i} | score={r['score']}")
            print("-" * 50)
            print(doc.properties)

        context = build_context(top_docs, stage)

        print("\nFINAL CONTEXT SENT TO LLM")
        print("-" * 50)
        print(context)

        print("\nGENERATING ANSWER WITH LLM...\n")
        answer = generate_answer(query, context)

        print("\nFINAL ANSWER")
        print("-" * 50)
        print(answer)

        print("\nPARSED QUERY")
        print(parsed)

        print("\nSEARCH QUERY")
        print(parsed["search_query"])

        print("\nSTRICT FILTERS")
        print(clean_filters(parsed.get("strict_filters", {})))

        print("\nSOFT FILTERS")
        print(clean_filters(parsed.get("soft_filters", {})))

    finally:
        client.close()


if __name__ == "__main__":
    main()