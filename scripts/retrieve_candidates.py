import weaviate
from weaviate.classes.query import Filter
from query_parser import parse_query

client = weaviate.connect_to_local(port=8079, grpc_port=50050)
collection = client.collections.get("StoreDocs")


def build_filters(filters: dict):
    where_filter = None

    for key, value in filters.items():
        current_filter = Filter.by_property(key).equal(value)

        if where_filter is None:
            where_filter = current_filter
        else:
            where_filter = where_filter & current_filter

    return where_filter


query = input("Enter your query: ").strip()

parsed = parse_query(query)
filters = parsed["filters"].copy()

# article_type را فعلاً از فیلتر سخت برمی‌داریم
# و بعداً روی نتایج post-filter می‌کنیم
requested_article_type = filters.pop("article_type", None)

where_filter = build_filters(filters)

response = collection.query.hybrid(
    query=parsed["search_query"],
    filters=where_filter,
    alpha=0.5,
    limit=20
)

candidates = response.objects

if requested_article_type:
    filtered_candidates = []
    for obj in candidates:
        article_type_value = obj.properties.get("article_type", "")
        if article_type_value and article_type_value.lower() == requested_article_type.lower():
            filtered_candidates.append(obj)
    if filtered_candidates:
        candidates = filtered_candidates

print("\nPARSED QUERY")
print(parsed)

print("\nRESULTS")
for i, obj in enumerate(candidates[:10], start=1):
    print(f"\nResult {i}")
    print("-" * 50)
    print("product_name:", obj.properties.get("product_name", ""))
    print("gender:", obj.properties.get("gender", ""))
    print("color:", obj.properties.get("color", ""))
    print("article_type:", obj.properties.get("article_type", ""))
    print("usage:", obj.properties.get("usage", ""))
    print("content:", obj.properties.get("content", "")[:200])

client.close()