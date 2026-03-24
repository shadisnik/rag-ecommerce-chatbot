from pathlib import Path

from tqdm import tqdm
import weaviate
from weaviate.util import generate_uuid5


def empty_product_metadata() -> dict:
    return {
        "product_name": "",
        "gender": "",
        "category": "",
        "sub_category": "",
        "article_type": "",
        "color": "",
        "season": "",
        "year": None,
        "usage": "",
        "image_path": ""
    }


def parse_product_file(text: str) -> dict:
    metadata = empty_product_metadata()

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("Product Name:"):
            metadata["product_name"] = line.replace("Product Name:", "").strip()

        elif line.startswith("Gender:"):
            metadata["gender"] = line.replace("Gender:", "").strip()

        elif line.startswith("Category:"):
            metadata["category"] = line.replace("Category:", "").strip()

        elif line.startswith("Sub Category:"):
            metadata["sub_category"] = line.replace("Sub Category:", "").strip()

        elif line.startswith("Article Type:"):
            metadata["article_type"] = line.replace("Article Type:", "").strip()

        elif line.startswith("Color:"):
            metadata["color"] = line.replace("Color:", "").strip()

        elif line.startswith("Season:"):
            metadata["season"] = line.replace("Season:", "").strip()

        elif line.startswith("Year:"):
            year_value = line.replace("Year:", "").strip()
            try:
                metadata["year"] = int(float(year_value))
            except ValueError:
                metadata["year"] = None

        elif line.startswith("Usage:"):
            metadata["usage"] = line.replace("Usage:", "").strip()

        elif line.startswith("Image Path:"):
            metadata["image_path"] = line.replace("Image Path:", "").strip()

    return metadata


def chunk_text_with_overlap(text: str, chunk_size: int = 100, overlap_fraction: float = 0.25) -> list[str]:
    text_words = text.split()
    overlap_int = int(chunk_size * overlap_fraction)
    step = max(chunk_size - overlap_int, 1)

    chunks = []

    for i in range(0, len(text_words), step):
        chunk_words = text_words[i:i + chunk_size]
        if not chunk_words:
            continue
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

    return chunks


def build_chunk_objs(text: str, source_path: str, doc_type: str) -> list[dict]:
    chunks = chunk_text_with_overlap(text)
    chunk_objs = []

    for i, chunk in enumerate(chunks):
        chunk_obj = {
            "content": chunk,
            "source": f"{source_path}#chunk_{i}",
            "doc_type": doc_type,
        }
        chunk_objs.append(chunk_obj)

    return chunk_objs


weaviate_client = weaviate.connect_to_local(
    port=8079,
    grpc_port=50050
)

try:
    collection = weaviate_client.collections.get("StoreDocs")

    all_files = list(Path("docs").rglob("*.md"))
    print(f"Found {len(all_files)} markdown files.")

    objects_to_insert = []

    for file_path in tqdm(all_files, desc="Preparing documents"):
        text = file_path.read_text(encoding="utf-8").strip()
        folder_name = file_path.parent.name.lower()
        source_path = str(file_path).replace("\\", "/")

        if folder_name == "products":
            doc_type = "product"
            product_meta = parse_product_file(text)

            properties = {
                "content": text,
                "source": source_path,
                "doc_type": doc_type,
                "product_name": product_meta["product_name"],
                "gender": product_meta["gender"],
                "category": product_meta["category"],
                "sub_category": product_meta["sub_category"],
                "article_type": product_meta["article_type"],
                "color": product_meta["color"],
                "season": product_meta["season"],
                "year": product_meta["year"],
                "usage": product_meta["usage"],
                "image_path": product_meta["image_path"],
            }

            uuid = generate_uuid5(source_path)
            objects_to_insert.append({
                "properties": properties,
                "uuid": uuid
            })

        elif folder_name in {"policies", "faq", "support"}:
            doc_type = "policy" if folder_name == "policies" else folder_name

            chunk_objs = build_chunk_objs(
                text=text,
                source_path=source_path,
                doc_type=doc_type,
            )

            for obj in chunk_objs:
                objects_to_insert.append({
                    "properties": obj,
                    "uuid": generate_uuid5(obj["source"])
                })

        else:
            properties = {
                "content": text,
                "source": source_path,
                "doc_type": "unknown",
            }

            uuid = generate_uuid5(source_path)

            objects_to_insert.append({
                "properties": properties,
                "uuid": uuid
            })

    print(f"Prepared {len(objects_to_insert)} objects for insertion.")

    with collection.batch.fixed_size(batch_size=20, concurrent_requests=2) as batch:
        for obj in tqdm(objects_to_insert, desc="Inserting into Weaviate"):
            batch.add_object(
                properties=obj["properties"],
                uuid=obj["uuid"]
            )

    print("All documents inserted successfully!")

finally:
    weaviate_client.close()