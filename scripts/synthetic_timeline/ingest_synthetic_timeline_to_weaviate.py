import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from sentence_transformers import SentenceTransformer


load_dotenv(".env.cloud")

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

INPUT_FILE = Path("data/synthetic_patient_timeline/synthetic_pediatric_oncology_timeline.jsonl")


def load_records():
    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def build_uuid(record):
    unique_key = record["document_url"]
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_key))


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    records = load_records()
    print(f"Found timeline records: {len(records)}")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
    )

    try:
        collection = client.collections.get("MedicalDocs")

        success_count = 0
        failed_count = 0

        for index, record in enumerate(records, start=1):
            content = record["content"]

            text_for_embedding = "Represent this sentence for searching relevant passages: " + content
            vector = model.encode(text_for_embedding, normalize_embeddings=True).tolist()

            properties = {
                "content": content,
                "question": "",
                "answer": "",
                "category": "pediatric_patient_timeline",
                "document_source": "Semi-Synthetic Pediatric Oncology Treatment Timeline Dataset",
                "question_type": "timeline",
                "document_url": record["document_url"],
            }

            try:
                collection.data.insert(
                    properties=properties,
                    vector={"default": vector},
                    uuid=build_uuid(record),
                )
                success_count += 1

            except Exception as e:
                failed_count += 1
                print(f"Failed to insert {record['document_url']}: {e}")

            if index % 100 == 0:
                print(f"Processed {index}/{len(records)}")

        print("=" * 80)
        print(f"Inserted: {success_count}")
        print(f"Failed: {failed_count}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
