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
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5",
)

INPUT_FILES = [
    Path(
        "data/synthetic_patient_timeline/"
        "article_holdout_pediatric_oncology_timeline.jsonl"
    ),
    Path(
        "data/synthetic_patient_timeline/"
        "article_holdout_pediatric_oncology_timeline_extra150.jsonl"
    ),
]


def load_records():
    records = []

    for path in INPUT_FILES:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    return records


def build_uuid(record):
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            record["document_url"],
        )
    )


def main():
    records = load_records()

    if len(records) != 18000:
        raise SystemExit(
            f"ERROR: expected 18000 records, found {len(records)}"
        )

    print("Records to ingest:", len(records))
    print("Embedding model:", EMBEDDING_MODEL_NAME)

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=AuthApiKey(
            WEAVIATE_API_KEY
        ),
    )

    try:
        collection = client.collections.get(
            "MedicalDocs"
        )

        inserted = 0
        skipped = 0
        failed = 0

        for index, record in enumerate(
            records,
            start=1,
        ):
            object_uuid = build_uuid(record)

            if collection.data.exists(object_uuid):
                skipped += 1
                continue

            content = record["content"]

            text_for_embedding = (
                "Represent this sentence for searching "
                "relevant passages: "
                + content
            )

            vector = model.encode(
                text_for_embedding,
                normalize_embeddings=True,
            ).tolist()

            properties = {
                "content": content,
                "question": "",
                "answer": "",
                "category":
                    "pediatric_patient_timeline",
                "document_source":
                    "Semi-Synthetic Pediatric Oncology Treatment Timeline Dataset",
                "question_type": "timeline",
                "document_url":
                    record["document_url"],
            }

            try:
                collection.data.insert(
                    properties=properties,
                    vector={"default": vector},
                    uuid=object_uuid,
                )
                inserted += 1

            except Exception as exc:
                failed += 1
                print(
                    "FAILED:",
                    record["document_url"],
                    exc,
                )

            if index % 500 == 0:
                print(
                    f"Processed {index}/18000 | "
                    f"Inserted {inserted} | "
                    f"Skipped {skipped} | "
                    f"Failed {failed}",
                    flush=True,
                )

        print("\nDONE")
        print("Inserted:", inserted)
        print("Skipped:", skipped)
        print("Failed:", failed)

    finally:
        client.close()


if __name__ == "__main__":
    main()
