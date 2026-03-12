import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.services.embedding import embed_text
from backend.services.vector_db import store_document, create_collection
from backend.services.chunking import semantic_chunk


def ingest_file(filepath):

    print(f"Reading file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    print("Performing semantic chunking...")

    chunks = semantic_chunk(text)

    print(f"Total chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks):

        embedding = embed_text(chunk)

        metadata = {
            "text": chunk,
            "source": os.path.basename(filepath),
            "chunk_id": i
        }

        store_document(embedding, metadata)

    print("Document successfully ingested into vector database.")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python ingest_data.py <file_path>")
        sys.exit(1)

    create_collection()

    file_path = sys.argv[1]

    ingest_file(file_path)