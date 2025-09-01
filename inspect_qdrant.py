"""
Inspect chunks and payloads stored in Qdrant.
Reads config.yaml for db_path and collection.
Prints section, chunk_id, and start of text for each chunk.
"""

import yaml
from qdrant_client import QdrantClient


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("config.yaml")
    db_path = cfg["db_path"]
    collection = cfg["collection"]

    client = QdrantClient(path=db_path)

    # Scroll through all points
    scroll_res, _ = client.scroll(
        collection_name=collection,
        limit=50,  # adjust if you want more
        with_payload=True,
    )

    print(f"\n--- Chunks in collection '{collection}' ---\n")
    for i, point in enumerate(scroll_res, 1):
        pl = point.payload
        text_preview = pl.get("text", "")[:80].replace("\n", " ")
        print(f"[{i}] Section: {pl.get('section')} | Chunk: {pl.get('chunk_id')} | Text: {text_preview}...")


if __name__ == "__main__":
    main()
