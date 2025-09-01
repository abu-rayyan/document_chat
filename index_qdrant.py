"""
Index markdown/text documents into Qdrant (embedded/local mode) for RAG, using a YAML config file.

Config file (config.yaml) example:
----------------------------------
data_dir: ./sample_docs
collection: finance_policy
db_path: ./qdrant_data
model: sentence-transformers/all-MiniLM-L6-v2
max_tokens: 180
overlap: 40

Usage:
    pip install -r requirements.txt
    python index_qdrant.py

Notes:
- Uses embedded Qdrant via QdrantClient(path=...).
- All payloads include: doc_id, source_path, section, chunk_id, text.
- Returns chunk-level references for QA.
"""

import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Iterable, Tuple

import yaml
import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = ["data_dir", "collection", "db_path"]
    missing = [k for k in required if k not in cfg or cfg[k] is None]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    return cfg


def read_text_files(data_dir: Path):
    paths = list(data_dir.rglob("*.md")) + list(data_dir.rglob("*.txt"))
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            docs.append((str(p), f.read()))
    return docs


def split_by_headings(md_text: str):
    """Very simple markdown splitter: returns list of (section_title, section_text)."""
    lines = md_text.splitlines()
    sections = []
    current_title = "Introduction"
    buf = []
    for line in lines:
        if re.match(r"^\s*#{1,6}\s+", line):
            if buf:
                sections.append((current_title.strip(), "\n".join(buf).strip()))
                buf = []
            current_title = re.sub(r"^\s*#{1,6}\s+", "", line).strip()
        else:
            buf.append(line)
    if buf:
        sections.append((current_title.strip(), "\n".join(buf).strip()))
    return sections


def chunk_text(text: str, max_tokens: int = 180, overlap: int = 40):
    """Simple word-based chunking (approximate tokens)."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + max_tokens]
        chunks.append(" ".join(chunk))
        if i + max_tokens >= len(words):
            break
        i += max_tokens - overlap
    return chunks


def ensure_collection(client: QdrantClient, name: str, vector_size: int):
    try:
        client.get_collection(name)
        exists = True
    except Exception:
        exists = False

    if not exists:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def build_points(embeddings, payloads):
    points = []
    for vec, pl in zip(embeddings, payloads):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload=pl,
            )
        )
    return points


def main():
    cfg = load_config("config.yaml")
    data_dir = Path(cfg["data_dir"])
    collection = cfg["collection"]
    db_path = cfg["db_path"]
    model_name = cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    max_tokens = int(cfg.get("max_tokens", 180))
    overlap = int(cfg.get("overlap", 40))

    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir.resolve()}")

    # Embedded Qdrant
    client = QdrantClient(path=db_path)

    # Embedder
    model = SentenceTransformer(model_name)
    vector_size = model.get_sentence_embedding_dimension()

    ensure_collection(client, collection, vector_size)

    docs = read_text_files(data_dir)
    if not docs:
        print(f"No .md/.txt files found in {data_dir}")
        return

    all_vectors = []
    all_payloads = []

    for src_path, content in docs:
        doc_id = str(uuid.uuid4())
        sections = split_by_headings(content)
        for section_title, section_text in sections:
            chunks = chunk_text(section_text, max_tokens=max_tokens, overlap=overlap)
            if not chunks:
                continue

            embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
            for idx, (emb, chunk) in enumerate(zip(embeddings, chunks)):
                payload = {
                    "doc_id": doc_id,
                    "source_path": src_path,
                    "section": section_title,
                    "chunk_id": idx,
                    "text": chunk,
                }
                all_vectors.append(emb)
                all_payloads.append(payload)

    if not all_vectors:
        print("No chunks produced; check your documents.")
        return

    points = build_points(np.vstack(all_vectors), all_payloads)

    BATCH = 512
    for i in tqdm(range(0, len(points), BATCH), desc="Upserting to Qdrant"):
        client.upsert(collection_name=collection, wait=True, points=points[i : i + BATCH])

    print(f"Indexed {len(points)} chunks into collection '{collection}' at {db_path}")
    print("\nExample: quick search")
    print(
        "python -c \"from qdrant_client import QdrantClient; from sentence_transformers import SentenceTransformer; "
        f"c=QdrantClient(path='{db_path}'); m=SentenceTransformer('{model_name}'); "
        "q='What is the record retention period?'; v=m.encode([q], normalize_embeddings=True)[0]; "
        f"print(c.search(collection_name='{collection}', query_vector=v.tolist(), with_payload=True, limit=5))\""
    )


if __name__ == "__main__":
    main()
