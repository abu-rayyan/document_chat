import uuid
import time
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, models
import fitz  # PyMuPDF
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
# model_path = "./bge_large_en_v1.5"
# word_embedding_model = models.Transformer(model_path)
# word_embedding_model = models.Transformer(model_path, use_auth_token=False, cache_dir=model_path)

# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
# embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
embedding_model= SentenceTransformer("BAAI/bge-large-en-v1.5")

# FAISS index setup
embedding_dim = embedding_model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(embedding_dim)
id_mapping = {}  # Maps FAISS index to chunk_id
chunk_store = {}  # Maps chunk_id to chunk data

# Session timeout threshold
SESSION_TIMEOUT_THRESHOLD = 3600

app = FastAPI()

# Pydantic model for query requests
class QueryRequest(BaseModel):
    document_id: str
    query: str

def extract_text_from_pdf(pdf_file: UploadFile):
    file_bytes = pdf_file.file.read()
    pdf_reader = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    return "\n".join(page.get_text() for page in pdf_reader)

def chunk_text_by_paragraphs(text: str, chunk_size=500):
    paragraphs = text.split('\n')
    chunks, current_chunk = [], ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < chunk_size:
            current_chunk += paragraph + '\n'
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + '\n'
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def store_document(text: str):
    document_id = str(uuid.uuid4())
    chunks = chunk_text_by_paragraphs(text)
    embeddings = embedding_model.encode(chunks)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{document_id}_chunk_{i}"
        faiss_index.add(np.array([embedding]))
        index_pos = faiss_index.ntotal - 1
        id_mapping[index_pos] = chunk_id
        chunk_store[chunk_id] = {
            "text": chunk,
            "embedding": embedding,
            "document_id": document_id,
            "last_access_time": time.time()
        }

    return document_id

def retrieve_relevant_documents(query: str, document_id: str, top_k: int = 5):
    query_embedding = embedding_model.encode([query])[0]
    filtered = [(idx, chunk_id, data) for idx, (chunk_id, data) in enumerate(chunk_store.items()) if data["document_id"] == document_id]

    if not filtered:
        return {"message": f"No chunks found for document_id {document_id}"}

    embeddings = np.array([data["embedding"] for _, _, data in filtered])
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_chunks = [filtered[i][2]["text"] for i in top_indices]

    print("^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^")
    print("top chunks are:", top_chunks)
    print("^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^")

    return {"top_chunks": top_chunks, "total_chunks": len(filtered)}

@app.post("/upload_pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    text = extract_text_from_pdf(pdf_file)
    document_id = store_document(text)
    return {"document_id": document_id}

@app.post("/query/")
async def handle_query(query_request: QueryRequest):
    result = retrieve_relevant_documents(query_request.query, query_request.document_id)
    if "top_chunks" not in result:
        raise HTTPException(status_code=404, detail=result.get("message", "Query failed"))
    context = "\n".join(result["top_chunks"])
    return {"answer": f"Relevant documents:\n{context}", "total_chunks": result["total_chunks"]}

@app.delete("/delete_document/{document_id}")
async def delete_document(document_id: str):
    to_delete = [idx for idx, chunk_id in id_mapping.items() if chunk_store.get(chunk_id, {}).get("document_id") == document_id]
    if not to_delete:
        raise HTTPException(status_code=404, detail="Document ID not found")

    for idx in sorted(to_delete, reverse=True):
        faiss_index.remove_ids(np.array([idx]))
        chunk_id = id_mapping.pop(idx, None)
        if chunk_id:
            chunk_store.pop(chunk_id, None)

    print(f"Deleted document {document_id} with {len(to_delete)} chunks")
    return {"message": f"Deleted document {document_id} successfully", "deleted_chunks": len(to_delete)}
