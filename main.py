import uuid
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import fitz  # PyMuPDF for PDF text extraction
import io
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize Chroma client
client = chromadb.Client()

# Create or access a collection in Chroma
collection_name = "document_collection_2"
collection = client.create_collection(collection_name)

# Initialize the Sentence Transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Session timeout threshold (in seconds)
SESSION_TIMEOUT_THRESHOLD = 3600  # 1 hour

# In-memory store for documents (for filtering purposes)
document_store = {}

app = FastAPI()

# Pydantic model for request body
class QueryRequest(BaseModel):
    document_id: str
    query: str

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file: UploadFile):
    file_bytes = pdf_file.file.read()
    pdf_reader = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    text = ""
    for page in pdf_reader:
        text += page.get_text()
    return text

# Function to chunk text by paragraphs
def chunk_text_by_paragraphs(text: str, chunk_size=50):
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < chunk_size:
            current_chunk += paragraph + '\n'
        else:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + '\n'
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Store document and split it into chunks
def store_document(text: str):
    document_id = str(uuid.uuid4())
    chunks = chunk_text_by_paragraphs(text)
    embeddings = embedding_model.encode(chunks)

    ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"document_id": document_id} for _ in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    for i, chunk in enumerate(chunks):
        document_store[f"{document_id}_chunk_{i}"] = {
            'text': chunk,
            'embedding': embeddings[i],
            'document_id': document_id,
        }

    return document_id

# Load document_store from Chroma at startup
def populate_document_store():
    results = collection.get()
    for doc, meta, id_ in zip(results["documents"], results["metadatas"], results["ids"]):
        document_store[id_] = {
            "text": doc,
            "embedding": None,
            "document_id": meta.get("document_id")
        }

populate_document_store()

# Retrieve relevant chunks using Chroma's built-in filtering and similarity
def retrieve_relevant_documents(query: str, document_id: str, top_k: int = 5):
    try:
        query_embedding = embedding_model.encode([query])

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={"document_id": document_id}
        )

        top_chunks = results['documents'][0] if results['documents'] else []

        print ("^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^")
        print ("^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^")
        print ("top chunks are:", top_chunks)
        print ("^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^")
        print ("^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^^_^")

        # Count total chunks for the given document_id
        total_chunks = sum(1 for doc in document_store.values() if doc['document_id'] == document_id)

        return {
            "top_chunks": top_chunks,
            "total_chunks": total_chunks
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"error": "An unexpected error occurred."}

@app.post("/upload_pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    text = extract_text_from_pdf(pdf_file)
    document_id = store_document(text)
    return {"document_id": document_id}

@app.post("/query/")
async def handle_query(query_request: QueryRequest):
    document_id = query_request.document_id
    query = query_request.query

    result = retrieve_relevant_documents(query, document_id)
    if not result or "top_chunks" not in result:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    context = "\n".join(result["top_chunks"])
    answer = f"Relevant documents: \n{context}"

    return {
        "answer": answer,
        "total_chunks": result["total_chunks"]
    }

@app.delete("/delete_document/{document_id}")
async def delete_document(document_id: str):
    chunk_ids = [doc_id for doc_id, doc in document_store.items() if doc["document_id"] == document_id]
    if not chunk_ids:
        raise HTTPException(status_code=404, detail="Document ID not found")

    collection.delete(ids=chunk_ids)

    for chunk_id in chunk_ids:
        del document_store[chunk_id]

    print(f"Deleted document {document_id} with {len(chunk_ids)} chunks")
    return {"message": f"Deleted document {document_id} successfully", "deleted_chunks": len(chunk_ids)}
