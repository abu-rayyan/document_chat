from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import fitz  # PyMuPDF for PDF text extraction
import io
import docx
import uuid
import time
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize Chroma client
client = chromadb.Client()

# Create or access a collection in Chroma
collection_name = "document_collection"
collection = client.create_collection(collection_name)

# Initialize the Sentence Transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory store for documents (for filtering purposes)
document_store = []

app = FastAPI()

# Pydantic model for request body
class QueryRequest(BaseModel):
    session_id: str  # Session ID to filter documents
    query: str  # The actual query text to search for

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file: UploadFile):
    # Convert the file into a byte stream
    file_bytes = pdf_file.file.read()
    
    # Open the byte stream with fitz (PyMuPDF)
    pdf_reader = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
    
    text = ""
    for page in pdf_reader:
        text += page.get_text()
    
    return text

# Helper function to extract text from DOCX
def extract_text_from_docx(docx_file: UploadFile):
    doc = docx.Document(docx_file.file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def store_document(session_id: str, text: str):
    # Embed the document text
    embedding = embedding_model.encode([text])

    # Insert the document into Chroma collection, storing the session_id in metadata
    collection.add(
        documents=[text],
        embeddings=embedding,
        metadatas=[{"session_id": session_id}],  # Add session_id to metadata
        ids=[session_id]  # Optionally use session_id as the document ID
    )
    
    # Also store document in an in-memory list for manual filtering
    document_store.append({
        "text": text,
        "embedding": embedding[0],  # Store the first embedding in the list
        "session_id": session_id
    })

def retrieve_relevant_documents(query: str, session_id: str = None, top_k: int = 5):
    # Embed the query
    query_embedding = embedding_model.encode([query])

    # 1. If session_id is provided, filter documents manually by session_id before searching
    if session_id:
        # Filter documents in memory by session_id
        filtered_documents = [
            doc for doc in document_store
            if doc['session_id'] == session_id  # Ensure the session_id matches
        ]
    else:
        # If no session_id, use all documents
        filtered_documents = document_store

    # 2. Now, perform similarity search only on the filtered documents
    filtered_embeddings = [doc['embedding'] for doc in filtered_documents]

    # Perform similarity search on the filtered subset
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    # 3. Extract relevant documents
    relevant_documents = [result for result in results['documents']]
    print ("==============================================================")
    print ("relevant documents are:", relevant_documents)
    print ("==============================================================")

    return relevant_documents

# Function to delete inactive sessions (not implemented here but can be done based on payload metadata)
def delete_inactive_sessions():
    # Chroma currently doesn't have built-in support for automatic deletion by inactivity.
    # You can implement session expiration based on the `last_access_time` metadata in your application.
    pass

@app.post("/upload_pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    # Generate a session ID
    session_id = str(uuid.uuid4())
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_file)
    # Store the document text and embeddings
    store_document(session_id, text)
    return {"session_id": session_id}

@app.post("/upload_docx/")
async def upload_docx(docx_file: UploadFile = File(...)):
    # Generate a session ID
    session_id = str(uuid.uuid4())
    # Extract text from DOCX
    text = extract_text_from_docx(docx_file)
    # Store the document text and embeddings
    store_document(session_id, text)
    return {"session_id": session_id}

@app.post("/query/")
async def handle_query(query_request: QueryRequest):
    session_id = query_request.session_id  # Extract session_id from the body
    query = query_request.query  # Extract the query text from the body
    
    # Retrieve relevant documents based on the query and session_id
    relevant_docs = retrieve_relevant_documents(query, session_id)
    
    if not relevant_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # Ensure relevant_docs is a list of strings before joining
    context = "\n".join([str(doc) for doc in relevant_docs])

    # For simplicity, we just return the context as the "answer"
    # In practice, you would use a model like GPT-4 to generate the final answer
    answer = f"Relevant documents: \n{context}"
    
    return {"answer": answer}
