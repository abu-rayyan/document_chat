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
collection_name = "document_collection"
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
    document_id: str  # The document_id of the document to search for
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

def store_document(text: str):
    # Generate a unique document_id
    document_id = str(uuid.uuid4())  # Document ID is generated via UUID

    # Embed the document text
    embedding = embedding_model.encode([text])

    # Get the current time to store as the last access time
    last_access_time = time.time()

    # Insert the document into Chroma collection, storing the document_id as the document ID
    collection.add(
        documents=[text],
        embeddings=embedding,
        ids=[document_id],  # Use document_id as the document ID
        metadatas=[{"last_access_time": last_access_time}]  # Add last_access_time to metadata
    )

    # Store document in the in-memory store for manual filtering
    document_store[document_id] = {
        'text': text,
        'embedding': embedding[0],  # Store the first embedding in the list
        'last_access_time': last_access_time  # Add last_access_time to in-memory store
    }

    return document_id

def retrieve_relevant_documents(query: str, document_id: str, top_k: int = 5):
    try:
        # Step 1: Embed the query to get the query embedding
        query_embedding = embedding_model.encode([query])

        # Step 2: Check if the document_id exists in the document_store
        if document_id not in document_store:
            print("--------------##%^^^^^^^^^^^^^^^^&&&&&&&&&***********")
            print ("document not found")
            print("--------------##%^^^^^^^^^^^^^^^^&&&&&&&&&***********")
            return {"message": f"Document with ID {document_id} does not exist."}

        # Step 3: Remove all embeddings that do not belong to the given document_id
        # This is where you remove the embeddings from Chroma collection that are not relevant to the document_id.
        filtered_documents = [document_store[document_id]]
        filtered_embeddings = [document_store[document_id]['embedding']]  # Only keep embeddings for the selected document_id

        # Step 4: Perform the query only on the filtered embeddings in Chroma
        results = collection.query(
            query_embeddings=filtered_embeddings,
            n_results=top_k
        )

        # Step 5: Return the relevant documents directly
        return results['documents']  # Since we're already filtering by document_id, no need for additional filtering

    except Exception as e:
        # Catch any unexpected errors and return a message
        print(f"An error occurred: {e}")
        return {"error": "An unexpected error occurred. Please try again."}

# Function to delete inactive documents directly from Chroma
def delete_inactive_sessions():
    current_time = time.time()

    # Retrieve all document IDs to check their inactivity
    all_documents = collection.get_all_documents()  # Assuming Chroma supports this method to get all docs
    
    to_delete = []  # List to store document_ids to be deleted
    for doc in all_documents:
        # Access metadata directly from Chroma collection
        metadata = doc.get('metadata', {})
        last_access_time = metadata.get('last_access_time', 0)

        # If the document hasn't been accessed for too long, mark it for deletion
        if current_time - last_access_time > SESSION_TIMEOUT_THRESHOLD:
            to_delete.append(doc['id'])

    # Delete inactive documents from Chroma collection
    if to_delete:
        # If there are documents to delete, call the delete method on Chroma
        collection.delete(ids=to_delete)
        print(f"Deleted {len(to_delete)} inactive documents.")
    else:
        print("No inactive documents to delete.")

@app.post("/upload_pdf/")
async def upload_pdf(pdf_file: UploadFile = File(...)):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_file)
    # Store the document text and embeddings
    document_id = store_document(text)
    return {"document_id": document_id}  # Return document_id to the client


@app.post("/query/")
async def handle_query(query_request: QueryRequest):
    document_id = query_request.document_id  # Extract document_id from the body
    query = query_request.query  # Extract the query text from the body
    
    # Retrieve relevant documents based on the document_id and query
    relevant_docs = retrieve_relevant_documents(query, document_id)
    
    if not relevant_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # Ensure relevant_docs is a list of strings before joining
    context = "\n".join([str(doc) for doc in relevant_docs])

    # For simplicity, we just return the context as the "answer"
    # In practice, you would use a model like GPT-4 to generate the final answer
    answer = f"Relevant documents: \n{context}"    
    return {"answer": answer}
