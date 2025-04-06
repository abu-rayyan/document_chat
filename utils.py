import chromadb

# Initialize Chroma client
client = chromadb.Client()

# List all existing collections in ChromaDB
collections = client.list_collections()
print ("collections are:", collections)

for col in collections:
    print("collection name is",col.name)

