from chromadb import PersistentClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma"

def inspect_database():
    # Initialize Chroma client
    client = PersistentClient(path=CHROMA_PATH)
    
    # Get the collection (replace 'my_collection' with the name used in create_database.py)
    collection = client.get_collection(name="86bce1c3-d2a4-4967-9239-271b26cb393e")
    
    # Retrieve all items in the collection
    results = collection.get(include=["embeddings", "documents", "metadatas"])
    
    print(f"Total items in collection: {len(results['ids'])}")
    
    # Print a few items for inspection
    for i, (id_, embedding, document, metadata) in enumerate(zip(
        results["ids"], results["embeddings"], results["documents"], results["metadatas"]
    )):
        if i >= 5:  # Limit to first 5 items for brevity
            break
        print(f"\nItem {i+1}:")
        print(f"ID: {id_}")
        print(f"Embedding (first 5 values): {embedding[:5]}... (length: {len(embedding)})")
        print(f"Document: {document[:100]}...")  # Truncate for readability
        print(f"Metadata: {metadata}")

if __name__ == "__main__":
    inspect_database()