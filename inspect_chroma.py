import sqlite3
import numpy as np

def inspect_chroma_database():
    # Path to the SQLite database
    db_path = "chroma/chroma.sqlite3"
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query the embeddings table
    cursor.execute("SELECT id, segment_id, embedding_id, seq_id, created_at FROM embeddings LIMIT 5")
    for row in cursor.fetchall():
        id_, segment_id,embedding_id,seq_id, created_at = row
        # Deserialize the embedding blob (assuming float32 vectors)
        embedding = np.frombuffer(seq_id, dtype=np.float32)
        print(f"\nID: {id_}")
        print(f"Embedding (first 5 values): {embedding[:5]}... (length: {len(embedding)})")


    # Close the connection
    conn.close()

if __name__ == "__main__":
    inspect_chroma_database()