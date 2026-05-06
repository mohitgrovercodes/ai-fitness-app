import chromadb
import os
from pathlib import Path

root_dir = Path("/Users/shubhamjadhav/Documents/Fit Bot/ai-fitness-app")
chroma_dir = root_dir / "chromadb_store"

client = chromadb.PersistentClient(path=str(chroma_dir))
collection = client.get_collection("exercise_text")

results = collection.peek(limit=5)
for i in range(len(results['ids'])):
    print(f"ID: {results['ids'][i]}")
    print(f"Metadata: {results['metadatas'][i]}")
    print("-" * 20)
