import os
import json
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from openai import OpenAI
import warnings

warnings.filterwarnings("ignore")

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-small"

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = ROOT_DIR / "chromadb_store"
ARCHIVE_DIR = ROOT_DIR / "Data/Food-dataset/food/archive (7)/Indian Food Images/Indian Food Images"
MAPPING_FILE = ROOT_DIR / "Data/mapping_food_images.json"

# Connect to DB
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
food_col = chroma_client.get_collection("food_text")

print("🔍 Auto-mapping 80 archive categories using AI Search...")

# Load existing mapping
with open(MAPPING_FILE, "r") as f:
    mapping = json.load(f)

# Get the 80 categories
categories = [d for d in os.listdir(ARCHIVE_DIR) if os.path.isdir(ARCHIVE_DIR / d)]
categories.sort()

new_mappings_count = 0

for cat in categories:
    if cat in mapping:
        continue # Already mapped
        
    # Create search query from category name
    search_query = cat.replace("_", " ")
    
    # Generate vector for search
    try:
        res = openai_client.embeddings.create(model=EMBED_MODEL, input=[search_query])
        query_vector = res.data[0].embedding
        
        # Search DB for the closest match
        results = food_col.query(
            query_embeddings=[query_vector],
            n_results=1
        )
        
        best_id = results['ids'][0][0]
        best_name = results['metadatas'][0][0].get('food_name', 'Unknown')
        dist = results['distances'][0][0]
        
        # Add to mapping
        mapping[cat] = best_id
        new_mappings_count += 1
        print(f"  → Mapped: [{cat}] ---> '{best_name}' (ID: {best_id}) [Score: {1-dist:.2f}]")
        
    except Exception as e:
        print(f"  ❌ Failed to map {cat}: {e}")

# Save the updated mapping back to file
with open(MAPPING_FILE, "w") as f:
    json.dump(mapping, f, indent=2)

print(f"\n✅ DONE! Added {new_mappings_count} new categories to mapping_food_images.json")
