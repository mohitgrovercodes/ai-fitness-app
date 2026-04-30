import os
import chromadb
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Any
from app.core.config import settings

class NutritionRAGTool:
    """
    Step 7.2: ADAPTIVE RAG TOOL for Nutrition.
    Queries the 39,358 food items in ChromaDB.
    """
    def __init__(self):
        # Root directory logic to find the DB
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.chroma_dir = self.root_dir / "chromadb_store"
        
        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_collection("food_text")
        
        # OpenAI for query embedding
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def get_embedding(self, text: str) -> List[float]:
        """Generate vector for the search query."""
        res = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return res.data[0].embedding

    async def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        ADAPTIVE RAG: Phase 1 — Single Semantic Search.
        """
        print(f"🔍 [Nutrition Tool] Searching DB for: '{query}'")
        query_vector = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        return self._process_results(results)

    async def multi_query_search(self, query: str, sub_queries: List[str]) -> List[Dict[str, Any]]:
        """
        ADAPTIVE RAG: Phase 2 — Multi-Query Expansion.
        Runs multiple semantic searches and merges results for complex queries.
        """
        print(f"🔍 [Nutrition Tool] Multi-Query Search with {len(sub_queries)} variations...")
        all_results = {}

        for sub_query in sub_queries:
            q_vec = self.get_embedding(sub_query)
            results = self.collection.query(query_embeddings=[q_vec], n_results=3)
            for item in self._process_results(results):
                # Deduplicate by food ID
                all_results[item['id']] = item

        print(f"  → Merged {len(all_results)} unique results from multi-query")
        return list(all_results.values())

    def _process_results(self, results) -> List[Dict[str, Any]]:
        """Process and filter ChromaDB results through the policy filter."""
        cleaned = []
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            food_name = meta.get('food_name', '').lower()

            # Policy Check: Restrict flagged foods
            if any(r in food_name for r in settings.RESTRICTED_FOODS):
                print(f"  ⚠️ Policy Alert: Skipping restricted item '{food_name}'")
                continue

            cleaned.append({
                "id": results['ids'][0][i],
                "food_name": meta.get('food_name'),
                "calories": meta.get('calories_kcal'),
                "protein": meta.get('protein_g'),
                "fat": meta.get('fat_g'),
                "carbs": meta.get('carbohydrates_g'),
                "score": round(1 - dist, 3)
            })
        return cleaned
