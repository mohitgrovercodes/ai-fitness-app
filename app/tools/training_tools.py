import os
import chromadb
from pathlib import Path
from openai import AsyncOpenAI
from typing import List, Dict, Any
from app.core.config import settings

class TrainingRAGTool:
    """
    Step 7.3: TRAINING RAG TOOL.
    Queries the 2,800+ exercise items in ChromaDB safely.
    Maintains strict structural compatibility with NutritionRAGTool.
    """
    def __init__(self):
        # Root directory logic to find the DB
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.chroma_dir = self.root_dir / "chromadb_store"
        
        # Connect to ChromaDB with Error Handling
        try:
            self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self.collection = self.client.get_collection("exercise_text")
            self.is_connected = True
        except Exception as e:
            print(f"❌ [Training Tool] Database Connection Error: {e}")
            self.is_connected = False
            
        # Async OpenAI for non-blocking API calls
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def get_embedding(self, text: str) -> List[float]:
        """Generate vector for the search query asynchronously."""
        try:
            res = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=[text]
            )
            return res.data[0].embedding
        except Exception as e:
            print(f"❌ [Training Tool] Embedding Error: {e}")
            return []

    async def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Single Semantic Search for exercises.
        """
        if not self.is_connected:
            return []
            
        print(f"🏋️ [Training Tool] Searching DB for: '{query}'")
        try:
            query_vector = await self.get_embedding(query)
            if not query_vector:
                return []
                
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results
            )
            return self._process_results(results)
        except Exception as e:
            print(f"❌ [Training Tool] Search Error: {e}")
            return []

    async def multi_query_search(self, query: str, sub_queries: List[str]) -> List[Dict[str, Any]]:
        """
        Multi-Query Expansion for complex workout requests (e.g., full body routine).
        """
        if not self.is_connected:
            return []
            
        print(f"🏋️ [Training Tool] Multi-Query Search with {len(sub_queries)} variations...")
        all_results = {}

        try:
            for sub_query in sub_queries:
                q_vec = await self.get_embedding(sub_query)
                if not q_vec:
                    continue
                results = self.collection.query(query_embeddings=[q_vec], n_results=3)
                for item in self._process_results(results):
                    # Deduplicate by exercise ID
                    all_results[item['id']] = item

            print(f"  → Merged {len(all_results)} unique results from multi-query")
            return list(all_results.values())
        except Exception as e:
            print(f"❌ [Training Tool] Multi-Query Search Error: {e}")
            return []

    def _process_results(self, results) -> List[Dict[str, Any]]:
        """Process and structure ChromaDB results for the LLM."""
        cleaned = []
        if not results or 'ids' not in results or not results['ids']:
            return cleaned
            
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]

            cleaned.append({
                "id": results['ids'][0][i],
                "name": meta.get('name', 'Unknown Exercise'),
                "main_muscle": meta.get('main_muscle', 'N/A'),
                "equipment": meta.get('equipment', 'Bodyweight'),
                "preparation": meta.get('preparation', ''),
                "execution": meta.get('execution', ''),
                "target_muscles": meta.get('target_muscles', 'N/A'),
                "score": round(1 - dist, 3)
            })
        return cleaned
