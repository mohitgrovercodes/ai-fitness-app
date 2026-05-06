import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from typing import List, Dict, Any
from app.core.database import db_manager
from app.core.config import settings
from app.utils.logger import logger
from app.utils.gif_utils import media_matcher


class TrainingRAGTool:
    """
    Step 7.3: TRAINING RAG TOOL.
    Queries the 2,800+ exercise items in ChromaDB safely.
    Maintains strict structural compatibility with NutritionRAGTool.
    """
    def __init__(self):
        # Manager is a Singleton
        self.is_connected = db_manager._client is not None
        if self.is_connected:
            logger.info("✅ [Training Tool] Connected to shared ChromaDB via Manager.")
        else:
            logger.error("❌ [Training Tool] Could not connect to ChromaDB.")
            
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
            logger.error(f"❌ [Training Tool] Embedding Error: {e}")
            return []

    async def search(self, query: str, n_results: int = 5, multiplier: float = 1.0) -> List[Dict[str, Any]]:
        """
        Single Semantic Search for exercises.
        Unblocked using asyncio.to_thread.
        """
        if not self.is_connected:
            return []
            
        logger.info(f"🏋️ [Training Tool] Searching DB for: '{query}'")
        try:
            query_vector = await self.get_embedding(query)
            if not query_vector:
                return []
                
            # Use the Manager's single-threaded query runner
            results = await db_manager.run_query(
                collection_name="exercise_text",
                query_embeddings=[query_vector],
                n_results=n_results
            )
            return self._process_results(results)
        except Exception as e:
            logger.error(f"❌ [Training Tool] Search Error: {e}")
            return []

    async def multi_query_search(self, query: str, sub_queries: List[str], multiplier: float = 1.0) -> List[Dict[str, Any]]:
        """
        Multi-Query Expansion for complex workout requests.
        """
        if not self.is_connected:
            return []
            
        logger.info(f"🏋️ [Training Tool] Multi-Query Search with {len(sub_queries)} variations...")
        all_results = {}

        try:
            tasks_embeddings = [self.get_embedding(sq) for sq in sub_queries]
            vectors = await asyncio.gather(*tasks_embeddings)

            # Parallelize ChromaDB queries
            db_tasks = []
            for v in vectors:
                if v:
                    db_tasks.append(asyncio.to_thread(self.collection.query, query_embeddings=[v], n_results=3))
            
            if not db_tasks:
                return []
                
            db_results_list = await asyncio.gather(*db_tasks)

            for results in db_results_list:
                for item in self._process_results(results):
                    all_results[item['id']] = item

            logger.info(f"  → Merged {len(all_results)} unique results from parallel multi-query")
            return list(all_results.values())
        except Exception as e:
            logger.error(f"❌ [Training Tool] Multi-Query Search Error: {e}")
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
                "media": media_matcher.get_media(meta.get('name', '')),
                "score": round(1 - dist, 3)
            })
        return cleaned
