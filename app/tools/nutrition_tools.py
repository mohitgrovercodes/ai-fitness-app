import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from typing import List, Dict, Any
from app.core.database import db_manager
from app.core.config import settings
from app.utils.logger import logger

class NutritionRAGTool:
    """
    Step 7.2: ADAPTIVE RAG TOOL for Nutrition.
    Queries the 39,358 food items in ChromaDB safely.
    """
    def __init__(self):
        # Manager is a Singleton and uses lazy loading.
        self.is_connected = True
        logger.info("✅ [Nutrition Tool] Ready to connect to shared ChromaDB via Manager.")
        
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
            logger.error(f"❌ [Nutrition Tool] Embedding Error: {e}")
            return []

    async def search(self, query: str, n_results: int = 5, multiplier: float = 1.0) -> List[Dict[str, Any]]:
        """
        ADAPTIVE RAG: Phase 1 — Single Semantic Search.
        Unblocked using asyncio.to_thread for ChromaDB's sync query method.
        """
        if not self.is_connected:
            return []
            
        logger.info(f"🔍 [Nutrition Tool] Searching DB for: '{query}'")
        try:
            query_vector = await self.get_embedding(query)
            if not query_vector:
                return []
                
            # Use the Manager's single-threaded query runner
            results = await db_manager.run_query(
                collection_name="food_text",
                query_embeddings=[query_vector],
                n_results=n_results
            )
            return self._process_results(results, multiplier)
        except Exception as e:
            logger.error(f"❌ [Nutrition Tool] Search Error: {e}")
            return []

    async def multi_query_search(self, query: str, sub_queries: List[str], multiplier: float = 1.0) -> List[Dict[str, Any]]:
        """
        ADAPTIVE RAG: Phase 2 — Multi-Query Expansion.
        """
        if not self.is_connected:
            return []
            
        logger.info(f"🔍 [Nutrition Tool] Multi-Query Search with {len(sub_queries)} variations...")
        all_results = {}

        try:
            tasks_embeddings = [self.get_embedding(sq) for sq in sub_queries]
            vectors = await asyncio.gather(*tasks_embeddings)

            # Parallelize ChromaDB queries
            db_tasks = []
            for v in vectors:
                if v:
                    db_tasks.append(db_manager.run_query(collection_name="food_text", query_embeddings=[v], n_results=3))
            
            if not db_tasks:
                return []
                
            db_results_list = await asyncio.gather(*db_tasks)

            for results in db_results_list:
                for item in self._process_results(results, multiplier):
                    all_results[item['id']] = item

            logger.info(f"  → Merged {len(all_results)} unique results from parallel multi-query")
            return list(all_results.values())
        except Exception as e:
            logger.error(f"❌ [Nutrition Tool] Multi-Query Search Error: {e}")
            return []

    def _process_results(self, results, multiplier: float = 1.0) -> List[Dict[str, Any]]:
        """Process, filter ChromaDB results, and apply math multipliers safely."""
        cleaned = []
        if not results or 'ids' not in results or not results['ids']:
            return cleaned
            
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            food_name = meta.get('food_name', '')

            if not self._is_safe(food_name):
                continue

            def safe_calc(val):
                try:
                    return round(float(val) * multiplier, 2)
                except (ValueError, TypeError):
                    return "N/A"

            cleaned.append({
                "id": results['ids'][0][i],
                "food_name": meta.get('food_name'),
                "calories": safe_calc(meta.get('calories_kcal')),
                "protein": safe_calc(meta.get('protein_g')),
                "fat": safe_calc(meta.get('fat_g')),
                "carbs": safe_calc(meta.get('carbohydrates_g')),
                "score": round(1 - dist, 3)
            })
        return cleaned

    def _is_safe(self, text: str) -> bool:
        """
        Regex-based safety check to prevent false positives.
        Ensures 'Beefsteak tomato' is allowed while 'Beef' is blocked.
        """
        import re
        text_clean = text.lower()
        for food in settings.RESTRICTED_FOODS:
            if re.search(rf"\b{re.escape(food)}\b", text_clean):
                logger.warning(f"🛡️ [Nutrition Tool] Restricted food detected: '{food}' in '{text}'")
                return False
        return True
