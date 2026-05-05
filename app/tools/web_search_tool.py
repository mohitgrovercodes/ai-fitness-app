import asyncio
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.utils.logger import logger


class WebSearchTool:
    """
    Production Web Search Tool using Tavily API.
    Used as a fallback when ChromaDB RAG returns low-confidence results.
    
    Flow: ChromaDB RAG → (if low confidence) → WebSearchTool → Synthesize
    """

    def __init__(self):
        self._client = None
        self._available = False

        if settings.TAVILY_API_KEY:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=settings.TAVILY_API_KEY)
                self._available = True
                logger.info("✅ [WebSearchTool] Tavily client initialized.")
            except Exception as e:
                logger.error(f"⚠️ [WebSearchTool] Tavily init failed: {e}")
        else:
            logger.warning("⚠️ [WebSearchTool] No TAVILY_API_KEY found. Web search disabled.")

    @property
    def is_available(self) -> bool:
        return self._available

    async def search(
        self,
        query: str,
        topic: str = "nutrition",
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        Performs a fitness-scoped web search and returns structured results.
        Unblocked using asyncio.to_thread for the synchronous Tavily client.
        """
        if not self._available:
            return {
                "results": [],
                "summary": "Web search is unavailable. TAVILY_API_KEY not configured.",
                "source": "unavailable"
            }

        # Scope the query to the fitness/nutrition domain
        scoped_query = f"{query} {topic} nutritional information health"
        logger.info(f"🌐 [WebSearchTool] Searching web for: '{scoped_query}'")

        try:
            # Unblock the event loop for the synchronous Tavily API call
            response = await asyncio.to_thread(
                self._client.search,
                query=scoped_query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True
            )

            # Extract structured results
            results = []
            for r in response.get("results", []):
                content = r.get("content", "")
                # Apply policy filter: skip restricted content
                if any(food in content.lower() for food in settings.RESTRICTED_FOODS):
                    logger.warning(f"  ⚠️ Policy: Skipping web result containing restricted food")
                    continue
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": content[:500],
                    "score": r.get("score", 0)
                })

            direct_answer = response.get("answer", "")
            logger.info(f"  → Found {len(results)} web results. Direct answer: {'Yes' if direct_answer else 'No'}")

            return {
                "results": results,
                "summary": direct_answer,
                "source": "tavily_web_search"
            }

        except Exception as e:
            logger.error(f"  ❌ [WebSearchTool] Search failed: {e}")
            return {
                "results": [],
                "summary": f"Web search encountered an error: {str(e)}",
                "source": "error"
            }
