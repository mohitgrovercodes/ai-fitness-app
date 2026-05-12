from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.core.state import AgentState
from app.utils.logger import logger

from app.core.config import settings

class BaseRAGAgent:
    """
    Base class for Specialist Agents using Adaptive RAG / CRAG.
    Deduplicates the Phase 1 (DB), Phase 2 (Multi-Query), and Phase 3 (Web) logic.
    """
    def __init__(
        self, 
        agent_name: str,
        rag_tool: Any, 
        web_search_tool: Any, 
        output_schema: Type[BaseModel],
        system_prompt: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3
    ):
        self.agent_name = agent_name
        self.rag_tool = rag_tool
        self.web_search = web_search_tool
        self.llm = ChatOpenAI(
            model=model_name, 
            temperature=temperature, 
            api_key=settings.OPENAI_API_KEY,
            max_retries=3 # Production-grade retry logic
        ).with_structured_output(output_schema, method="function_calling")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "CONVERSATION SUMMARY (for context):\n{summary}\n\nQUESTION: {query}\n\nRETRIEVED DATA:\n{context}")
        ])

    async def run_logic(self, state: AgentState, specialist_key: str, topic: str = "general") -> Dict[str, Any]:
        """Core Adaptive RAG logic shared across specialists."""
        query = state['messages'][-1].content
        user_context = state.get('user_context', {})
        summary = state.get('conversation_summary', "No previous context.")
        
        # Extract common variables for the prompt
        goal = user_context.get("goal", "General Fitness") if isinstance(user_context, dict) else "General Fitness"
        injuries = ", ".join(user_context.get("injuries", [])) if isinstance(user_context, dict) and user_context.get("injuries") else "None"
        diet_pref = user_context.get("diet_preference", "None") if isinstance(user_context, dict) else "None"

        logger.info(f"🧬 [{self.agent_name}] Extracted Context: Goal='{goal}', Diet='{diet_pref}'")

        chain = self.prompt | self.llm

        # ── PHASE 1: Local ChromaDB Search ───────────────────
        db_results = await self.rag_tool.search(query)
        context_str = self._format_context(db_results)

        analysis = await chain.ainvoke({
            "query": query,
            "context": context_str or "No specific data retrieved from local database.",
            "goal": goal,
            "injuries": injuries,
            "diet_preference": diet_pref,
            "summary": summary
        })

        logger.info(f"[{self.agent_name}] Accurate: {analysis.is_accurate} | Web needed: {analysis.needs_web_search}")

        # ── PHASE 2: Multi-Query Expansion ───────────
        if not analysis.is_accurate and hasattr(analysis, 'sub_queries') and analysis.sub_queries:
            logger.info(f"  🔄 [{self.agent_name}] Expanding search with sub-queries...")
            
            # Use quantity multiplier if the analysis found one (e.g. for nutrition)
            multiplier = getattr(analysis, 'quantity_multiplier', 1.0)
            db_results = await self.rag_tool.multi_query_search(query, analysis.sub_queries, multiplier=multiplier)
            context_str = self._format_context(db_results)

            analysis = await chain.ainvoke({
                "query": query,
                "context": context_str or "Expanded search returned no additional data.",
                "goal": goal,
                "injuries": injuries,
                "diet_preference": diet_pref,
                "summary": summary
            })

        # ── PHASE 3: Web Fallback ─────────────────────────
        if not analysis.is_accurate and analysis.needs_web_search:
            if self.web_search.is_available:
                logger.info(f"  🌐 [{self.agent_name}] Triggering Web Search fallback...")
                web_data = await self.web_search.search(query, topic=topic)
                web_context = web_data.get("summary", "") or \
                              "\n".join([r['content'] for r in web_data.get("results", [])[:3]])

                analysis = await chain.ainvoke({
                    "query": query,
                    "context": f"{context_str}\n\n[Live Web Data]:\n{web_context}",
                    "goal": goal,
                    "injuries": injuries,
                    "diet_preference": diet_pref,
                    "summary": summary
                })
            else:
                logger.warning(f"  ⚠️ [{self.agent_name}] Web search unavailable. Falling back to expert knowledge.")
                analysis = await chain.ainvoke({
                    "query": query,
                    "context": f"{context_str}\n\n[Expert Knowledge Fallback]: Use expert knowledge to supplement the data.",
                    "goal": goal,
                    "injuries": injuries,
                    "diet_preference": diet_pref,
                    "summary": summary
                })

        # Prepare final specialist output
        specialist_output = {
            "answer": analysis.final_answer,
            "status": "success" if analysis.is_accurate else "expert_knowledge"
        }
        
        # Add optional fields if present in the analysis object
        # This dynamically captures metadata like 'exercise_gifs', 'nutritional_info', etc.
        logger.info(f"[{self.agent_name}] Raw Analysis: {analysis}")
        standard_fields = {"is_accurate", "needs_web_search", "sub_queries", "final_answer", "quantity_multiplier"}
        analysis_dict = analysis.model_dump() if hasattr(analysis, "model_dump") else analysis.__dict__
        # Media fields that should always be included (even if empty dict)
        always_include = {"exercise_gifs", "exercise_images"}
        
        for field, val in analysis_dict.items():
            if field not in standard_fields:
                # Always include media fields; only skip other falsy values
                if val or field in always_include:
                    specialist_output[field] = val

        # Run optional validation hook for subclasses
        specialist_output = self._validate_output(specialist_output, context_str)

        return {
            "specialist_results": {
                specialist_key: specialist_output
            }
        }

    def _validate_output(self, output: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Optional hook for subclasses to validate LLM output (e.g. check media hallucinations)."""
        return output

    def _format_context(self, results: List[Dict]) -> str:
        """To be implemented by subclasses if they need custom formatting."""
        raise NotImplementedError("Subclasses must implement _format_context")
