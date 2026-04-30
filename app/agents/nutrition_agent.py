from typing import Dict, Any, List
from app.core.state import AgentState
from app.tools.nutrition_tools import NutritionRAGTool
from app.tools.web_search_tool import WebSearchTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class NutritionAnalysis(BaseModel):
    is_accurate: bool = Field(
        description="True if retrieved data is relevant to the query."
    )
    needs_web_search: bool = Field(
        description="True if local DB lacks sufficient info and web search is needed."
    )
    sub_queries: List[str] = Field(
        default=[],
        description="3 alternative search phrases for adaptive multi-query RAG if is_accurate is False."
    )
    final_answer: str = Field(
        description="""REQUIRED. A dynamic, conversational coaching response. RULES:
        - ALWAYS provide a helpful answer, even if data is limited — use expert knowledge.
        - NEVER mention database, scores, IDs, or raw numbers without context.
        - ALWAYS explain WHY a food is good/bad for the user's specific goal.
        - If multiple items found, compare them and recommend the best one.
        - Use a warm, expert tone like a real fitness coach.
        - Include practical tips (e.g. 'best eaten post-workout', 'pair with X').
        - This field MUST always be filled. Never leave it empty."""
    )


class NutritionAgent:
    """
    Step 7.2: NUTRITION AGENT (Adaptive RAG)
    Specialist in diet, calories, food analysis, and meal planning.
    All responses are dynamic and conversational — no raw DB output ever reaches the user.
    """

    def __init__(self):
        self.rag_tool = NutritionRAGTool()
        self.web_search = WebSearchTool()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.3  # Slight warmth for coaching tone
        ).with_structured_output(NutritionAnalysis)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the expert Nutrition Coach for 'Agentic AI Gym'.
You have access to a nutritional database and your own expert knowledge.

YOUR ROLE:
- Answer any nutrition, diet, food, or calorie-related question conversationally.
- Use retrieved data as EVIDENCE, but NEVER dump it raw to the user.
- Translate numbers into meaningful insights (e.g. "165 kcal per 100g means it's very lean").
- Always connect your advice to the user's goal: {goal}
- If the user asks for recommendations, rank and explain your top choices.
- CRITICAL: Always provide a 'final_answer'. Even if data is missing, give expert advice.

STRICT POLICIES:
- NEVER recommend BEEF in any form.
- NEVER expose internal field names like 'score', 'id', 'food_name'.
- ALWAYS give actionable, personalized advice.
- ALWAYS fill in the 'final_answer' field, no matter what.

USER GOAL: {goal}"""),
            ("human", "QUESTION: {query}\n\nNUTRITIONAL CONTEXT:\n{context}")
        ])

    async def run(self, state: AgentState) -> Dict[str, Any]:
        query = state['messages'][-1].content
        user_context = state.get('user_context', {})
        goal = user_context.get("goal", "General Fitness") if isinstance(user_context, dict) else "General Fitness"

        chain = self.prompt | self.llm

        # ── PHASE 1: Local ChromaDB Search ───────────────────
        db_results = await self.rag_tool.search(query)
        context_str = self._format_context(db_results)

        analysis: NutritionAnalysis = await chain.ainvoke({
            "query": query,
            "context": context_str or "No specific data retrieved from local database.",
            "goal": goal
        })

        print(f"🍎 [Nutrition Agent] Accurate: {analysis.is_accurate} | Web needed: {analysis.needs_web_search}")

        # ── PHASE 2: Adaptive Multi-Query Expansion ───────────
        if not analysis.is_accurate and analysis.sub_queries:
            print(f"  🔄 Expanding search with {len(analysis.sub_queries)} sub-queries...")
            db_results = await self.rag_tool.multi_query_search(query, analysis.sub_queries)
            context_str = self._format_context(db_results)

            analysis = await chain.ainvoke({
                "query": query,
                "context": context_str or "Expanded search returned no additional data.",
                "goal": goal
            })
            print(f"  ✅ Expanded re-analysis. Accurate: {analysis.is_accurate}")

        # ── PHASE 3: Adaptive Web Fallback (final Adaptive RAG step) ──
        # NOTE: This is NOT CRAG. CRAG is used in the Training Agent.
        # Here, web search is simply the last adaptation step when the
        # local database (even with multi-query) cannot fulfill the request.
        if not analysis.is_accurate and analysis.needs_web_search:
            if self.web_search.is_available:
                web_data = await self.web_search.search(query, topic="nutrition")
                web_context = web_data.get("summary", "") or \
                              "\n".join([r['content'] for r in web_data.get("results", [])[:3]])

                analysis = await chain.ainvoke({
                    "query": query,
                    "context": f"[Live Web Data]:\n{web_context}",
                    "goal": goal
                })
                print(f"  ✅ Web synthesis complete. Accurate: {analysis.is_accurate}")
            else:
                # Even without web search, LLM expert knowledge kicks in
                analysis = await chain.ainvoke({
                    "query": query,
                    "context": "No database or web data available. Use your expert nutrition knowledge.",
                    "goal": goal
                })
                print("  💡 Using expert knowledge fallback.")

        return {
            "specialist_results": {
                "nutrition": {
                    "answer": analysis.final_answer,
                    "status": "success" if analysis.is_accurate else "expert_knowledge"
                }
            },
            "next_node": "synthesis_layer"
        }

    def _format_context(self, results: List[Dict]) -> str:
        """Convert DB results into meaningful nutritional context for the LLM."""
        if not results:
            return ""
        lines = []
        for r in results:
            name = r.get('food_name', 'Unknown')
            cal = r.get('calories', 'N/A')
            prot = r.get('protein', 'N/A')
            fat = r.get('fat', 'N/A')
            carbs = r.get('carbs', 'N/A')
            lines.append(
                f"• {name}: {cal} kcal, {prot}g protein, {fat}g fat, {carbs}g carbs"
            )
        return "\n".join(lines)
