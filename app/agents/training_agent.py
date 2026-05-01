from typing import Dict, Any, List
from app.core.state import AgentState
from app.tools.training_tools import TrainingRAGTool
from app.tools.web_search_tool import WebSearchTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class TrainingAnalysis(BaseModel):
    is_accurate: bool = Field(description="Are the retrieved exercises relevant and safe for the user query?")
    needs_web_search: bool = Field(description="True if the exercise is unknown or local DB lacks sufficient info.")
    sub_queries: List[str] = Field(default=[], description="If complex routine, list 3 alternative search terms")
    final_answer: str = Field(description="The professional coaching response detailing the workout/exercise")

class TrainingAgent:
    """
    Step 7.3: TRAINING AGENT (CRAG)
    Specialist in workout routines, form correction, and exercise instructions.
    Uses Corrective RAG (CRAG) to fall back to the web if local data is insufficient.
    """
    def __init__(self):
        self.rag_tool = TrainingRAGTool()
        self.web_search = WebSearchTool()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0.3
        ).with_structured_output(TrainingAnalysis)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the expert Training Coach for 'Agentic AI Gym'.
Your goal is to provide accurate, safe workout advice based on retrieved data.

STRICT POLICIES:
1. SAFETY FIRST: Always mention proper form or warm-ups if appropriate.
2. ACCURACY (CRAG Evaluator): If the retrieved exercises DO NOT match the user's specific request, set 'is_accurate' to false.
3. ADAPTABILITY: If the user has injuries, adapt the advice or suggest alternatives from the data.

USER DATA:
Goal: {goal}
Injuries/Medical: {injuries}"""),
            ("human", "USER QUERY: {query}\n\nRETRIEVED EXERCISE DATA:\n{context}")
        ])

    async def run(self, state: AgentState) -> Dict[str, Any]:
        query = state['messages'][-1].content
        user_context = state.get('user_context', {})
        
        goal = user_context.get("goal", "General Fitness") if isinstance(user_context, dict) else "General Fitness"
        injuries = ", ".join(user_context.get("injuries", [])) if isinstance(user_context, dict) and user_context.get("injuries") else "None"

        # ── PHASE 1: Local ChromaDB Search ───────────────────
        db_results = await self.rag_tool.search(query)
        context_str = self._format_context(db_results)

        chain = self.prompt | self.llm
        analysis: TrainingAnalysis = await chain.ainvoke({
            "query": query,
            "context": context_str or "No specific data retrieved from local database.",
            "goal": goal,
            "injuries": injuries
        })

        print(f"🏋️ [Training Agent] Accurate: {analysis.is_accurate} | Web needed: {analysis.needs_web_search}")

        # ── PHASE 2: CRAG Correction / Multi-Query Expansion ──
        if not analysis.is_accurate and analysis.sub_queries:
            print(f"  🔄 CRAG: Expanding search with {len(analysis.sub_queries)} sub-queries...")
            db_results = await self.rag_tool.multi_query_search(query, analysis.sub_queries)
            context_str = self._format_context(db_results)

            analysis = await chain.ainvoke({
                "query": query,
                "context": context_str or "Expanded search returned no additional data.",
                "goal": goal,
                "injuries": injuries
            })
            print(f"  ✅ CRAG: Re-evaluation complete. Accurate: {analysis.is_accurate}")

        # ── PHASE 3: CRAG Web Fallback ─────────────────────────
        if not analysis.is_accurate and analysis.needs_web_search:
            print("  🌐 CRAG: Local data insufficient. Triggering Web Search...")
            if self.web_search.is_available:
                web_data = await self.web_search.search(query, topic="fitness workout exercise")
                web_context = web_data.get("summary", "") or \
                              "\n".join([r['content'] for r in web_data.get("results", [])[:3]])

                analysis = await chain.ainvoke({
                    "query": query,
                    "context": f"[Live Web Data]:\n{web_context}",
                    "goal": goal,
                    "injuries": injuries
                })
                print(f"  ✅ Web synthesis complete. Accurate: {analysis.is_accurate}")
            else:
                print("  ⚠️ Web search unavailable. Falling back to expert knowledge.")
                analysis = await chain.ainvoke({
                    "query": query,
                    "context": "No database or web data available. Use your expert fitness knowledge safely.",
                    "goal": goal,
                    "injuries": injuries
                })

        return {
            "specialist_results": {
                "training": {
                    "data": db_results,
                    "answer": analysis.final_answer,
                    "status": "success" if analysis.is_accurate else "expert_knowledge"
                }
            }
        }

    def _format_context(self, results: List[Dict]) -> str:
        """Convert DB results into meaningful exercise context for the LLM."""
        if not results:
            return ""
        lines = []
        for r in results:
            name = r.get('name', 'Unknown')
            muscle = r.get('main_muscle', 'N/A')
            equip = r.get('equipment', 'N/A')
            prep = r.get('preparation', '')
            exe = r.get('execution', '')
            lines.append(
                f"• {name} (Muscle: {muscle}, Equipment: {equip})\n  Prep: {prep}\n  Execution: {exe}"
            )
        return "\n\n".join(lines)
