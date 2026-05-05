from typing import Dict, Any, List
from app.core.state import AgentState
from app.tools.nutrition_tools import NutritionRAGTool
from app.tools.web_search_tool import WebSearchTool
from app.agents.base import BaseRAGAgent
from pydantic import BaseModel, Field


class NutritionAnalysis(BaseModel):
    is_accurate: bool = Field(description="True if retrieved data is relevant to the query.")
    needs_web_search: bool = Field(description="True if local DB lacks info and web search is needed.")
    quantity_multiplier: float = Field(default=1.0, description="Multiplier if user asks for specific amount (e.g., 200g = 2.0). Default is 1.0 (100g).")
    sub_queries: List[str] = Field(default=[], description="3 alternative search phrases for expansion.")
    final_answer: str = Field(
        description="""REQUIRED. A dynamic, conversational coaching response. RULES:
        - ALWAYS provide a helpful answer, even if data is limited — use expert knowledge.
        - NEVER mention database, scores, IDs, or raw numbers without context.
        - SCALE the nutritional values based on the 'quantity_multiplier'.
        - ALWAYS explain WHY a food is good/bad for the user's specific goal and health.
        - Use a warm, expert tone like a real fitness coach.
        - Include practical tips (e.g. 'best eaten post-workout', 'pair with X')."""
    )


class NutritionAgent(BaseRAGAgent):
    """
    Step 7.2: NUTRITION AGENT (Adaptive RAG)
    Specialist in diet, calories, food analysis, and meal planning.
    """

    def __init__(self):
        system_prompt = """You are the expert Nutrition Coach for 'Agentic AI Gym'.
You have access to a nutritional database and your own expert knowledge.

YOUR ROLE:
- Answer any nutrition, diet, food, or calorie-related question conversationally.
- NOTE: Retrieved database values are typically per 100g.
- If the user specifies an amount (e.g., '200g', 'half a kilo'), set the 'quantity_multiplier' accordingly.
- Use retrieved data as EVIDENCE, but NEVER dump it raw to the user.
- Translate numbers into meaningful insights (e.g. "165 kcal per 100g means it's very lean").
- Always connect your advice to the user's goal: {goal}
- Consider the user's medical background/injuries: {injuries}

STRICT POLICIES:
- NEVER recommend BEEF in any form.
- NEVER expose internal field names like 'score', 'id', 'food_name'.
- ALWAYS give actionable, personalized advice.
- ALWAYS fill in the 'final_answer' field, no matter what.

USER DATA:
Goal: {goal}
Medical/Injuries: {injuries}"""
        
        super().__init__(
            agent_name="Nutrition Agent",
            rag_tool=NutritionRAGTool(),
            web_search_tool=WebSearchTool(),
            output_schema=NutritionAnalysis,
            system_prompt=system_prompt
        )

    async def run(self, state: AgentState) -> Dict[str, Any]:
        return await self.run_logic(state, specialist_key="nutrition", topic="nutrition")

    def _format_context(self, results: List[Dict]) -> str:
        """Convert DB results into meaningful nutritional context (Standardized to 100g base)."""
        if not results:
            return ""
        lines = []
        for r in results:
            name = r.get('food_name', 'Unknown')
            cal = r.get('calories', 'N/A')
            prot = r.get('protein', 'N/A')
            fat = r.get('fat', 'N/A')
            carbs = r.get('carbs', 'N/A')
            lines.append(f"• {name} (per 100g): {cal} kcal, {prot}g protein, {fat}g fat, {carbs}g carbs")
        return "\n".join(lines)
