from typing import Dict, Any, List
from app.core.state import AgentState
from app.tools.nutrition_tools import NutritionRAGTool
from app.tools.web_search_tool import WebSearchTool
from app.agents.base import BaseRAGAgent
from pydantic import BaseModel, Field


class MealPlanItem(BaseModel):
    type: str = Field(description="Meal type (e.g., Breakfast, Lunch, Snack).")
    name: str = Field(description="Name of the dish/food.")
    calories: int = Field(description="Estimated calories.")
    protein: str = Field(description="Protein amount in grams (e.g., '12g'). NEVER 'N/A'.")
    carbs: str = Field(description="Carbs amount in grams (e.g., '18g'). NEVER 'N/A'. MUST estimate if unknown.")
    fat: str = Field(description="Fat amount in grams (e.g., '6g'). NEVER 'N/A'. MUST estimate if unknown.")
    benefit: str = Field(description="Why this meal helps the user's goal.")

class DailyTotals(BaseModel):
    calories: int = Field(description="Total daily calories.")
    protein: str = Field(description="Total daily protein.")
    carbs: str = Field(description="Total daily carbs.")
    fat: str = Field(description="Total daily fat.")

class NutritionAnalysis(BaseModel):
    is_accurate: bool = Field(description="True if retrieved data is relevant to the query.")
    needs_web_search: bool = Field(description="True if local DB lacks info and web search is needed.")
    quantity_multiplier: float = Field(default=1.0, description="Multiplier if user asks for specific amount (e.g., 200g = 2.0). Default is 1.0 (100g).")
    sub_queries: List[str] = Field(default=[], description="3 alternative search phrases for expansion.")
    summary: str = Field(default="", description="Brief introduction/summary of the meal plan.")
    meals: List[MealPlanItem] = Field(default=[], description="List of structured meals.")
    daily_totals: DailyTotals = Field(default=None, description="Total macros and calories for the day.")
    tip: str = Field(default="", description="Closing tip for hydration or nutrition.")
    final_answer: str = Field(
        description="REQUIRED. A warm, motivating paragraph (3-4 sentences) explaining the diet strategy. DO NOT list meals or macros here."
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
- CROSS-AGENT INTELLIGENCE: If the user is asking for a workout (Workout Intent), you must provide a complementary nutrition recommendation (e.g., a post-workout high-protein meal) even if they didn't explicitly ask for food.
- Use retrieved data as EVIDENCE, but NEVER dump it raw to the user.
- Always connect your advice to the user's goal: {goal}
- Consider the user's medical background/injuries: {injuries}

STRICT POLICIES:
- DIETARY RESTRICTIONS: If the user asks for "pure veg" or "vegetarian", you MUST ONLY provide 100% vegetarian foods. Avoid suggesting foods with names that sound like meat (e.g., "Kebab") unless you explicitly clarify it is made of vegetables or soy. NEVER recommend beef.
- DYNAMIC KNOWLEDGE FALLBACK: If the DB is missing calories/macros, you MUST generate a realistic numerical estimate (e.g., "45g") using your expert knowledge. It is strictly FORBIDDEN to output "N/A" or "null".
- STRUCTURED JSON FIELDS: You MUST populate the `summary`, `meals`, `daily_totals`, and `tip` fields with structured data for interactive UI display.
- CLEAN TEXT RESPONSE: The `final_answer` string MUST be a warm, motivating paragraph (3-4 sentences) explaining how this meal plan strategically helps the user's goal. However, DO NOT list the individual meals, bullet points, or raw macros inside `final_answer`.
- CRITICAL: If the user is referring to an uploaded image (e.g. "what is this?", "these calories"), DO NOT guess the food. The Vision Agent will handle it. ONLY provide nutrition info for foods the user EXPLICITLY names in their text. If they didn't name a food, just give general advice and do not mention any specific food from the database.

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
