from typing import Dict, Any, List
from app.core.state import AgentState
from app.tools.nutrition_tools import NutritionRAGTool
from app.tools.web_search_tool import WebSearchTool
from app.agents.base import BaseRAGAgent
from pydantic import BaseModel, Field
from app.utils.logger import logger

class MealPlanItem(BaseModel):
    type: str = Field(description="Meal type (e.g., Breakfast, Lunch, Snack).")
    name: str = Field(description="Name of the dish/food.")
    portion: str = Field(description="Amount of food to eat (e.g. '300g' or '2 cups'). MUST scale to hit the daily calorie goal!")
    calories: float = Field(description="Total calories for this specific portion (can be decimal).")
    protein: str = Field(description="Total protein for this portion in grams (e.g., '12g').")
    carbs: str = Field(description="Total carbs for this portion in grams (e.g., '18g').")
    fat: str = Field(description="Total fat for this portion in grams (e.g., '6g').")
    benefit: str = Field(description="Why this meal helps the user's goal.")

class DailyTotals(BaseModel):
    calories: float = Field(description="Total daily calories.")
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
        description="REQUIRED. A warm, motivating paragraph (3-4 sentences) explaining the diet strategy and how it helps the user's goal. DO NOT include any specific numbers (calories, protein grams, carbs, fat) here — those belong ONLY in the structured meals and daily_totals fields. Write narrative text only."
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
- DYNAMIC KNOWLEDGE FALLBACK: If the DB is missing calories/macros (shows as Unknown), you MUST generate a realistic numerical estimate (e.g., "45g") using your expert knowledge. It is strictly FORBIDDEN to output "N/A" or "null".
- PORTION SIZING & MACRO MATH: The database provides values per 100g. Scale portions appropriately so that the `daily_totals` actually sum up to the target calories required for their goal!
- STRUCTURED JSON FIELDS: You MUST populate the `summary`, `meals`, `daily_totals`, and `tip` fields with structured data for interactive UI display.
- CLEAN TEXT RESPONSE: The `final_answer` string MUST be a warm, motivating paragraph (3-4 sentences) explaining how this meal plan strategically helps the user's goal. However, DO NOT list the individual meals, bullet points, or raw macros inside `final_answer`.
- CRITICAL: If the user is referring to an uploaded image (e.g. "what is this?", "these calories"), DO NOT guess the food. The Vision Agent will handle it. ONLY provide nutrition info for foods the user EXPLICITLY names in their text. If they didn't name a food, just give general advice and do not mention any specific food from the database.

DATA SANITY CHECK (MANDATORY — apply to EVERY retrieved food before using it):
- SANITY RULE 1 (CALORIE DENSITY): If any food item shows more than 500 kcal per 100g, that data is WRONG. Ignore DB value and use your expert knowledge instead.
- SANITY RULE 2 (IMPOSSIBLE FAT): Calculate fat_calories = fat_g × 9. If fat_calories > total_calories_kcal, the fat value is physically impossible. Ignore DB fat and estimate from your knowledge.
- SANITY RULE 3 (FAT QUALITY — DYNAMIC): For each food, calculate: fat_calories = fat_g × 9, and protein_carb_calories = (protein_g × 4) + (carbs_g × 4). Then judge based on the user's goal:
  • Weight loss: protein_carb_calories MUST be greater than fat_calories. If fat dominates, REJECT the food and use a healthier alternative from your knowledge.
  • Weight gain/maintenance: moderate fat is acceptable, but fat_calories should not exceed total_calories × 0.45.
  This dynamically filters out deep-fried and excessively oily foods based on the actual goal.
- SANITY RULE 4 (CEILING): No single meal may exceed its allocated % of the daily target. Example: if daily target = 1480 kcal and lunch budget = 35%, then max lunch = 1480 × 0.35 = 518 kcal. If a food at its normal portion exceeds this, REDUCE the portion size proportionally.
- SANITY RULE 5 (SUM VERIFICATION): After generating all meals, SUM their calories. If sum < daily target, SCALE UP portions of healthy foods already chosen — do NOT switch to unhealthy alternatives just to add calories.
- SANITY RULE 6 (NO DUPLICATES & EXPERT FALLBACK): NEVER repeat the same food item in more than one meal. You MUST create 4 distinct meals (Breakfast, Lunch, Snack, Dinner). If the database returns limited items, DO NOT repeat them. Instead, use your EXPERT KNOWLEDGE to generate healthy, goal-aligned vegetarian meals to complete the 4-meal structure.

GOAL-SPECIFIC DIETARY RULES (MANDATORY):

🔴 WEIGHT LOSS (when user mentions: lose weight, fat loss, slim down, lose Xkg):
- Daily calories: Create a calorie deficit. If user gives a specific target (e.g. "lose 5kg in 4 months"), calculate: daily_deficit = (kg × 7700) / days, then target = estimated_TDEE - daily_deficit. Minimum floor: 1200 kcal/day.
- Protein: Use 1.2–1.5g per kg of estimated body weight to preserve muscle.
- Per-meal budget: Breakfast 25%, Lunch 35%, Snack 15%, Dinner 25% of daily target.
- Avoid: deep-fried foods, heavy sweets, refined snacks.
- Prefer: high-fiber, high-protein whole foods (oats, sprouts, paneer, dal, curd, salads, fruits, vegetables).

🟢 WEIGHT GAIN / MUSCLE GAIN (when user mentions: gain weight, muscle gain, bulking):
- Daily calories: Calorie surplus. Calculate: daily_surplus = (kg × 7700) / days, target = TDEE + surplus.
- Protein: Use 1.6–2.2g per kg of estimated body weight.
- Per-meal budget: Breakfast 25%, Lunch 35%, Snack 15%, Dinner 25% of daily target.
- Prefer: calorie-dense nutritious foods (paneer, rajma, chana, rice, roti, banana, milk, nuts, dal).

⚖️ GENERAL FITNESS / MAINTENANCE:
- Daily calories: Estimated TDEE (no surplus, no deficit).
- Protein: 1.0–1.2g per kg of estimated body weight.
- Per-meal budget: Breakfast 25%, Lunch 35%, Snack 15%, Dinner 25% of daily target.
- Focus on balanced whole foods and variety across food groups.

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
        result = await self.run_logic(state, specialist_key="nutrition", topic="nutrition")
        return result

    def _validate_output(self, output: Dict[str, Any], context: str) -> Dict[str, Any]:
        """
        Code-level post-processor — LLM cannot override this.
        1. Remove duplicate meals (same food in multiple slots)
        2. Recalculate daily_totals from actual meal data (always accurate)
        """
        meals = output.get("meals", [])
        if not meals:
            return output

        # --- Step 1: Remove duplicate food items ---
        seen = set()
        unique_meals = []
        for meal in meals:
            key = meal.get("name", "").lower().strip()
            if key not in seen:
                seen.add(key)
                unique_meals.append(meal)
            else:
                logger.warning(f"❌ [Nutrition Validator] Removed duplicate meal: '{meal.get('name')}'")
        output["meals"] = unique_meals

        # --- Step 2: Recalculate daily_totals from actual meals ---
        def parse_num(val) -> float:
            try:
                return float(str(val).replace("g", "").replace(",", "").strip())
            except (ValueError, TypeError):
                return 0.0

        total_cal   = sum(parse_num(m.get("calories", 0)) for m in unique_meals)
        total_prot  = sum(parse_num(m.get("protein",  0)) for m in unique_meals)
        total_carbs = sum(parse_num(m.get("carbs",    0)) for m in unique_meals)
        total_fat   = sum(parse_num(m.get("fat",      0)) for m in unique_meals)

        output["daily_totals"] = {
            "calories": round(total_cal, 1),
            "protein":  f"{round(total_prot,  1)}g",
            "carbs":    f"{round(total_carbs, 1)}g",
            "fat":      f"{round(total_fat,   1)}g"
        }
        logger.info(f"✅ [Nutrition Validator] Recalculated totals: {total_cal:.0f} kcal, {total_prot:.0f}g protein")
        return output

    def _format_context(self, results: List[Dict]) -> str:
        """Convert DB results into meaningful nutritional context (Standardized to 100g base).
        Filters out foods with obviously bad data before LLM ever sees them.
        """
        if not results:
            return ""
        lines = []
        for r in results:
            name = r.get('food_name', 'Unknown')
            try:
                cal  = float(r.get('calories', 0) or 0)
                prot = float(r.get('protein',  0) or 0)
                fat  = float(r.get('fat',      0) or 0)

                # --- Code-level filter: reject impossible calorie density ---
                if cal > 500:
                    logger.warning(f"⚠️ [Nutrition DB Filter] Skipping '{name}': {cal} kcal/100g exceeds limit.")
                    continue

                # Mathematically calculate missing carbs using 4-4-9 macro rule
                raw_carbs = r.get('carbs', 'Unknown')
                if raw_carbs in ['N/A', 'Unknown', None, '', 0, '0']:
                    carb_cals = cal - (prot * 4) - (fat * 9)
                    carbs = max(0, round(carb_cals / 4, 1))
                else:
                    carbs = float(raw_carbs)

                lines.append(f"• {name} (base 100g): {cal} kcal, {prot}g protein, {fat}g fat, {carbs}g carbs")
            except Exception:
                lines.append(f"• {name} (base 100g): {r.get('calories')} kcal")
        return "\n".join(lines)
