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
        default="",
        description="A warm, motivating paragraph (3-4 sentences) explaining the diet strategy and how it helps the user's goal. DO NOT include any specific numbers (calories, protein grams, carbs, fat) here — those belong ONLY in the structured meals and daily_totals fields. Write narrative text only."
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
- DIETARY FLEXIBILITY (CRITICAL): You MUST strictly respect the user's dietary preferences (e.g., Veg, Non-Veg, Vegan, Keto).
- If the user specifies "Veg", "Vegetarian", or "Pure Veg", you MUST ONLY provide vegetarian foods (no meat, fish, or eggs unless specified).
- If the user specifies "Non-Veg", "Nonveg", "non_vegetarian", "non-vegetarian", or "Meat-eater", you SHOULD include healthy animal proteins (chicken, fish, eggs, lean meats, etc.) in the plan.

- If no preference is specified, provide a balanced diet. 
- There are NO global food restrictions. If the user wants beef, chicken, or pork, you are allowed to recommend it if it fits their nutritional goals.
- DYNAMIC KNOWLEDGE FALLBACK: If the DB is missing calories/macros (shows as Unknown), you MUST generate a realistic numerical estimate (e.g., "45g") using your expert knowledge. It is strictly FORBIDDEN to output "N/A", "null", or "Unknown" for any macro or calorie field. Every meal MUST have numeric values for protein, carbs, fat, and calories.
- PORTION SIZING & MACRO MATH: The database provides values per 100g. Scale portions appropriately so that the `daily_totals` actually sum up to the target calories required for their goal!
- STRUCTURED JSON FIELDS: You MUST populate the `summary`, `meals`, `daily_totals`, and `tip` fields with structured data for interactive UI display.
- CLEAN TEXT RESPONSE: The `final_answer` string MUST be a warm, motivating paragraph (3-4 sentences) explaining how this meal plan strategically helps the user's goal. However, DO NOT list the individual meals, bullet points, or raw macros inside `final_answer`.
- CRITICAL: If the user is referring to an uploaded image (e.g. "what is this?", "these calories"), DO NOT guess the food. The Vision Agent will handle it. ONLY provide nutrition info for foods the user EXPLICITLY names in their text. If they didn't name a food, just give general advice and do not mention any specific food from the database.

DATA SANITY CHECK (MANDATORY — apply to EVERY retrieved food before using it):
- SANITY RULE 1 (CALORIE DENSITY - DYNAMIC): Evaluate calorie density based on the food's macronutrient profile. Do not enforce a static calorie cap (e.g., nuts and oils can safely exceed 600 kcal/100g). Only reject a food if its calories physically exceed the theoretical maximum for its macros (Protein/Carbs = 4 kcal/g, Fat = 9 kcal/g).
- SANITY RULE 2 (IMPOSSIBLE FAT): Calculate fat_calories = fat_g × 9. If fat_calories > total_calories_kcal, the fat value is physically impossible. Ignore DB fat and estimate from your knowledge.
- SANITY RULE 3 (FAT QUALITY — DYNAMIC): Dynamically adjust the acceptable fat ratio based on the user's specific diet preference. 
  • For Keto diets, fat MUST be high (70-80%).
  • For Standard Weight Loss diets, ensure protein_carb_calories > fat_calories.
  • For Standard Weight/Muscle Gain diets, moderate fat is acceptable, but scale it dynamically to fit their goal without forcing an arbitrary ceiling.
- SANITY RULE 4 (CEILING): No single meal may exceed its allocated % of the daily target.
- SANITY RULE 5 (SUM VERIFICATION): After generating all meals, SUM their calories. If sum < daily target, SCALE UP portions of healthy foods already chosen.
- SANITY RULE 6 (NO DUPLICATES & EXPERT FALLBACK): NEVER repeat the same food item in more than one meal. Create 4 distinct meals. If the database returns limited items, use your EXPERT KNOWLEDGE to generate healthy, goal-aligned meals consistent with the user's dietary preference.

GOAL-SPECIFIC DIETARY RULES (MANDATORY):

🔴 WEIGHT LOSS (when user mentions: lose weight, fat loss, slim down, lose Xkg):
- Daily calories: Dynamically calculate a sustainable, healthy calorie deficit based on the user's estimated TDEE (Total Daily Energy Expenditure). You MUST strictly avoid generating unsustainably low calorie counts that would be considered extreme crash diets. If your initial calculated sum falls into a dangerous crash-diet range, dynamically scale up the portion sizes or add healthy snacks to ensure a safe, sustainable deficit.
- Protein: Dynamically calculate optimal protein intake based on the user's estimated body weight to effectively preserve muscle mass during the deficit.
- Prefer: High-protein lean sources tailored dynamically to the user's specific diet type. Use your expert knowledge to select the most satiating, high-quality protein options that strictly align with whatever dietary restrictions or preferences the user provides (e.g., whether they specify vegetarian, vegan, keto, pescatarian, jain, or any other diet).

🟢 WEIGHT GAIN / MUSCLE GAIN (when user mentions: gain weight, muscle gain, bulking):
- Daily calories: Dynamically calculate a healthy calorie surplus based on the user's estimated TDEE to support steady weight gain.
- Protein: Dynamically calculate optimal high-protein intake required to support muscle hypertrophy and recovery.
- Prefer: Dynamically select high-quality, calorie-dense nutritious foods that strictly align with the user's specific dietary preferences.

⚖️ GENERAL FITNESS / MAINTENANCE:
- Daily calories: Estimated TDEE.
- Protein: 1.0–1.2g per kg of estimated body weight.

USER DATA:
Goal: {goal}
Medical/Injuries: {injuries}
Dietary Preference: {diet_preference}
Current Context: {summary}
"""
        
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

    def _validate_output(self, output: Dict[str, Any], context: str, state: Any = None) -> Dict[str, Any]:
        """
        Code-level post-processor — LLM cannot override this.
        1. Remove duplicate meals (same food in multiple slots)
        2. Correct per-meal calories using macro math (protein×4 + carbs×4 + fat×9)
        3. Recalculate daily_totals from actual meal data (always accurate)
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

        # --- Step 2: Correct per-meal calories using 4-4-9 macro rule ---
        def parse_num(val) -> float:
            try:
                return float(str(val).replace("g", "").replace(",", "").strip())
            except (ValueError, TypeError):
                return 0.0

        for meal in unique_meals:
            prot = parse_num(meal.get("protein", 0))
            carbs = parse_num(meal.get("carbs", 0))
            fat = parse_num(meal.get("fat", 0))
            macro_calories = round((prot * 4) + (carbs * 4) + (fat * 9), 1)
            llm_calories = parse_num(meal.get("calories", 0))

            if macro_calories > 0:
                # Macro-based correction (primary check)
                if llm_calories == 0 or abs(llm_calories - macro_calories) / max(macro_calories, 1) > 0.20:
                    logger.warning(
                        f"⚠️ [Macro Validator] '{meal.get('name')}': LLM said {llm_calories} kcal, "
                        f"macros say {macro_calories} kcal → corrected."
                    )
                    meal["calories"] = macro_calories
            else:
                # Fallback: Physics-based density check (max 9 kcal/gram = pure fat)
                # Parse portion grams for density check
                portion_str = str(meal.get("portion", "100g"))
                portion_g = parse_num(portion_str) or 100.0
                max_possible_kcal = portion_g * 9  # Absolute physical maximum
                if llm_calories > max_possible_kcal:
                    logger.warning(
                        f"⚠️ [Density Validator] '{meal.get('name')}': {llm_calories} kcal "
                        f"for {portion_g}g is physically impossible (max={max_possible_kcal}). Capping."
                    )
                    # Cap at a reasonable density (~4 kcal/g = mixed food average)
                    meal["calories"] = round(portion_g * 4.0, 1)

        output["meals"] = unique_meals

        # --- Step 3: Recalculate daily_totals from actual meals ---
        total_cal   = sum(parse_num(m.get("calories", 0)) for m in unique_meals)
        total_prot  = sum(parse_num(m.get("protein",  0)) for m in unique_meals)
        total_carbs = sum(parse_num(m.get("carbs",    0)) for m in unique_meals)
        total_fat   = sum(parse_num(m.get("fat",      0)) for m in unique_meals)

        # --- Step 4: GOAL-AWARE AUTO SCALER (Dynamic TDEE Matcher) ---
        target_calories = None
        if state and "user_context" in state:
            target_calories = state["user_context"].get("target_calories")
        
        if target_calories and total_cal > 0:
            # If the LLM missed the target by more than 5%, scale it
            if abs(total_cal - target_calories) / target_calories > 0.05:
                ratio = target_calories / total_cal
                logger.info(f"⚖️ [Auto-Scaler] Target: {target_calories} | Actual: {total_cal}. Scaling by {ratio:.2f}")
                
                # Apply scaling to all meals
                total_cal, total_prot, total_carbs, total_fat = 0, 0, 0, 0
                for meal in unique_meals:
                    # Scale macros
                    m_prot = parse_num(meal.get("protein", 0)) * ratio
                    m_carbs = parse_num(meal.get("carbs", 0)) * ratio
                    m_fat = parse_num(meal.get("fat", 0)) * ratio
                    
                    meal["protein"] = f"{round(m_prot, 1)}g"
                    meal["carbs"] = f"{round(m_carbs, 1)}g"
                    meal["fat"] = f"{round(m_fat, 1)}g"
                    meal["calories"] = round((m_prot * 4) + (m_carbs * 4) + (m_fat * 9), 1)
                    
                    # Scale portion size if possible
                    portion_g = parse_num(meal.get("portion", 0))
                    if portion_g > 0:
                        meal["portion"] = f"{round(portion_g * ratio)}g"
                        
                    total_cal += meal["calories"]
                    total_prot += m_prot
                    total_carbs += m_carbs
                    total_fat += m_fat

        output["daily_totals"] = {
            "calories": round(total_cal, 1),
            "protein":  f"{round(total_prot,  1)}g",
            "carbs":    f"{round(total_carbs, 1)}g",
            "fat":      f"{round(total_fat,   1)}g"
        }
        logger.info(f"✅ [Nutrition Validator] Final Totals: {total_cal:.0f} kcal, {total_prot:.0f}g protein")
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
