from typing import Dict, Any, List, Optional
from app.core.state import AgentState
from app.tools.nutrition_tools import NutritionRAGTool
from app.tools.web_search_tool import WebSearchTool
from app.agents.base import BaseRAGAgent
from pydantic import BaseModel, Field
from app.utils.logger import logger

class MealPlanItem(BaseModel):
    day: Optional[str] = Field(default="", description="Day label for multi-day plans (e.g., 'Day 1 - Monday (Training Day)'). Leave empty for single-day plans.")
    type: str = Field(description="Meal type (e.g., Breakfast, Lunch, Pre-Workout Snack, Dinner).")
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
    sub_queries: List[str] = Field(
        default=[], 
        description="3 alternative search phrases. CRITICAL: If the user has a specific diet (e.g. 'non-veg', 'vegan'), you MUST dynamically translate it into specific food keywords in these queries (e.g. 'chicken, meat, eggs' for non-veg) because semantic search struggles with abstract diet names like 'non-veg'."
    )
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
- DIETARY FLEXIBILITY (CRITICAL): You MUST strictly respect the user's dietary preferences. Apply these rules WITHOUT EXCEPTION:

  🥗 VEGETARIAN / VEG / PURE VEG:
    - NO meat, NO fish, NO seafood, NO eggs
    - Dairy (milk, curd, paneer, cheese, ghee, butter) IS allowed

  🌱 VEGAN:
    - NO meat, NO fish, NO seafood, NO eggs
    - NO dairy of ANY kind: no milk, no curd/dahi, no raita, no paneer, no cheese, no butter, no ghee, no whey protein
    - NO honey
    - ONLY plant-based: legumes, vegetables, fruits, nuts, seeds, plant-based oils, tofu, tempeh

  🍗 NON-VEG / NON-VEGETARIAN / MEAT-EATER:
    - Include healthy animal proteins: chicken, fish, eggs, lean meats
    - Do NOT default to veg items — actively include non-veg options

  🥑 KETO:
    - Fat MUST be 70-80% of daily calories
    - Carbs MUST be under 50g per day
    - NO grains, NO rice, NO bread, NO sugar, NO high-carb fruits
    - Include: eggs, fatty fish, chicken, nuts, avocado, olive oil, cheese, paneer (if not vegan)

  ⚖️ NO PREFERENCE: Provide a balanced mixed diet. No restrictions.

- There are NO global food restrictions beyond what the user's diet type requires.
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
- SANITY RULE 6 (NO DUPLICATES & EXPERT FALLBACK): NEVER repeat the same food item in more than one meal slot. If the database returns limited items, use your EXPERT KNOWLEDGE to generate healthy, goal-aligned meals consistent with the user's dietary preference.

MULTI-DAY PLAN RULES (CRITICAL):
- Detect exactly what duration (N days) the user is asking for from their message:
  • If N=1 (e.g. "daily", "today", or no duration) → Generate exactly 1 day (4-5 meals), leave `day` field empty.
  • If N is between 2 and 7 (e.g. "weekly" = 7 days, "5 days") → Generate exactly N unique days.
    - Each day MUST have DIFFERENT foods and calorie targets.
    - Populate `day` field: e.g. "Day 1 - Monday (Training Day)".
  • If N > 7 (e.g. "monthly" = 30/31 days, "45 days", "2 weeks") → Real Gym Approach: Generate exactly a 7-DAY ROTATION TEMPLATE.
    - DO NOT generate N days. Generate a 7-day template that the user will repeat.
    - CRITICAL INSTRUCTION: You MUST generate exactly 7 unique days in the JSON output! This means your `meals` list MUST contain EXACTLY 35 items (Day 1 x 5 meals ... up to Day 7 x 5 meals). DO NOT stop after Day 1!
    - ANTI-LAZINESS RULE: DO NOT copy and paste the same meals from Day 1 to Day 2, Day 3, etc. Every single one of the 35 meals MUST be completely different. No day should look like another day. If the database context runs out, use your EXPERT KNOWLEDGE to invent new, goal-aligned meals.
    - In `summary`: Write a dynamic progression plan explaining how the user should adjust their macros/calories phase-by-phase (e.g., week-by-week) across the full N days to reach their goal based on their TDEE.
    - In `tip`: Explain how to repeat this 7-day template across the requested duration.
    - Populate `day` field: e.g. "Day 1 - Monday (Training Day)" up through "Day 7 - Sunday (Rest Day)".
- Calorie and macro targets MUST vary dynamically per day (training day ≠ rest day).

GOAL-SPECIFIC DIETARY RULES (MANDATORY):

🔴 WEIGHT LOSS (when user mentions: lose weight, fat loss, slim down, lose Xkg):
- Daily calories: Dynamically calculate a sustainable, healthy calorie deficit based on the user's estimated TDEE. You MUST strictly avoid generating unsustainably low calorie counts.
- Protein: Dynamically calculate optimal protein intake based on the user's estimated body weight to preserve muscle mass.
- CLEAN EATING MANDATE: You MUST prioritize clean, high-volume, single-ingredient foods (e.g., dal, oats, sprouts, roasted chana, grilled tofu/paneer, huge green salads, khichdi, boiled vegetables).
- STRICTLY AVOID recommending heavy comfort foods, fried foods (pakoras, nuggets, vada, fries), and calorie-dense pasta/lasagne/burgers on a daily basis. These are NOT fat loss foods.
- Prefer: High-protein lean sources tailored dynamically to the user's specific diet type.

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

        # --- Step 1: Remove duplicate food items (per-day scope) ---
        # CRITICAL: key = day + name
        # Same food on DIFFERENT days = ALLOWED (e.g. "Chicken" on Day 1 AND Day 4)
        # Same food on SAME day = BLOCKED (e.g. "Chicken" twice on Day 1)
        seen = set()
        unique_meals = []
        for meal in meals:
            day_label  = meal.get("day",  "").strip()
            name_label = meal.get("name", "").lower().strip()
            key = f"{day_label}|{name_label}"
            if key not in seen:
                seen.add(key)
                unique_meals.append(meal)
            else:
                logger.warning(f"❌ [Nutrition Validator] Duplicate in same day: '{meal.get('name')}' on '{day_label}'")
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
            portion_str = str(meal.get("portion", "100g"))
            portion_g = parse_num(portion_str) or 100.0

            # Physics Validator: Macros cannot exceed portion size
            total_macros = prot + carbs + fat
            if total_macros > portion_g:
                logger.warning(f"⚠️ [Physics Validator] {meal.get('name')}: macros ({total_macros}g) exceed portion ({portion_g}g). Scaling down.")
                ratio = portion_g / max(total_macros, 1)
                prot *= ratio
                carbs *= ratio
                fat *= ratio
                meal["protein"] = f"{round(prot, 1)}g"
                meal["carbs"] = f"{round(carbs, 1)}g"
                meal["fat"] = f"{round(fat, 1)}g"

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
                max_possible_kcal = portion_g * 9  # Absolute physical maximum
                if llm_calories > max_possible_kcal:
                    logger.warning(
                        f"⚠️ [Density Validator] '{meal.get('name')}': {llm_calories} kcal "
                        f"for {portion_g}g is physically impossible (max={max_possible_kcal}). Capping."
                    )
                    # Cap at a reasonable density (~4 kcal/g = mixed food average)
                    meal["calories"] = round(portion_g * 4.0, 1)

        output["meals"] = unique_meals

        # --- Step 3: Detect plan duration from day fields ---
        unique_days = set(
            m.get("day", "").strip()
            for m in unique_meals
            if m.get("day", "").strip()  # only non-empty day fields
        )
        num_days = max(len(unique_days), 1)  # 1 = daily plan, 7 = weekly, 4 = monthly patterns
        is_multi_day = num_days > 1

        # --- Step 4: Sum totals across all meals ---
        total_cal, total_prot, total_carbs, total_fat = 0, 0, 0, 0
        per_day_data = {}

        for meal in unique_meals:
            day_label = meal.get("day", "").strip()
            
            c = parse_num(meal.get("calories", 0))
            p = parse_num(meal.get("protein", 0))
            cb = parse_num(meal.get("carbs", 0))
            f = parse_num(meal.get("fat", 0))

            if day_label:
                if day_label not in per_day_data:
                    per_day_data[day_label] = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}

                per_day_data[day_label]["calories"] += c
                per_day_data[day_label]["protein"] += p
                per_day_data[day_label]["carbs"] += cb
                per_day_data[day_label]["fat"] += f

            total_cal += c
            total_prot += p
            total_carbs += cb
            total_fat += f

        # --- Step 5: GOAL-AWARE AUTO SCALER ---
        target_calories = None
        if state and "user_context" in state:
            target_calories = state["user_context"].get("target_calories")

        if target_calories and total_cal > 0:
            if is_multi_day:
                avg_daily_cal = total_cal / num_days
                if abs(avg_daily_cal - target_calories) / target_calories > 0.05:
                    ratio = target_calories / avg_daily_cal
                    logger.info(f"⚖️ [Auto-Scaler] Multi-day plan. Target Avg: {target_calories} | Actual Avg: {avg_daily_cal:.1f}. Scaling all days universally by {ratio:.2f}")

                    total_cal, total_prot, total_carbs, total_fat = 0, 0, 0, 0
                    for day_label in per_day_data:
                        per_day_data[day_label] = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}

                    for meal in unique_meals:
                        m_prot  = parse_num(meal.get("protein", 0)) * ratio
                        m_carbs = parse_num(meal.get("carbs",   0)) * ratio
                        m_fat   = parse_num(meal.get("fat",     0)) * ratio

                        meal["protein"] = f"{round(m_prot,  1)}g"
                        meal["carbs"]   = f"{round(m_carbs, 1)}g"
                        meal["fat"]     = f"{round(m_fat,   1)}g"
                        meal["calories"] = round((m_prot * 4) + (m_carbs * 4) + (m_fat * 9), 1)

                        portion_g = parse_num(meal.get("portion", 0))
                        if portion_g > 0:
                            meal["portion"] = f"{round(portion_g * ratio)}g"

                        total_cal   += meal["calories"]
                        total_prot  += m_prot
                        total_carbs += m_carbs
                        total_fat   += m_fat
                        
                        day_label = meal.get("day", "").strip()
                        if day_label:
                            per_day_data[day_label]["calories"] += meal["calories"]
                            per_day_data[day_label]["protein"] += m_prot
                            per_day_data[day_label]["carbs"] += m_carbs
                            per_day_data[day_label]["fat"] += m_fat
            else:
                if abs(total_cal - target_calories) / target_calories > 0.05:
                    ratio = target_calories / total_cal
                    logger.info(f"⚖️ [Auto-Scaler] Single-day plan. Target: {target_calories} | Actual: {total_cal}. Scaling by {ratio:.2f}")

                    total_cal, total_prot, total_carbs, total_fat = 0, 0, 0, 0
                    for meal in unique_meals:
                        m_prot  = parse_num(meal.get("protein", 0)) * ratio
                        m_carbs = parse_num(meal.get("carbs",   0)) * ratio
                        m_fat   = parse_num(meal.get("fat",     0)) * ratio

                        meal["protein"] = f"{round(m_prot,  1)}g"
                        meal["carbs"]   = f"{round(m_carbs, 1)}g"
                        meal["fat"]     = f"{round(m_fat,   1)}g"
                        meal["calories"] = round((m_prot * 4) + (m_carbs * 4) + (m_fat * 9), 1)

                        portion_g = parse_num(meal.get("portion", 0))
                        if portion_g > 0:
                            meal["portion"] = f"{round(portion_g * ratio)}g"

                        total_cal   += meal["calories"]
                        total_prot  += m_prot
                        total_carbs += m_carbs
                        total_fat   += m_fat

        # Format per-day totals for the API response (only for multi-day plans)
        if is_multi_day and per_day_data:
            formatted_per_day = {}
            for day, totals in per_day_data.items():
                formatted_per_day[day] = {
                    "calories": round(totals["calories"], 1),
                    "protein": f"{round(totals['protein'], 1)}g",
                    "carbs": f"{round(totals['carbs'], 1)}g",
                    "fat": f"{round(totals['fat'], 1)}g"
                }
            output["per_day_totals"] = formatted_per_day

        # daily_totals = per-day average for multi-day plans, actual total for single-day
        output["daily_totals"] = {
            "calories": round(total_cal   / num_days, 1),
            "protein":  f"{round(total_prot  / num_days, 1)}g",
            "carbs":    f"{round(total_carbs / num_days, 1)}g",
            "fat":      f"{round(total_fat   / num_days, 1)}g",
        }
        if is_multi_day:
            output["daily_totals"]["note"] = f"Average per day across {num_days} days"

        logger.info(f"✅ [Nutrition Validator] Plan: {num_days} day(s) | Avg/day: {output['daily_totals']['calories']} kcal")
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

                # --- Code-level filter: reject physically impossible calorie density (4-4-9 rule) ---
                # Compute the theoretical maximum calories this food can have from its macros.
                # Per-100g base: protein × 4 + carbs × 4 + fat × 9.
                # If carbs are missing we use the absolute physical ceiling (100g × 9 kcal/g = 900).
                raw_carbs_check = r.get('carbs', 'Unknown')
                carbs_check: float | None = None
                try:
                    if raw_carbs_check not in ('N/A', 'Unknown', None, '', 0, '0'):
                        carbs_check = float(raw_carbs_check)
                except (ValueError, TypeError):
                    carbs_check = None

                if carbs_check is not None:
                    # Full 4-4-9 maximum: every gram of macro at its highest caloric equivalent
                    max_possible_kcal = (prot * 4) + (carbs_check * 4) + (fat * 9)
                else:
                    # Carbs unknown — use absolute physical ceiling (pure fat = 9 kcal/g × 100g)
                    max_possible_kcal = 900.0

                if cal > max_possible_kcal:
                    logger.warning(
                        f"⚠️ [Nutrition DB Filter] Skipping '{name}': {cal} kcal/100g exceeds "
                        f"the 4-4-9 theoretical maximum ({max_possible_kcal:.0f} kcal) — physically impossible."
                    )
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
