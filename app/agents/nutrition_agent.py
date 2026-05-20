from typing import Dict, Any, List, Optional
from app.core.state import AgentState
from app.tools.nutrition_tools import NutritionRAGTool
from app.tools.web_search_tool import WebSearchTool
from app.agents.base import BaseRAGAgent
from pydantic import BaseModel, Field
from app.utils.logger import logger

class MealPlanItem(BaseModel):
    day: Optional[str] = Field(default="", description="Day label for multi-day plans (e.g., 'Day 1 - Monday (Training Day)'). Leave empty for single-day plans.")
    type: Optional[str] = Field(default="", description="Meal type (e.g., Breakfast, Lunch, Pre-Workout Snack, Dinner).")
    name: Optional[str] = Field(default="", description="Name of the dish/food.")
    portion: Optional[str] = Field(default="", description="Amount of food to eat (e.g. '300g' or '2 cups'). MUST scale to hit the daily calorie goal!")
    calories: float = Field(default=0.0, description="Total calories for this specific portion (can be decimal).")
    protein: Optional[str] = Field(default="", description="Total protein for this portion in grams (e.g., '12g').")
    carbs: Optional[str] = Field(default="", description="Total carbs for this portion in grams (e.g., '18g').")
    fat: Optional[str] = Field(default="", description="Total fat for this portion in grams (e.g., '6g').")
    benefit: Optional[str] = Field(default="", description="Why this meal helps the user's goal.")

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
    
    PESCATARIAN:
    - NO red meat (beef, pork, lamb, mutton) and NO poultry (chicken, turkey).
    - Seafood (fish, shellfish) is the ONLY animal protein.
    - Include: whole grains, legumes, vegetables, fruits, nuts, seeds, healthy oils.
    - Optional: eggs and dairy (if not vegan).
    
    POLLOTARIAN:
    - NO red meat (beef, pork, lamb, mutton) and NO seafood/fish.
    - Poultry (chicken, turkey) is the ONLY animal protein.
    - Include: whole grains, legumes, vegetables, fruits, nuts, seeds, healthy oils, eggs and dairy.

    FLEXITARIAN:
    - STRICT RULE: NEVER include lamb, beef, pork, or mutton.
    - Primary Protein: Plant-based proteins (beans, lentils, tofu, tempeh) are the main focus.
    - Meat allowance: Chicken or fish can be consumed occasionally, but NO red meat.
    - Include: whole grains, legumes, vegetables, fruits, nuts, seeds, healthy oils, eggs and dairy.
    

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

- USER QUERY STRICTNESS RULE:
  Always strictly follow the user's requested duration and structure.
  - If the user explicitly asks for a specific number of days (e.g. "3-day plan", "10 days", "2 weeks", "monthly"), generate the response according to that exact request.
  - If the user asks for specific day types (e.g. "only training days", "weekdays only", "Monday to Friday", "rest day meal plan"), generate meals only for those requested days/types.
  - If the user asks for a specific meal frequency (e.g. "3 meals per day", "6 meals", "intermittent fasting"), strictly follow that structure.
  - If the user asks for a specific schedule pattern (e.g. alternating training/rest days, veg on weekdays and non-veg on weekends), adapt the generated plan dynamically according to the query.
  - Never assume a default duration or structure when the user has explicitly provided one.
  - The generated JSON output MUST always align exactly with the user's requested duration, schedule, meal frequency, and day pattern.

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

USER BIOMETRIC DATA:
Name: {full_name}
Age: {age} | Gender: {gender}
Weight: {weight_kg} kg | Height: {height_cm} cm
Activity Level: {activity_level}
TDEE & Calorie Targets:
  {tdee}
Goal: {goal}
Medical Conditions/Allergies: {medical}
Dietary Preference: {diet_preference}

Current Context: {summary}

{intelligence_context}
"""

        super().__init__(
            agent_name="Nutrition Agent",
            rag_tool=NutritionRAGTool(),
            web_search_tool=WebSearchTool(),
            output_schema=NutritionAnalysis,
            system_prompt=system_prompt
        )
        self._rotation_size = self._compute_chunk_size()

    async def _detect_n_days(self, query: str) -> int:
        from langchain_openai import ChatOpenAI
        from app.core.config import settings

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)
        prompt = (
            "You are a duration parser. Read the user's message and determine how many total days "
            "they are requesting a fitness or diet plan for.\n"
            "Convert any time expression (in any language) to a number of days.\n"
            "If no duration is mentioned, return 1.\n"
            "Reply with ONLY a single integer. No explanation, no units, just the number.\n\n"
            f"User message: {query}\n\n"
        )
        try:
            response = await llm.ainvoke(prompt)
            n = int(response.content.strip())
            return max(n, 1)
        except Exception:
            return 1

    async def run(self, state: AgentState) -> Dict[str, Any]:
        state = dict(state)
        self._rotation_size = self._compute_chunk_size()
        state.setdefault("_extra_prompt_vars", {})["rotation_size"] = self._rotation_size

        query = state['messages'][-1].content
        n_days = await self._detect_n_days(query)
        chunk_size = self._compute_chunk_size()

        if n_days > chunk_size:
            return await self._run_chunked(state, query, n_days)
        else:
            return await self.run_logic(state, specialist_key="nutrition", topic="nutrition")

    def _compute_chunk_size(self) -> int:
        """
        Formula:
          chunk_size = floor(safe_output_tokens / (meals_per_day × tokens_per_meal))

        These are physical model constraints, not business rules:
          - safe_output_tokens: gpt-4o-mini max output (4096), with 10% safety margin
          - tokens_per_meal: 130 tokens per structured meal JSON object (safer buffer)
          - meals_per_day: 5 (standard full-day coverage)
        """
        safe_output_tokens = int(self.llm.max_tokens * 0.90) if hasattr(self.llm, 'max_tokens') and self.llm.max_tokens else 3680
        tokens_per_meal = 130
        meals_per_day = 5
        return max(1, safe_output_tokens // (tokens_per_meal * meals_per_day))

    def _compute_days_to_generate(self, n_days: int) -> int:
        """
        How many unique days to actually generate.
        If n_days fits within max_api_calls × chunk_size, generate ALL n_days.
        Otherwise generate as many as possible within a reasonable API call budget.

        max_api_calls is derived from UX budget (each call ~15s, total <90s acceptable):
        max_api_calls = floor(acceptable_wait_seconds / seconds_per_call)
        """
        seconds_per_call = 15       # approximate LLM call latency
        acceptable_wait = 90        # max seconds user waits comfortably
        max_api_calls = max(1, acceptable_wait // seconds_per_call)  # = 6
        chunk_size = self._compute_chunk_size()
        max_generatable = chunk_size * max_api_calls
        return min(n_days, max_generatable)

    def _build_chunks(self, days_to_generate: int) -> list:
        """
        Dynamically partition days_to_generate into chunks.
        Each chunk size = _compute_chunk_size() (token-budget derived).
        Returns: [{"label": "Days 1-8", "days": "Day 1, Day 2, ..., Day 8"}, ...]
        """
        chunk_size = self._compute_chunk_size()
        chunks = []
        day_num = 1
        while day_num <= days_to_generate:
            end = min(day_num + chunk_size - 1, days_to_generate)
            day_labels = [f"Day {i}" for i in range(day_num, end + 1)]
            chunks.append({
                "label": f"Days {day_num}-{end}" if day_num != end else f"Day {day_num}",
                "days": ", ".join(day_labels)
            })
            day_num = end + 1
        return chunks

    async def _generate_plan_summary(self, all_meals: list, n_days: int, days_generated: int, goal: str, original_query: str) -> tuple:
        """
        Generate plan summary + tip via a dedicated LLM call AFTER all chunks are merged.
        This ensures summary reflects the FULL plan, not just the first chunk.
        Returns (summary_str, tip_str, answer_str).
        """
        from langchain_openai import ChatOpenAI
        from app.core.config import settings

        day_labels = list({m.get('day', '') for m in all_meals if m.get('day', '')})
        day_labels_str = ', '.join(sorted(day_labels)[:5])  # show first 5 for brevity
        is_rotation = n_days > days_generated

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=settings.OPENAI_API_KEY)
        prompt = (
            f"You are a nutrition coach. A {days_generated}-day meal plan has been generated for a user with goal: {goal}.\n"
            f"The plan covers: {day_labels_str}{'...' if len(day_labels) > 5 else ''}.\n"
            + (f"This is a rotation template — the user should repeat it to cover all {n_days} days.\n" if is_rotation else "")
            + f"Original request: '{original_query}'\n\n"
            "Write THREE short paragraphs in JSON format:\n"
            '{"summary": "<2 sentences: what the plan covers and how it supports the goal>", '
            '"tip": "<1 sentence: actionable nutrition tip or how to use the rotation>", '
            '"answer": "<1 warm motivating sentence to open the response>"}\n'
            "Reply with ONLY the JSON object, no markdown."
        )
        try:
            resp = await llm.ainvoke(prompt)
            import json
            data = json.loads(resp.content.strip())
            return data.get("summary", ""), data.get("tip", ""), data.get("answer", "")
        except Exception as e:
            logger.warning(f"⚠️ [Nutrition Chunked] Summary generation failed: {e}")
            rotation_note = f" Repeat this {days_generated}-day rotation to cover all {n_days} days." if is_rotation else ""
            return (
                f"Here is your {days_generated}-day meal plan tailored for your {goal} goal.",
                f"Stay consistent with your meals and hydration.{rotation_note}",
                "Here is your personalized meal plan!"
            )

    async def _run_chunked(self, state: AgentState, original_query: str, n_days: int) -> Dict[str, Any]:
        """
        Fully dynamic chunked generation — zero hardcoded values.
        - days_to_generate: computed from token budget × UX latency budget
        - chunk_size: computed from model output token limit
        - summary/tip: generated post-merge via dedicated LLM call
        - day labels: LLM decides (Training/Rest/etc) based on user goal
        """
        from langchain_core.prompts import ChatPromptTemplate

        user_context = state.get('user_context', {}) or {}
        conv_summary = state.get('conversation_summary', "No previous context.")

        # ── Extract full profile (same as base.py run_logic) ─────────────────
        goal           = user_context.get("goal", "General Fitness") or "General Fitness"
        diet_pref      = user_context.get("diet_preference") or "None"
        injuries_list  = user_context.get("injuries", []) or []
        injuries       = ", ".join(str(i) for i in injuries_list) if injuries_list else "None"
        med_list       = user_context.get("medical_conditions", []) or []
        medical        = ", ".join(str(m) for m in med_list) if med_list else "None"
        full_name      = user_context.get("full_name") or "User"
        age            = str(user_context.get("age") or "Unknown")
        gender         = str(user_context.get("gender") or "Unknown")
        weight_kg      = user_context.get("weight_kg") or "Unknown"
        height_cm      = str(user_context.get("height_cm") or "Unknown")
        activity_level = str(user_context.get("activity_level") or "Unknown")

        # ── Compute TDEE + Intelligence Context (same as base.py) ─────────
        from app.agents.base import _compute_tdee
        tdee = _compute_tdee(weight_kg, height_cm, age, gender, activity_level)
        if tdee["tdee"]:
            tdee_str = (
                f"BMR {tdee['bmr']} kcal | TDEE {tdee['tdee']} kcal/day\n"
                f"  Weight-loss: {tdee['cal_loss']} kcal | "
                f"Maintenance: {tdee['cal_maintenance']} kcal | "
                f"Weight-gain: {tdee['cal_gain']} kcal"
            )
        else:
            tdee_str = "Unknown — profile incomplete."

        intelligence_context = self._build_intelligence_context(
            weight_kg, goal, tdee, diet_pref, injuries, medical
        )

        # ── Step 1: Profile-enriched RAG search (Layer 1) ──────────────
        logger.info(f"🔍 [Nutrition Chunked] Single RAG search for context...")
        enriched_query = self._build_enriched_query(original_query, goal, diet_pref, weight_kg, injuries)
        db_results = await self.rag_tool.search(enriched_query, diet_preference=diet_pref)
        context_str = self._format_context(db_results) or "No specific data retrieved."

        # ── Step 2: Compute rotation/chunk plan (all dynamic) ────────────────
        rotation_size    = self._compute_chunk_size()        # token-budget derived (e.g. 8)
        is_rotation      = n_days > rotation_size             # long plan = rotation template
        days_to_generate = rotation_size if is_rotation else n_days  # generate only 1 cycle for long plans
        DAY_CHUNKS       = self._build_chunks(days_to_generate)
        import math as _math
        repeat_times     = _math.ceil(n_days / rotation_size) if is_rotation else 1
        logger.info(
            f"📦 [Nutrition Chunked] n_days={n_days} | rotation={is_rotation} | "
            f"rotation_size={rotation_size} | days_to_generate={days_to_generate} | "
            f"repeat={repeat_times}x | chunks={[c['label'] for c in DAY_CHUNKS]}"
        )

        # ── Compute meal count dynamically from query OR token budget ──────────
        import re as _re
        _meal_match = _re.search(r'(\d+)\s*(meal|meals)', original_query.lower())
        if _meal_match:
            meal_count = max(1, min(int(_meal_match.group(1)), 8))  # cap at 8 (physics)
        else:
            # Derive from token budget: tokens_per_day = meal_count * 85
            # chunk_size = safe_tokens / (meal_count * 85) → meal_count = safe_tokens / (chunk_size * 85)
            # Default: use same meals_per_day assumption as _compute_chunk_size (5)
            meal_count = 5
        logger.info(f"🍽️ [Nutrition Chunked] Meal count per day: {meal_count} (from query or default)")

        # ── Step 3: Chunk prompt ────────────────────────────────────────
        chunk_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt.messages[0].prompt.template),
            ("human", (
                "CONVERSATION SUMMARY:\n{conv_summary}\n\n"
                "ORIGINAL REQUEST: {original_query}\n"
                "USER GOAL: {goal} | INJURIES: {injuries} | DIET: {diet_pref}\n\n"
                "{rotation_instruction}"
                "TASK: Generate meals ONLY for: {days}\n"
                "- Generate exactly {meal_count} meals per day. "
                "  Choose meal types (Breakfast / Lunch / Snack / Dinner / Post-Workout / Pre-Workout / etc.) "
                "  that best fit the user's goal and day type.\n"
                "- Each day must have DIFFERENT foods — no repeats across days\n"
                "- In the `day` field use: 'Day X - [Day Type]' where you decide "
                "  [Day Type] based on user goal (Training Day / Rest Day / Active Recovery / High Carb / etc.)\n"
                "- Set is_accurate=True, needs_web_search=False, sub_queries=[]\n\n"
                "FOOD DATA:\n{context}"
            ))
        ])
        chunk_chain = chunk_prompt | self.llm

        # ── Step 4: Sequential chunk calls ─────────────────────────────
        all_meals = []
        for chunk in DAY_CHUNKS:
            logger.info(f"🍽️  [Nutrition Chunked] Generating {chunk['label']}...")
            try:
                analysis = await chunk_chain.ainvoke({
                    # Human prompt variables
                    "conv_summary":         conv_summary,
                    "original_query":       original_query,
                    "goal":                 goal,
                    "injuries":             injuries,
                    "diet_pref":            diet_pref,
                    "meal_count":           meal_count,
                    "days":                 chunk["days"],
                    "context":              context_str,
                    "rotation_instruction": (
                        f"THIS IS A ROTATION TEMPLATE (not {n_days} unique days).\n"
                        f"User wants {n_days} days. Generate {rotation_size} unique days as a repeating cycle.\n"
                        f"They will repeat this cycle {repeat_times}x to complete {n_days} days.\n\n"
                    ) if is_rotation else "",
                    # System prompt variables
                    "summary":              conv_summary,
                    "diet_preference":      diet_pref,
                    "full_name":            full_name,
                    "age":                  age,
                    "gender":               gender,
                    "weight_kg":            str(weight_kg),
                    "height_cm":            height_cm,
                    "activity_level":       activity_level,
                    "tdee":                 tdee_str,
                    "medical":              medical,
                    "intelligence_context": intelligence_context,
                    "rotation_size":        rotation_size,
                })
                chunk_meals = analysis.meals if hasattr(analysis, 'meals') else []
                all_meals.extend([m.model_dump() if hasattr(m, 'model_dump') else m for m in chunk_meals])
                logger.info(f"✅ [Nutrition Chunked] {chunk['label']} — {len(chunk_meals)} meals")
            except Exception as e:
                logger.error(f"❌ [Nutrition Chunked] Failed {chunk['label']}: {e}")

        logger.info(f"✅ [Nutrition Chunked] {len(all_meals)} total meals across {days_to_generate} days.")

        # ── Step 4.5: Expand rotation cycle to cover n_days (Frontend Expects Full Array) ──
        if is_rotation and all_meals:
            logger.info(f"🔄 [Nutrition Chunked] Expanding {days_to_generate}-day cycle to {n_days} days for UI.")
            import math
            import re
            
            # Group meals by original day string
            day_groups = []
            current_day_str = None
            current_group = []
            
            for m in all_meals:
                m_day = m.get("day", "")
                if m_day != current_day_str:
                    if current_group:
                        day_groups.append(current_group)
                    current_day_str = m_day
                    current_group = []
                current_group.append(m)
            if current_group:
                day_groups.append(current_group)
                
            actual_cycle_length = len(day_groups)
            expanded_meals = []
            
            if actual_cycle_length > 0:
                for target_day in range(1, n_days + 1):
                    # Zero-indexed day in the cycle
                    cycle_idx = (target_day - 1) % actual_cycle_length
                    cycle_num = math.ceil(target_day / actual_cycle_length)
                    
                    original_group = day_groups[cycle_idx]
                    
                    for m in original_group:
                        new_meal = m.copy()
                        orig_day = new_meal.get("day", "")
                        # Try to remove the "Day X - " prefix if it exists
                        day_type = orig_day
                        match = re.search(r'Day \d+\s*-\s*(.*)', orig_day, re.IGNORECASE)
                        if match:
                            day_type = match.group(1).strip()
                            
                        # Format the new day string: Day 15 - Training Day (Cycle 3)
                        new_meal["day"] = f"Day {target_day} - {day_type} (Cycle {cycle_num})"
                        
                        # DYNAMIC PROGRESSION: Slightly adjust portions/calories for later cycles
                        # This ensures the plan is not 100% statically copied, matching real-world macro progression.
                        if cycle_num > 1:
                            try:
                                p_val = float(str(new_meal.get("protein", "0")).replace("g", "")) * (1 + 0.02 * (cycle_num - 1))
                                c_val = float(str(new_meal.get("carbs", "0")).replace("g", "")) * (1 + 0.02 * (cycle_num - 1))
                                f_val = float(str(new_meal.get("fat", "0")).replace("g", "")) * (1 + 0.02 * (cycle_num - 1))
                                portion_val = float(str(new_meal.get("portion", "0")).replace("g", "")) * (1 + 0.02 * (cycle_num - 1))
                                
                                new_meal["protein"] = f"{round(p_val, 1)}g"
                                new_meal["carbs"] = f"{round(c_val, 1)}g"
                                new_meal["fat"] = f"{round(f_val, 1)}g"
                                new_meal["calories"] = round((p_val * 4) + (c_val * 4) + (f_val * 9), 1)
                                if portion_val > 0:
                                    new_meal["portion"] = f"{round(portion_val)}g"
                            except Exception:
                                pass

                        expanded_meals.append(new_meal)
                        
                all_meals = expanded_meals
                logger.info(f"✅ [Nutrition Chunked] Expanded to {len(all_meals)} total meals for {n_days} days.")

        # ── Step 5: Generate summary/tip/answer POST-MERGE (full plan aware) ──
        summary_str, tip_str, answer_str = await self._generate_plan_summary(
            all_meals, n_days, days_to_generate, goal, original_query
        )

        # ── Step 6: Build & validate output ────────────────────────────
        specialist_output = {
            "answer": answer_str,
            "status": "success",
            "summary": summary_str,
            "meals": all_meals,
            "tip": tip_str,
            "daily_totals": None,
        }
        specialist_output = self._validate_output(specialist_output, context_str, state)
        return {"specialist_results": {"nutrition": specialist_output}}


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

                        # CAP unrealistic protein in a single non-supplement meal (Fix for 67g salad issue)
                        meal_name = meal.get("name", "").lower()
                        if m_prot > 50.0 and "shake" not in meal_name and "protein powder" not in meal_name and "whey" not in meal_name:
                            m_prot = 50.0

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

                        # CAP unrealistic protein
                        meal_name = meal.get("name", "").lower()
                        if m_prot > 50.0 and "shake" not in meal_name and "protein powder" not in meal_name and "whey" not in meal_name:
                            m_prot = 50.0

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

        # --- Step 6: PROTEIN FLOOR VALIDATOR (Layer 3) ---
        # Physiology constant: 1.6g/kg is the minimum protein for any active person.
        # This is NOT goal-specific — it's a universal floor. No hardcoded goal logic.
        # If avg daily protein is below this floor, scale protein up while keeping
        # total calories intact by reducing fat proportionally.
        try:
            w_kg = float(state["user_context"].get("weight_kg", 0)) if state and "user_context" in state else 0.0
        except (ValueError, TypeError):
            w_kg = 0.0

        if w_kg > 0 and total_prot > 0:
            protein_floor_total = w_kg * 1.6 * num_days  # total across all days
            if total_prot < protein_floor_total:
                prot_ratio = protein_floor_total / total_prot
                logger.info(
                    f"💪 [Protein Floor] Avg {round(total_prot/num_days,1)}g/day < floor "
                    f"{round(w_kg*1.6,1)}g/day → scaling protein ×{prot_ratio:.2f}"
                )
                total_prot_new = 0.0
                total_fat_new  = 0.0
                total_cal_new  = 0.0
                for meal in unique_meals:
                    m_prot  = parse_num(meal.get("protein", 0)) * prot_ratio
                    m_fat   = parse_num(meal.get("fat",     0))
                    m_carbs = parse_num(meal.get("carbs",   0))
                    # Reduce fat to compensate the added protein calories so
                    # total calories stay close to target (1g protein = 4 kcal,
                    # 1g fat = 9 kcal — pure physiology, no goal rule).
                    added_prot_kcal = (m_prot - parse_num(meal.get("protein", 0))) * 4
                    fat_reduction_g = min(m_fat, added_prot_kcal / 9)
                    m_fat = max(0.0, m_fat - fat_reduction_g)

                    meal["protein"] = f"{round(m_prot,  1)}g"
                    meal["fat"]     = f"{round(m_fat,   1)}g"
                    meal["calories"] = round((m_prot*4) + (m_carbs*4) + (m_fat*9), 1)

                    total_prot_new += m_prot
                    total_fat_new  += m_fat
                    total_cal_new  += meal["calories"]

                total_prot  = total_prot_new
                total_fat   = total_fat_new
                total_cal   = total_cal_new

                # Rebuild per_day_data after protein scaling
                per_day_data = {}
                for meal in unique_meals:
                    day_label = meal.get("day", "").strip()
                    if day_label:
                        if day_label not in per_day_data:
                            per_day_data[day_label] = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
                        per_day_data[day_label]["calories"] += meal["calories"]
                        per_day_data[day_label]["protein"]  += parse_num(meal.get("protein", 0))
                        per_day_data[day_label]["carbs"]    += parse_num(meal.get("carbs", 0))
                        per_day_data[day_label]["fat"]      += parse_num(meal.get("fat", 0))

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
