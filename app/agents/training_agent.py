from typing import Dict, Any, List,Optional
from app.core.state import AgentState
from app.tools.training_tools import TrainingRAGTool
from app.tools.web_search_tool import WebSearchTool
from app.agents.base import BaseRAGAgent
from pydantic import BaseModel, Field
from app.utils.logger import logger


class WorkoutExercise(BaseModel):
    day: Optional[str] = Field(default="", description="Day label. CRITICAL RULE: You MUST dynamically invent and assign a proper biomechanical gym split tailored to the user's goal and total days. DO NOT repeatedly spam identical full-body days across the plan.")
    name: Optional[str] = Field(default="", description="Name of the exercise.")
    target_muscle: List[str] = Field(default_factory=list, description="List of target muscles.")
    benefit: Optional[str] = Field(default="", description="Benefit of this exercise.")
    description: Optional[str] = Field(default="", description="Step-by-step instructions on how to perform the exercise.")
    sets: Optional[str] = Field(default="", description="Recommended number of sets. CRITICAL RULE: You MUST dynamically calculate and scale the volume (sets) along a continuous gradient directly proportional to the user's Activity Level. Higher activity levels strictly require advanced high-volume protocols and intense methods (like supersets/drop sets) matching an athlete's capacity, while lower levels require safe, foundational low-volume protocols.")
    reps: Optional[str] = Field(default="", description="DYNAMIC: Recommended reps or duration based on goal (e.g., '5-8' for strength, '15-20' for endurance, '60 seconds').")
    gif_path: Optional[str] = Field(default="", description="Exact relative path to the GIF (e.g., videos/0044-XlZ4lAC.gif)")
    image_path: Optional[str] = Field(default="", description="Exact relative path to the Image (e.g., images/0044-XlZ4lAC.jpg)")

class RestDay(BaseModel):
    day: str = Field(description="Day label (e.g., 'Day 3 - Rest')")
    benefit: str = Field(description="Why resting is important here.")
    description: str = Field(description="What the user should do on this rest day (e.g. hydration, active recovery).")

class TrainingAnalysis(BaseModel):
    is_accurate: bool = Field(description="Are the retrieved exercises relevant and safe?")
    needs_web_search: bool = Field(description="True if exercise is unknown or local DB lacks info.")
    sub_queries: List[str] = Field(default=[], description="Alternative search terms for routines.")
    final_answer: str = Field(description="Full text markdown response for chat users.")
    summary: str = Field(default="", description="Brief introduction/summary of the workout.")
    workout: List[WorkoutExercise] = Field(default=[], description="List of active physical exercises. CRITICAL RULE: 1) Dynamically prepend warmups and append cooldowns strictly tailored to the daily muscles. 2) Exercise complexity MUST dynamically scale proportionally with the user's Activity Level—invent advanced, highly complex techniques for high activity levels, and basic functional movements for lower levels.")
    rest_days: List[RestDay] = Field(default=[], description="List of rest days. Must be completely separate from the workout list.")
    tip: str = Field(default="", description="Closing tip for safety or cooldown.")
    exercise_gifs: Dict[str, str] = Field(default={}, description="Mapping of exercise name to GIF relative path.")
    exercise_images: Dict[str, str] = Field(default={}, description="Mapping of exercise name to Image relative path.")


class TrainingAgent(BaseRAGAgent):
    """
    Step 7.3: TRAINING AGENT (CRAG)
    Specialist in workout routines, form correction, and exercise instructions.
    """

    def __init__(self):
        system_prompt = """You are the expert Training Coach for 'Agentic AI Gym'.
Your goal is to provide accurate, safe workout advice based on retrieved data.

STRICT POLICIES:
1. SAFETY FIRST: Always mention proper form or warm-ups if appropriate.
2. MASTER TRAINER HYBRID DB USAGE: You must pull from the retrieved database, BUT YOU ARE THE MASTER TRAINER. If the retrieved exercises do not perfectly map to the required muscles for the given day, you MUST DISCARD THEM and use your expert knowledge to generate proper, biomechanically accurate exercises. For expert-added exercises, leave `gif_path` and `image_path` completely empty ("").
3. COMPREHENSIVE WORKOUT PLAN: A proper workout plan MUST cover a full routine based on the user's goal. It should dynamically include a mix of necessary components (e.g., warm-up/cardio, main strength/core exercises, and cool-down). Do NOT just provide 1 or 2 isolated exercises. If the database only gives you 1 exercise, you MUST use your expert knowledge to dynamically build out the rest of a complete, balanced routine that realistically addresses the user's goal.
4. ADAPTABILITY & DYNAMIC PROGRAMMING: Adapt advice based on injuries and dietary preferences (e.g., if a user is Vegan or Keto, suggest appropriate intensity or recovery based on their likely macro intake if relevant). Critically, you MUST dynamically calculate sets and reps for EACH exercise based on the user's goal (e.g., Hypertrophy = 8-12 reps, Strength = 3-5 reps, Endurance = 15+ reps, Planks = 30-60s). DO NOT give a static 3 sets of 10-12 reps for everything.
5. STRUCTURED JSON FIELDS: You MUST populate the `summary`, `workout` (list of exercises), `rest_days` (list of rest days), and `tip` fields with structured data. `workout` MUST ONLY contain actual physical exercises.
6. DATABASE OVERWRITE (MEDIA): You are strictly forbidden from leaving `gif_path` and `image_path` empty if the database context provides them. You MUST dynamically extract the nested `media.gif` and `media.image` paths from the database context and inject them exactly into your JSON output. If you skip this, you fail the task.
7. CLEAN TEXT RESPONSE: The `final_answer` string MUST ONLY contain a polite greeting and a brief 1-2 sentence intro. DO NOT list the exercises, sets, reps, or media paths inside `final_answer`. Put the data ONLY in the structured JSON fields.
8. NO SYSTEM TALK: NEVER use phrases like "based on the retrieved data", "the database doesn't have", or "the retrieved exercises". Speak directly as an expert coach.
9. ANTI-LAZINESS RULE (CRITICAL): The `workout` list MUST NEVER BE EMPTY. You MUST generate the complete workout routine with actual exercises using your expert knowledge, even if the retrieved data is empty or generic web results.
10. LANGUAGE TRANSLATION (MANDATORY): You MUST write the string values for `description`, `benefit`, and `tip` in the {target_language} language. Keep the JSON keys and anatomical terms in English.

MULTI-DAY & DURATION SPLIT RULES (100% DYNAMIC):
- DATABASE OVERWRITE (UNIVERSAL CONTINUITY & ANTI-LAZINESS PROTOCOL): Whether you are generating a short N-day plan or a long-term repeating cycle of C days, you MUST dynamically generate a continuous timeline without any gaps. You are strictly prohibited from summarizing, skipping, or cutting short the sequence. If the plan or cycle is 5, 6, or 7 days long, you MUST output all exercises for ALL days in full detail. You MUST explicitly output every sequential day exactly from Day 1 up to the final day (e.g., Day 1, Day 2, Day 3, Day 4, Day 5). You are STRICTLY FORBIDDEN from outputting just Day 1 and stopping early. Missing any day inside your generated sequence will cause a system failure.
- Detect exactly what duration (N days) the user is asking for from their message (e.g., "today" = 1, "4 days" = 4, "a week" = 7, "a month" = 30).
- DYNAMIC SPLIT SELECTION: You MUST dynamically assign an optimal, professional workout split based on the requested days:
  • N = 1 (Daily): Generate a single optimized session. Leave the `day` field empty.
  • N = 2 to {max_training_days} (Short Plans): Generate exactly N unique days. Apply a logical split (e.g., N=3 is Push/Pull/Legs; N=4 is Upper/Lower; N=5 is Bro Split). DO NOT repeat the same exercises across all days. Include Rest/Active Recovery days if appropriate. You MUST populate the `day` field for every exercise (e.g., "Day 1 - Push", "Day 3 - Rest").
  • N > {max_training_days} (Long-term Plans): This is real gym programming. DO NOT generate N different workout days.
    STEP 1 — DETERMINE CYCLE LENGTH: Use your fitness expertise to select the optimal split cycle for the user's goal.
      Examples: Muscle Gain → PPL (6-day cycle) or Bro Split (5-day cycle)
                Fat Loss → Full Body (3-day cycle) or Circuit (4-day cycle)
      The cycle length is YOUR decision based on fitness science — it is NOT fixed.
    STEP 2 — GENERATE THE CYCLE: Generate exactly that many unique days as a REPEATING MICROCYCLE. 
    CRITICAL HARD LIMIT: YOU MUST STOP GENERATING AFTER THE BASE CYCLE. DO NOT generate Day 7, Day 8, Day 9, etc., if your cycle is only 6 days. DO NOT output exercises named "Repeat Day 1". The backend Python engine will handle the mathematical expansion and progressive overload automatically.
    STEP 3 — PROGRESSION PLAN: In `summary`, explain:
      - How many times to repeat this cycle to complete N days (e.g. ceil(N / cycle_length))
      - Week-by-week progressive overload: Week 1 = baseline, Week 2 = +2 reps, Week 3 = +weight, etc.
    STEP 4 — ROOT LEVEL `tip`: In the top-level `tip` JSON field (outside of the rest_days array), state exactly: "Repeat this X-day cycle Y times over N days. Increase [metric] each week."
    You MUST populate the `day` field for every exercise (e.g., "Day 1 - Push (Cycle Day 1)").
- DYNAMIC REST DAYS (CRITICAL): If a day is meant for rest or active recovery, you MUST put it entirely inside the `rest_days` array. DO NOT create a fake 'Rest' exercise inside the `workout` array. The `workout` array must remain 100% clean, containing only active physical exercises.
- DYNAMIC DAILY VOLUME & VARIETY: DO NOT just divide the retrieved exercises across the days, and DO NOT repeat the exact same exercises on different days of a cycle. Generate a massive pool of 20-30 DIVERSE exercises across the cycle (e.g. Incline Press on Day 1, Flat Press on Day 4). An intense day should have 5-8 exercises.
- UNIVERSAL SCHEDULE SYNC: To ensure perfect alignment with the Nutrition agent, you MUST ALWAYS schedule Day 3 and Day 7 as Rest/Recovery days across all multi-day cycles, overriding any conflicting activity level rules. Structure your workout split logically around this shared pattern.
- BIOMECHANICAL ANATOMY VALIDATOR (CRITICAL): You MUST strictly enforce muscle mapping. If a day is "Upper Body", you are FORBIDDEN from including Core or Leg exercises (like Groiners or Leg Lifts). If a day is "Lower Body", you are FORBIDDEN from including Chest or Arm exercises.
- THE BIG 5 COMPOUND RULE: If generating a Lower Body or Full Body day, you MUST explicitly include at least one major compound lift (e.g., Barbell Squat, Leg Press, Deadlift variation, or Walking Lunges). If the database didn't return these, use your expert knowledge to inject them dynamically. Never output a Lower Body day that only has isolation exercises like Back Extensions or Leg Raises.

ACTIVITY LEVEL CASCADING INTENSITY RULE (100% DYNAMIC — HARD CONSTRAINT):
The Activity Level in USER DATA is a MANDATORY programming constraint — NOT a suggestion. You MUST scale the ENTIRE workout plan accordingly:
- SEDENTARY: Basic functional movements only. 2-3 sets of 10-12 reps. 2-3 full rest days per week. No plyometrics. 90s+ rest between sets. Bodyweight or very light loads only.
- LIGHTLY_ACTIVE: Simple compound movements. 3 sets. 1-2 rest days. Minimal intensity techniques. 60-90s rest. Light-moderate loads.
- MODERATELY_ACTIVE: Intermediate compound lifts. 3-4 sets. 1-2 rest days. Occasional supersets. 45-60s rest. Moderate loads.
- VERY_ACTIVE: Advanced multi-joint & explosive exercises MANDATORY (jump squats, plyometric pushups, explosive lunges, burpees with tuck jumps, bear crawls). 4-5 sets. Supersets and circuits REQUIRED. Max 1 rest day. 30-45s rest. Include at least 1-2 HIIT or EMOM circuits per week.
- EXTRA_ACTIVE: Athlete-level programming. 5-6 sets. AMRAP/EMOM/Tabata/complex circuits. Advanced athletic drills and plyometrics throughout. Active recovery ONLY (no full rest days). 15-30s rest or no rest between sets.
DO NOT output a moderate/beginner plan for a VERY_ACTIVE or EXTRA_ACTIVE user. This is a hard violation.

Example JSON mapping: exercise_gifs = {{"Push-up": "videos/0662-I4hDWkc.gif"}}, exercise_images = {{"Push-up": "images/0662-I4hDWkc.jpg"}}.

GOAL-SPECIFIC WORKOUT RULES (MANDATORY):

🔴 FAT LOSS (when user mentions: lose weight, fat loss, slim down, lose Xkg):
- INTENSITY & CONDITIONING: You MUST structure the plan to maximize calorie burn. A fat loss plan is NOT just light walking and pushups.
- You MUST explicitly include at least 1-2 days of High-Intensity Interval Training (HIIT) or Metabolic Conditioning (MetCon) circuits.
- You MUST explicitly include Zone 2 Cardio (e.g., steady-state jogging, cycling) for aerobic fat burning.
- You MUST assign a Daily NEAT Goal (Non-Exercise Activity Thermogenesis), such as "10,000 steps per day" in the `description` or `summary`.
- Strength training should use shorter rest periods and supersets to elevate heart rate.

🟢 MUSCLE GAIN / BULKING (when user mentions: gain weight, muscle gain, hypertrophy):
- Focus entirely on Progressive Overload on heavy compound lifts.
- You MUST use a 5-day or 6-day cycle for Muscle Gain. You are strictly forbidden from shrinking it to 3 or 4 days.
- You MUST generate AT LEAST 20-30 physical exercises in the `workout` array across the cycle.
- Keep cardio minimal to avoid burning excess calories.
- Use traditional hypertrophy rep ranges (8-12 reps) and longer rest periods (90-120 seconds).

UNIVERSAL INJURY OVERRIDE (HIGHEST PRIORITY — OVERRIDES ALL OTHER RULES):
If the user reports ANY injury, pain, or medical condition (e.g., shoulder, knee, lower back, wrist), INJURY SAFETY WINS over ALL other rules. 
- Overrides "THE BIG 5" & "MUSCLE GAIN": You MUST skip heavy lifts (squats, bench press, deadlifts, overhead presses) if they load the injured area, regardless of the day's focus or hypertrophy goals.
- INJURY VOLUME CAP: Injured users need recovery capacity. You MUST cap volume to a maximum of 3-4 sets per exercise. NEVER prescribe 5-6 sets for an injured user. DO NOT add extra Rest Days or shrink the cycle length just because of an injury. The cycle length must remain standard (e.g. 5 or 6 days).

INJURY-AWARE EXERCISE SELECTION (100% DYNAMIC — BIOMECHANICS SAFETY PROTOCOL):
1. DEDUCE affected regions from the reported injury (e.g., "lower back pain" → lumbar spine, hips; "shoulder pain" → deltoids, rotator cuff; "wrist pain" → forearms, wrists).
2. ZERO STABILIZATION RULE: You are STRICTLY FORBIDDEN from including any exercise that requires the injured area to stabilize the body or bear load. 
   - Example (Lower Body Injury): NO standing presses, NO bent-over rows, NO burpees. Use chest-supported, seated, or lying variations ONLY.
   - Example (Upper Body/Wrist Injury): NO planks, NO push-ups, NO front squats. Use leg press, belt squats, or hands-free core exercises.
3. NO SNEAKY LEAKS: Do NOT include machine variations of forbidden movements (e.g., no Smith Machine Squats for knee pain, no Smith Bench for shoulder pain). Do NOT add "Full Body HIIT" finishers that use the injured joints.
4. MANDATORY DAILY REHAB WARMUP: You MUST start EVERY single active workout day with 1-2 specific Rehab or Mobility exercises targeted at recovering the injured joint (e.g., external rotations for shoulder pain, cat-cow for back pain) BEFORE the main lifts.
5. SAFE CARDIO: Use purely unloaded cardio for the injured joint (e.g., swimming or stationary bike for leg pain; walking or stationary bike for shoulder pain).
6. You MUST explicitly state in EVERY substituted/rehab exercise's `description`: "Adapted to protect/recover your [injured area]."

USER DATA:
Name: {full_name}
Age: {age} | Gender: {gender}
Weight: {weight_kg} kg | Height: {height_cm} cm
Activity Level: {activity_level}
TDEE & Calorie Targets:
  {tdee}
Goal: {goal}
Injuries/Medical: {injuries}
Medical Conditions: {medical}
Dietary Preference: {diet_preference}

Current Context: {summary}

{intelligence_context}
"""

        super().__init__(
            agent_name="Training Agent",
            rag_tool=TrainingRAGTool(),
            web_search_tool=WebSearchTool(),
            output_schema=TrainingAnalysis,
            system_prompt=system_prompt
        )
        # Override the base prompt to include the high-visibility recency safety mandate (Tier 2 Safety)
        from langchain_core.prompts import ChatPromptTemplate
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """CONVERSATION SUMMARY (for context):
{summary}

QUESTION: {query}

RETRIEVED DATA (Safe, filtered database items):
{context}

⚠️ CRITICAL SAFETY MANDATE:
The user has reported the following injuries/conditions: '{injuries}'.
You MUST absolutely drop and avoid any movement that loads or stabilizes the injured joint/area.
If the RETRIEVED DATA contains any unsafe exercises for this injury, you MUST DISCARD them and substitute them with 100% safe rehab/mobility variations.
Explain in every exercise's description: "Adapted to protect/recover your {injuries}."
""")
        ])

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

    def _expand_cycle(self, workout_list: list, rest_list: list, n_days: int) -> tuple:
        import math
        import re

        # Filter out LLM static repeats
        workout_list = [ex for ex in workout_list if "repeat" not in str(ex.get("day", "") if isinstance(ex, dict) else getattr(ex, "day", "")).lower()]
        rest_list = [r for r in rest_list if "repeat" not in str(r.get("day", "") if isinstance(r, dict) else getattr(r, "day", "")).lower()]

        # Combine all days to find cycle sequence
        all_days = set()
        for ex in workout_list:
            all_days.add(ex.get("day", "") if isinstance(ex, dict) else getattr(ex, "day", ""))
        for r in rest_list:
            all_days.add(r.get("day", "") if isinstance(r, dict) else getattr(r, "day", ""))
        
        # Sort by day number
        def extract_day_num(d_str):
            match = re.search(r'Day\s*(\d+)', d_str, re.IGNORECASE)
            return int(match.group(1)) if match else 999

        sorted_days = sorted(list(all_days), key=extract_day_num)
        cycle_length = len(sorted_days)

        if cycle_length == 0 or cycle_length >= n_days:
            return workout_list, rest_list

        expanded_workouts = []
        expanded_rests = []

        for target_day in range(1, n_days + 1):
            cycle_idx = (target_day - 1) % cycle_length
            cycle_num = math.ceil(target_day / cycle_length)
            orig_day_str = sorted_days[cycle_idx]
            
            day_type = orig_day_str
            match = re.search(r'Day \d+\s*-\s*(.*)', orig_day_str, re.IGNORECASE)
            if match:
                day_type = re.sub(r'\(Cycle.*?\)', '', match.group(1)).strip()
            new_day_str = f"Day {target_day} - {day_type} (Cycle {cycle_num})"

            # Workouts
            w_items = [ex for ex in workout_list if (ex.get("day", "") if isinstance(ex, dict) else getattr(ex, "day", "")) == orig_day_str]
            for ex in w_items:
                new_ex = ex.model_dump() if hasattr(ex, "model_dump") else dict(ex)
                new_ex["day"] = new_day_str
                
                if cycle_num > 1 and "rest" not in str(new_ex.get("name", "")).lower() and "stretch" not in str(new_ex.get("name", "")).lower():
                    sets_str = str(new_ex.get("sets", "3"))
                    sets_match = re.search(r'(\d+)', sets_str)
                    if sets_match:
                        s_val = int(sets_match.group(1))
                        if cycle_num % 2 == 0:
                            s_val += 1
                        new_ex["sets"] = sets_str.replace(sets_match.group(1), str(min(s_val, 6)))

                    reps_str = str(new_ex.get("reps", ""))
                    if "sec" not in reps_str.lower() and "min" not in reps_str.lower():
                        reps_match = re.findall(r'(\d+)', reps_str)
                        if len(reps_match) == 2:
                            r1, r2 = int(reps_match[0]), int(reps_match[1])
                            r1 += (cycle_num - 1) * 2
                            r2 += (cycle_num - 1) * 2
                            new_ex["reps"] = f"{r1}-{r2}"
                        elif len(reps_match) == 1:
                            r1 = int(reps_match[0])
                            r1 += (cycle_num - 1) * 2
                            new_ex["reps"] = f"{r1}"

                    desc = new_ex.get("description", "")
                    if "Progressive Overload" not in desc:
                        new_ex["description"] = desc + f"\n\n🔥 **Progressive Overload (Cycle {cycle_num})**: Try to increase the weight by 2.5kg or push for extra reps compared to Cycle 1."
                expanded_workouts.append(new_ex)

            # Rests
            r_items = [r for r in rest_list if (r.get("day", "") if isinstance(r, dict) else getattr(r, "day", "")) == orig_day_str]
            for r in r_items:
                new_r = r.model_dump() if hasattr(r, "model_dump") else dict(r)
                new_r["day"] = new_day_str
                expanded_rests.append(new_r)

        return expanded_workouts, expanded_rests

    async def run(self, state: AgentState) -> Dict[str, Any]:
        # Inject max_training_days into state so run_logic passes it to the prompt.
        safe_output_tokens = 15000  # Large token limit to allow 6-day splits without backend truncation
        tokens_per_exercise = 200   # Sets + reps + long descriptions (conservative buffer)
        exercises_per_day   = 6     # typical session volume max
        max_days = max(1, safe_output_tokens // (tokens_per_exercise * exercises_per_day))
        # Store so run_logic picks it up via prompt_vars injection below
        state = dict(state)
        state.setdefault("_extra_prompt_vars", {})["max_training_days"] = max_days
        
        query = state['messages'][-1].content
        n_days = await self._detect_n_days(query)
        
        result = await self.run_logic(state, specialist_key="training", topic="fitness workout exercise")
        
        if "specialist_results" in result and "training" in result["specialist_results"]:
            training_data = result["specialist_results"]["training"]
            
            # ── Tier 3 Safety: Run post-generation safety gate if injuries are present ──
            user_context = state.get("user_context", {}) or {}
            injuries_list = user_context.get("injuries", []) or []
            injuries = ", ".join(str(i) for i in injuries_list) if injuries_list else "None"
            has_injuries = injuries and injuries.lower() != "none" and injuries.strip() != ""
            
            if has_injuries and "workout" in training_data and training_data["workout"]:
                # 1. Run LLM Biomechanical Safety Gate
                audited_workout = await self._run_injury_safety_gate(training_data["workout"], injuries)
                
                # 2. Run Deterministic Python Backstop Filter to catch any remaining leaks (e.g. Seated Leg Extension)
                backstop_workout = []
                exclusions = {
                    "knee": ["squat", "lunge", "leg press", "leg extension", "quadriceps", "leg lift", "jump", "burpee", "groiner", "thruster", "box jump"],
                    "back": ["deadlift", "barbell row", "squat", "overhead press", "bent-over row", "good morning", "kettlebell swing", "thruster"],
                    "shoulder": ["overhead press", "military press", "bench press", "dip", "handstand", "pushup", "push-up", "shoulder press", "upright row"],
                    "wrist": ["push-up", "pushup", "plank", "handstand", "clean", "snatch", "bench press", "barbell wrist curl", "barbell curl"]
                }
                
                active_exclusions = set()
                for injury in injuries_list:
                    injury_lower = str(injury).lower()
                    for key, words in exclusions.items():
                        if key in injury_lower:
                            active_exclusions.update(words)
                
                for ex in audited_workout:
                    name = ex.get("name") if isinstance(ex, dict) else getattr(ex, "name", "")
                    name_lower = name.lower()
                    
                    is_unsafe = False
                    for word in active_exclusions:
                        if word in name_lower:
                            is_unsafe = True
                            break
                            
                    if is_unsafe:
                        logger.warning(f"🛡️ [Tier 3 Backstop] Caught unsafe exercise leak: '{name}'. Replacing with joint-safe rehab fallback.")
                        
                        # Select an appropriate fallback based on injury
                        if "knee" in injuries.lower():
                            fallback_name = "Glute Bridge"
                            fallback_target = ["Glutes", "Hamstrings"]
                            fallback_desc = f"Adapted to protect/recover your {injuries}. Lie on your back with knees bent and feet flat. Press through your heels to lift your hips."
                            fallback_benefit = "Unloads the knee completely while strengthening the posterior chain."
                        elif "back" in injuries.lower():
                            fallback_name = "Bird-Dog"
                            fallback_target = ["Core", "Lower Back"]
                            fallback_desc = f"Adapted to protect/recover your {injuries}. Keep a neutral spine, extend opposite arm and leg."
                            fallback_benefit = "Builds core stability without compressing the spine."
                        elif "shoulder" in injuries.lower():
                            fallback_name = "Rotator Cuff External Rotation"
                            fallback_target = ["Rotator Cuff", "Shoulders"]
                            fallback_desc = f"Adapted to protect/recover your {injuries}. Keep elbow at 90 degrees against your side, rotate forearm outward."
                            fallback_benefit = "Strengthens shoulder stabilizers safely."
                        else:
                            fallback_name = "Wall Sit (Light)"
                            fallback_target = ["Quadriceps"]
                            fallback_desc = f"Adapted to protect/recover your {injuries}. Perform a light-depth supported wall sit."
                            fallback_benefit = "Supported isometric holding."
                            
                        # Build safe replacement item
                        if isinstance(ex, dict):
                            new_ex = {
                                "day": ex.get("day", ""),
                                "name": fallback_name,
                                "target_muscle": fallback_target,
                                "benefit": fallback_benefit,
                                "description": fallback_desc,
                                "sets": ex.get("sets", "3"),
                                "reps": ex.get("reps", "12"),
                                "gif_path": "",
                                "image_path": ""
                            }
                        else:
                            new_ex = WorkoutExercise(
                                day=getattr(ex, "day", ""),
                                name=fallback_name,
                                target_muscle=fallback_target,
                                benefit=fallback_benefit,
                                description=fallback_desc,
                                sets=getattr(ex, "sets", "3"),
                                reps=getattr(ex, "reps", "12"),
                                gif_path="",
                                image_path=""
                            )
                        backstop_workout.append(new_ex)
                    else:
                        backstop_workout.append(ex)
                        
                training_data["workout"] = backstop_workout
                
                # ── Tier 3.5 Media Resolution Sync ──
                # Re-run your _validate_output function on the final, safety-audited workout list!
                # This will cleanly fuzzy-match the newly substituted exercises (e.g. Glute Bridge, Lying Leg Raises)
                # against the local media library and inject their correct GIF/image paths.
                training_data = self._validate_output(training_data, context="", state=state)

            workout_list = training_data.get("workout", [])
            rest_list = training_data.get("rest_days", [])
            
            if n_days > 1 and (workout_list or rest_list):
                e_workout, e_rest = self._expand_cycle(workout_list, rest_list, n_days)
                training_data["workout"] = e_workout
                training_data["rest_days"] = e_rest
                
        return result

    async def _run_injury_safety_gate(self, workout_list: List[Any], injuries: str) -> List[Dict]:
        """
        Tier 3 Safety: Run a gpt-4o-mini structured validation sweep over the generated workout list.
        Uses advanced anatomical, biomechanical, kinetic-chain and joint-stabilization reasoning.
        """
        if not workout_list or not injuries or injuries.lower() == "none":
            return workout_list

        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from pydantic import BaseModel, Field
        from app.core.config import settings
        import json

        # Prepare a custom safety schema container for structured output
        class WorkoutSafetyContainer(BaseModel):
            workout: List[WorkoutExercise]

        # Convert the existing workout list to a plain dictionary list
        workout_data = []
        for item in workout_list:
            if hasattr(item, "model_dump"):
                workout_data.append(item.model_dump())
            elif hasattr(item, "dict"):
                workout_data.append(item.dict())
            elif isinstance(item, dict):
                workout_data.append(item)
            else:
                workout_data.append(getattr(item, "__dict__", {}))

        logger.info(f"🛡️ [Tier 3 Safety Gate] Reviewing workout plan for injury: '{injuries}'")

        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=0, 
                api_key=settings.OPENAI_API_KEY
            ).with_structured_output(WorkoutSafetyContainer, method="function_calling")

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are the Chief Biomechanics & Safety Officer at 'Agentic AI Gym'.
Your sole mission is to review the generated workout plan and replace any biomechanically unsafe exercises to protect the user's reported injuries/conditions: '{injuries}'.

IMPORTANT: DO NOT USE SIMPLE KEYWORD MATCHING. You MUST perform deep anatomical, kinetic-chain, and joint-stabilization reasoning.

BIOMECHANICAL RISK ASSESSMENT RULES:
1. Joint Load & Kinetic Chain: Deduce which joints are affected by the injury (e.g., lumbar spine for back pain; patellofemoral joint for knee pain; glenohumeral joint for shoulder pain).
2. Action Check: For each exercise, analyze its execution. Does it load the affected joint, require it to absorb high impact, or force it to stabilize heavy weight?
   - Example: Seated Shoulder Press loads the lumbar spine compressively. Standing Barbell Row requires intense spinal stabilization. Broad Jumps place high impact on the knee.
3. Zero-Risk Substitution Mandate: If an exercise has even a minor risk of aggravation, you MUST replace it with a 100% safe, supported, or unloaded rehab alternative targeting the same muscle group if possible, or a restorative movement.
   - Knee Injury (e.g., knee pain, patella, meniscus, ACL): AVOID squats, lunges, leg press, extensions, jumps, thrusters. REPLACE WITH: Seated leg curls, glute bridges, clamshells, straight leg raises, seated calf raises.
   - Lower Back Injury (e.g., back pain, spinal strain): AVOID deadlifts, squats, standing rows, standing overhead press, bent-over rows. REPLACE WITH: Seated chest-supported cable/machine rows, bird-dog, glute bridges, lying leg curls, planks (if pain-free).
   - Shoulder Injury (e.g., shoulder pain, rotator cuff): AVOID overhead press, military press, flat bench press, dips. REPLACE WITH: Pec deck flys, internal/external rotator cuff rotations, face pulls, light chest-supported rows.
   - Wrist Injury (e.g., wrist pain, sprained wrist): AVOID planks/pushups on palms, straight-bar curls/presses. REPLACE WITH: Forearm planks, fist pushups, neutral-grip dumbbell work (with wrist braces/wraps), or pure legs/core machine exercises.

For each exercise in the provided workout list:
- Determine if it is unsafe/aggravating.
- If it is unsafe, replace it with a safe rehab or joint-friendly alternative.
- If you replace an exercise:
  a. Keep the original 'day' value.
  b. Set 'gif_path' and 'image_path' to empty strings ("").
  c. Append/prepend " [Adapted for {injuries} recovery]" to its 'description' and 'benefit' so the user knows it's customized.
- If it is already safe, do NOT modify it (keep all its original fields, including media paths, exactly as-is)."""),
                ("human", "Input Workout JSON:\n{workout_json}\n\nReturn the audited safe workout matching the structured schema.")
            ])

            chain = prompt | llm
            safety_output = await chain.ainvoke({
                "injuries": injuries,
                "workout_json": json.dumps(workout_data, ensure_ascii=False)
            })

            if safety_output and safety_output.workout:
                logger.info(f"🛡️ [Tier 3 Safety Gate] Clean sweep completed successfully. Returning validated exercises.")
                return safety_output.workout
            return workout_list

        except Exception as e:
            logger.error(f"❌ [Tier 3 Safety Gate] Validation Error: {e}. Failing open to avoid crashing.")
            return workout_list

    def _format_context(self, results: List[Dict]) -> str:
        """Convert DB results into meaningful exercise context."""
        if not results:
            return ""
        lines = []
        for r in results:
            name = r.get('name', 'Unknown')
            muscle = r.get('main_muscle', 'N/A')
            equip = r.get('equipment', 'N/A')
            prep = r.get('preparation', '')
            exe = r.get('execution', '')
            media = r.get('media', {})
            gif = media.get('gif')
            img = media.get('image')
            media_info = []
            if gif: media_info.append(f"GIF Available: {gif}")
            if img: media_info.append(f"Image Available: {img}")
            
            media_str = "\n  ".join(media_info) if media_info else "No media available"
            lines.append(f"• {name} (Muscle: {muscle}, Equipment: {equip})\n  {media_str}\n  Prep: {prep}\n  Execution: {exe}")
        return "\n\n".join(lines)
    
    def _validate_output(self, output: Dict[str, Any], context: str, state: Any = None) -> Dict[str, Any]:
        """
        1. Validates that media paths in exercise_gifs/exercise_images exist on disk.
        2. If LLM put paths in the text instead of JSON, extracts and rescues them.
        """
        import re
        from app.utils.gif_utils import media_matcher
        media_matcher._load_mappings()
        all_valid_gifs = set(media_matcher.gifs.values())
        all_valid_images = set(media_matcher.images.values())

        # ── STEP 1: Rescue paths that LLM dumped in the text instead of JSON ──
        final_answer = output.get("answer", "")
        # Filter out hallucinated paths from LLM's raw dict immediately
        gifs_dict = {k: v for k, v in output.get("exercise_gifs", {}).items() if v in all_valid_gifs}
        imgs_dict = {k: v for k, v in output.get("exercise_images", {}).items() if v in all_valid_images}
        
        # New approach: Extract from the structured 'workout' list directly!
        workout_list = output.get("workout", [])
        if isinstance(workout_list, list):
            for item in workout_list:
                # Handle both dict and Pydantic object
                if hasattr(item, "model_dump"):
                    item_dict = item.model_dump()
                elif hasattr(item, "dict"):
                    item_dict = item.dict()
                elif isinstance(item, dict):
                    item_dict = item
                else:
                    item_dict = getattr(item, "__dict__", {})

                name = item_dict.get("name")
                g_path = item_dict.get("gif_path")
                i_path = item_dict.get("image_path")
                
                if name and g_path and g_path in all_valid_gifs:
                    gifs_dict[name] = g_path
                if name and i_path and i_path in all_valid_images:
                    imgs_dict[name] = i_path

        # ── STEP 1.5: Auto-fill missing media using fuzzy MediaMatcher ──
        if isinstance(workout_list, list):
            for item in workout_list:
                if hasattr(item, "model_dump"):
                    item_dict = item.model_dump()
                elif hasattr(item, "dict"):
                    item_dict = item.dict()
                elif isinstance(item, dict):
                    item_dict = item
                else:
                    item_dict = getattr(item, "__dict__", {})

                name = item_dict.get("name")
                if not name:
                    continue
                # Only fill if AI left it empty
                if name not in gifs_dict or name not in imgs_dict:
                    matched = media_matcher.get_media(name)
                    if name not in gifs_dict and matched.get("gif"):
                        gifs_dict[name] = matched["gif"]
                    if name not in imgs_dict and matched.get("image"):
                        imgs_dict[name] = matched["image"]
        if not gifs_dict and final_answer:
            # Extract all gif/image paths from final_answer text
            gif_matches = re.findall(r'videos/[\w\-]+\.gif', final_answer)
            img_matches = re.findall(r'images/[\w\-]+\.jpg', final_answer)
            # Also extract markdown link labels: [ExerciseName GIF](videos/...)
            gif_labeled = re.findall(r'\[([^\]]+?)\s*GIF\]\((videos/[^)]+)\)', final_answer)
            img_labeled = re.findall(r'\[([^\]]+?)\s*Image\]\((images/[^)]+)\)', final_answer)

            for label, path in gif_labeled:
                if path in all_valid_gifs:
                    clean_label = label.strip()
                    if clean_label in gifs_dict:
                        clean_label = f"{clean_label} {len(gifs_dict)+1}"
                    gifs_dict[clean_label] = path
                    
            for label, path in img_labeled:
                if path in all_valid_images:
                    clean_label = label.strip()
                    if clean_label in imgs_dict:
                        clean_label = f"{clean_label} {len(imgs_dict)+1}"
                    imgs_dict[clean_label] = path

            # Fallback: unnamed paths
            if not gifs_dict:
                for i, path in enumerate(gif_matches):
                    if path in all_valid_gifs:
                        gifs_dict[f"Exercise {i+1}"] = path
            if not imgs_dict:
                for i, path in enumerate(img_matches):
                    if path in all_valid_images:
                        imgs_dict[f"Exercise {i+1}"] = path

        output["exercise_gifs"] = gifs_dict
        output["exercise_images"] = imgs_dict

        # ── STEP 2: Validate whatever is now in the dicts ──
        for field, valid_set in [("exercise_gifs", all_valid_gifs), ("exercise_images", all_valid_images)]:
            if field in output and isinstance(output[field], dict):
                validated = {}
                for name, path in output[field].items():
                    if path and path in valid_set:
                        validated[name] = path
                    else:
                        logger.warning(f"[Training Agent] Hallucination Blocked: '{path}' not in media files.")
                output[field] = validated

        # ── STEP 3: Inject back into workout list ──
        if isinstance(workout_list, list):
            for item in workout_list:
                if hasattr(item, "name"):
                    name = getattr(item, "name")
                    if name:
                        setattr(item, "gif_path", output["exercise_gifs"].get(name, ""))
                        setattr(item, "image_path", output["exercise_images"].get(name, ""))
                elif isinstance(item, dict):
                    name = item.get("name")
                    if name:
                        item["gif_path"] = output["exercise_gifs"].get(name, "")
                        item["image_path"] = output["exercise_images"].get(name, "")

        return output
