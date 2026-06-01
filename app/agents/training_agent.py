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
2. VETTED EXERCISE SELECTION: You MUST select exercises ONLY from the valid exercise_ids provided in the schema. You are strictly forbidden from generating exercises outside this pool. If the pool is insufficient, rely on the refusal mechanism.
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
- DYNAMIC SPLIT SELECTION: {dynamic_split_rules}
- DYNAMIC REST DAYS (CRITICAL): If a day is meant for rest or active recovery, you MUST put it entirely inside the `rest_days` array. DO NOT create a fake 'Rest' exercise inside the `workout` array. The `workout` array must remain 100% clean, containing only active physical exercises.
- DYNAMIC DAILY VOLUME & VARIETY: DO NOT just divide the retrieved exercises across the days, and DO NOT repeat the exact same exercises on different days. MINIMUM DAILY VOLUME RULE (CRITICAL): You MUST generate a minimum of 3 exercises for EVERY single training day (preferably 4-6 depending on activity level). It is completely unacceptable and a failure of the task to output a training day with only 1 or 2 exercises. If the database provides too few exercises, you MUST use your expert knowledge to dynamically invent and add standard exercises (e.g., Push-ups, Squats, Planks) to meet the minimum threshold.
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
The RETRIEVED DATA has been medically pre-vetted and contains ONLY safe exercises. You must ONLY use exercises from the RETRIEVED DATA.
Explain in every exercise's coaching_note: "Adapted to protect/recover your {injuries}."
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

    def _expand_cycle(self, workout_list: list, rest_list: list, n_days: int, max_days: int = 12) -> tuple:
        import math
        import re
        
        # Filter out LLM static repeats
        workout_list = [ex for ex in workout_list if "repeat" not in str(ex.get("day", "") if isinstance(ex, dict) else getattr(ex, "day", "")).lower()]
        rest_list = [r for r in rest_list if "repeat" not in str(r.get("day", "") if isinstance(r, dict) else getattr(r, "day", "")).lower()]

        expanded_workouts = []
        expanded_rests = []
        
        all_days = set()
        for ex in workout_list:
            d = ex.get("day", "") if isinstance(ex, dict) else getattr(ex, "day", "")
            if d: all_days.add(d)
        for r in rest_list:
            d = r.get("day", "") if isinstance(r, dict) else getattr(r, "day", "")
            if d: all_days.add(d)

        def extract_day_num(day_str: str):
            match = re.search(r'Day\s*(\d+)', day_str, re.IGNORECASE)
            return int(match.group(1)) if match else 999

        sorted_days = sorted(list(all_days), key=extract_day_num)
        cycle_length = len(sorted_days)

        if cycle_length == 0 or cycle_length >= n_days:
            return workout_list, rest_list, cycle_length

        is_short_plan = n_days <= max_days

        for target_day in range(1, n_days + 1):
            cycle_idx = (target_day - 1) % cycle_length
            cycle_num = math.ceil(target_day / cycle_length)
            orig_day_str = sorted_days[cycle_idx]
            
            day_type = orig_day_str
            match = re.search(r'Day \d+\s*-\s*(.*)', orig_day_str, re.IGNORECASE)
            if match:
                day_type = re.sub(r'\(Cycle.*?\)', '', match.group(1)).strip()
            
            if is_short_plan:
                new_day_str = f"Day {target_day} - {day_type}"
            else:
                new_day_str = f"Day {target_day} - {day_type} (Cycle {cycle_num})"

            # Workouts
            w_items = [ex for ex in workout_list if (ex.get("day", "") if isinstance(ex, dict) else getattr(ex, "day", "")) == orig_day_str]
            for ex in w_items:
                new_ex = ex.model_dump() if hasattr(ex, "model_dump") else dict(ex)
                new_ex["day"] = new_day_str
                
                if not is_short_plan and cycle_num > 1 and "rest" not in str(new_ex.get("name", "")).lower() and "stretch" not in str(new_ex.get("name", "")).lower():
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

        return expanded_workouts, expanded_rests, cycle_length

    async def run(self, state: AgentState) -> Dict[str, Any]:
        import json
        from app.safety.intake import translate_injury_to_constraint
        from app.safety.filter import filter_with_audit
        from app.safety.refusal import maybe_refuse, classify_goal, SegmentedTags
        from app.safety.dynamic_schema import build_dynamic_training_analysis
        from langchain_openai import ChatOpenAI
        from app.core.config import settings
        
        user_context = state.get("user_context", {}) or {}
        injuries_list = user_context.get("injuries", []) or []
        injuries = ", ".join(str(i) for i in injuries_list) if injuries_list else "None"
        
        # Load tagged safe pool (Phase 10: will load all 2,840 exercises, currently 41 prototypes)
        TAGS_PATH = r"d:\AI\IMGProjects\ai-fitness-app\ai-fitness-app\app\safety\tags_lower_body.json"
        SEGMENTS_PATH = r"d:\AI\IMGProjects\ai-fitness-app\ai-fitness-app\app\safety\segments_lower_body.json"
        with open(TAGS_PATH, encoding="utf-8") as f:
            raw_tags = {e["exercise_id"]: e for e in json.load(f)}
        with open(SEGMENTS_PATH, encoding="utf-8") as f:
            raw_segs = {e["exercise_id"]: e for e in json.load(f)}
        
        tags = []
        for eid, tag_data in raw_tags.items():
            if eid in raw_segs:
                tags.append(SegmentedTags(**{**tag_data, **raw_segs[eid]}))
                
        # Tier 1 Safety: Deterministic Filtering
        constraint = translate_injury_to_constraint(injuries)
        safe_pool, _ = filter_with_audit(tags, constraint)
        
        # Segment Coverage Refusal Mechanism
        query = state['messages'][-1].content
        goal_key = classify_goal(query)
        decision = maybe_refuse(safe_pool, goal_key, min_per_segment=2)
        
        if decision.should_refuse:
            return {
                "specialist_results": {
                    "training": {
                        "answer": decision.refusal_message,
                        "status": "success",
                        "is_accurate": True,
                        "needs_web_search": False,
                        "sub_queries": [],
                        "final_answer": decision.refusal_message,
                        "summary": decision.refusal_message,
                        "workout": [],
                        "rest_days": [],
                        "tip": "Consult a physiotherapist.",
                        "exercise_gifs": {},
                        "exercise_images": {}
                    }
                }
            }
            
        safe_pool_ids = [ex.exercise_id for ex in safe_pool]

        # Dynamically rebuild structured output schema to strictly enforce safe_pool
        DynamicAnalysisSchema = build_dynamic_training_analysis(safe_pool)
        original_llm = self.llm
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.3, 
            api_key=settings.OPENAI_API_KEY,
            max_retries=3
        ).with_structured_output(DynamicAnalysisSchema, method="function_calling")

        safe_output_tokens = 15000
        tokens_per_exercise = 200
        exercises_per_day   = 6
        max_days = max(1, safe_output_tokens // (tokens_per_exercise * exercises_per_day))
        state = dict(state)
        state.setdefault("_extra_prompt_vars", {})["max_training_days"] = max_days
        state.setdefault("_extra_prompt_vars", {})["allowed_ids"] = safe_pool_ids
        
        n_days = await self._detect_n_days(query)
        if n_days == 1:
            split_rules = "• N = 1 (Daily): Generate a single optimized session. Leave the `day` field empty."
        elif n_days <= max_days:
            split_rules = f"• N = {n_days} (Short Plan): Generate exactly {n_days} unique days. Apply a logical split (e.g., Push/Pull/Legs). Try to minimize repeating exercises, but you MAY repeat them across different days if necessary to fulfill a highly specific user focus. Include Rest/Active Recovery days if appropriate. You MUST populate the `day` field for every exercise (e.g., \"Day 1 - Push\", \"Day 3 - Rest\")."
        else:
            split_rules = f"• N = {n_days} (Long-term Plan): Generate a microcycle and explain repeats."

        state.setdefault("_extra_prompt_vars", {})["dynamic_split_rules"] = split_rules
        
        try:
            # Execute Adaptive RAG with strictly constrained safe_pool vocabulary
            result = await self.run_logic(state, specialist_key="training", topic="fitness workout exercise")
        except Exception as e:
            # If the LLM hallucinates an exercise outside the safe_pool (because it's desperate to fulfill 
            # a specific request like "calves"), Pydantic will throw a ValidationError.
            if "validation error" in str(e).lower() or "literal_error" in str(e).lower():
                import logging
                logging.getLogger("fit_bot").error(f"Validation Error caught: {e}")
                user_context = state.get("user_context", {})
                has_injuries = bool(user_context.get("injuries") or user_context.get("medical_conditions"))
                reason = "within your medical constraints" if has_injuries else "with the currently available database exercises"
                
                msg = f"I could not find enough unique exercises matching your highly specific request {reason}."
                return {
                    "specialist_results": {
                        "training": {
                            "answer": msg,
                            "status": "success",
                            "is_accurate": True,
                            "needs_web_search": False,
                            "sub_queries": [],
                            "final_answer": msg,
                            "summary": msg,
                            "workout": [],
                            "rest_days": [],
                            "tip": "Try a broader muscle group (e.g., Leg Day instead of Calves) so I can build a balanced plan.",
                            "exercise_gifs": {},
                            "exercise_images": {}
                        }
                    }
                }
            raise e
        finally:
            self.llm = original_llm
            
        if "specialist_results" in result and "training" in result["specialist_results"]:
            training_data = result["specialist_results"]["training"]
            
            # Hydrate WorkoutItem back to WorkoutExercise for the frontend
            hydrated_workout = []
            for item in training_data.get("workout", []):
                if hasattr(item, "model_dump"):
                    item_dict = item.model_dump()
                elif hasattr(item, "dict"):
                    item_dict = item.dict()
                elif isinstance(item, dict):
                    item_dict = item
                else:
                    item_dict = getattr(item, "__dict__", {})

                eid = item_dict.get("exercise_id")
                tag_info = raw_tags.get(eid, {})
                ex_name = tag_info.get("name", eid)
                
                from app.utils.gif_utils import media_matcher
                matched = media_matcher.get_media(ex_name)
                instructions = matched.get("instructions")
                if not instructions:
                    instructions = item_dict.get("coaching_note", "Follow standard form securely.")

                hydrated_workout.append({
                    "day": item_dict.get("day", ""),
                    "name": ex_name,
                    "target_muscle": tag_info.get("primary_joints_involved", []),
                    "benefit": item_dict.get("coaching_note", "Adapted safely."),
                    "description": instructions,
                    "sets": str(item_dict.get("sets", "3")),
                    "reps": str(item_dict.get("reps", "")),
                    "gif_path": "",
                    "image_path": ""
                })
            
            training_data["workout"] = hydrated_workout
            training_data = self._validate_output(training_data, context="", state=state)
            
            workout_list = training_data.get("workout", [])
            rest_list = training_data.get("rest_days", [])
            
            # Post-generation fallback: If the LLM generated no exercises, the safe pool 
            # couldn't satisfy the highly specific user query (e.g. asking for calves when ankle is blocked)
            # A workout with only rest days is not a workout.
            if not workout_list:
                user_context = state.get("user_context", {})
                has_injuries = bool(user_context.get("injuries") or user_context.get("medical_conditions"))
                reason = "within your medical constraints" if has_injuries else "with the currently available database exercises"
                
                refusal_msg = f"I could not find enough unique active exercises matching your highly specific request {reason}."
                result["specialist_results"]["training"]["summary"] = refusal_msg
                result["specialist_results"]["training"]["tip"] = "Try a broader muscle group (e.g., Leg Day instead of Calves) so I can build a balanced plan."
                return result

            if n_days > 1 and (workout_list or rest_list):
                e_workout, e_rest, cycle_length = self._expand_cycle(workout_list, rest_list, n_days, max_days=max_days)
                training_data["workout"] = e_workout
                training_data["rest_days"] = e_rest
        return result

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
