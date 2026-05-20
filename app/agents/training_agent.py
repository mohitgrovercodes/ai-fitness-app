from typing import Dict, Any, List,Optional
from app.core.state import AgentState
from app.tools.training_tools import TrainingRAGTool
from app.tools.web_search_tool import WebSearchTool
from app.agents.base import BaseRAGAgent
from pydantic import BaseModel, Field
from app.utils.logger import logger


class WorkoutExercise(BaseModel):
    day: Optional[str] = Field(default="", description="Day label for multi-day splits (e.g., 'Day 1 - Upper Body'). Leave empty for single-day plans.")
    name: Optional[str] = Field(default="", description="Name of the exercise.")
    target_muscle: List[str] = Field(default_factory=list, description="List of target muscles.")
    benefit: Optional[str] = Field(default="", description="Benefit of this exercise.")
    description: Optional[str] = Field(default="", description="Step-by-step instructions on how to perform the exercise.")
    sets: Optional[str] = Field(default="", description="DYNAMIC: Recommended number of sets based on goal (e.g., '4', '3', '5').")
    reps: Optional[str] = Field(default="", description="DYNAMIC: Recommended reps or duration based on goal (e.g., '5-8' for strength, '15-20' for endurance, '60 seconds').")
    gif_path: Optional[str] = Field(default="", description="Exact relative path to the GIF (e.g., videos/0044-XlZ4lAC.gif)")
    image_path: Optional[str] = Field(default="", description="Exact relative path to the Image (e.g., images/0044-XlZ4lAC.jpg)")

class TrainingAnalysis(BaseModel):
    is_accurate: bool = Field(description="Are the retrieved exercises relevant and safe?")
    needs_web_search: bool = Field(description="True if exercise is unknown or local DB lacks info.")
    sub_queries: List[str] = Field(default=[], description="Alternative search terms for routines.")
    final_answer: str = Field(description="Full text markdown response for chat users.")
    summary: str = Field(default="", description="Brief introduction/summary of the workout.")
    workout: List[WorkoutExercise] = Field(default=[], description="List of structured exercises.")
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
2. HYBRID DATABASE & EXPERT USAGE: You MUST prioritize building the workout using the exercises provided in the retrieved data. However, if the retrieved exercises are NOT SUFFICIENT for the user's specific goal (e.g., losing 8kg requires significant cardio/full-body movements, but DB only has push-ups), you are AUTHORIZED to use your expert knowledge to inject necessary exercises (like Brisk Walking, Jogging, Cycling). For expert-added exercises, leave `gif_path` and `image_path` completely empty (""). Set `is_accurate` to true.
3. COMPREHENSIVE WORKOUT PLAN: A proper workout plan MUST cover a full routine based on the user's goal. It should dynamically include a mix of necessary components (e.g., warm-up/cardio, main strength/core exercises, and cool-down). Do NOT just provide 1 or 2 isolated exercises. If the database only gives you 1 exercise, you MUST use your expert knowledge to dynamically build out the rest of a complete, balanced routine that realistically addresses the user's goal.
4. ADAPTABILITY & DYNAMIC PROGRAMMING: Adapt advice based on injuries and dietary preferences (e.g., if a user is Vegan or Keto, suggest appropriate intensity or recovery based on their likely macro intake if relevant). Critically, you MUST dynamically calculate sets and reps for EACH exercise based on the user's goal (e.g., Hypertrophy = 8-12 reps, Strength = 3-5 reps, Endurance = 15+ reps, Planks = 30-60s). DO NOT give a static 3 sets of 10-12 reps for everything.
5. STRUCTURED JSON FIELDS: You MUST populate the `summary`, `workout` (list of exercises), and `tip` fields with structured data for interactive UI display.
6. MEDIA PATHS: You MUST include the correct `gif_path` and `image_path` directly inside each exercise object in the `workout` list.
7. CLEAN TEXT RESPONSE: The `final_answer` string MUST ONLY contain a polite greeting and a brief 1-2 sentence intro. DO NOT list the exercises, sets, reps, or media paths inside `final_answer`. Put the data ONLY in the structured JSON fields.
8. NO SYSTEM TALK: NEVER use phrases like "based on the retrieved data", "the database doesn't have", or "the retrieved exercises". Speak directly as an expert coach.

9. ANTI-LAZINESS RULE (CRITICAL): The `workout` list MUST NEVER BE EMPTY. You MUST generate the complete workout routine with actual exercises using your expert knowledge, even if the retrieved data is empty or generic web results.

MULTI-DAY & DURATION SPLIT RULES (100% DYNAMIC):
- Detect exactly what duration (N days) the user is asking for from their message (e.g., "today" = 1, "4 days" = 4, "a week" = 7, "a month" = 30).
- DYNAMIC SPLIT SELECTION: You MUST dynamically assign an optimal, professional workout split based on the requested days:
  • N = 1 (Daily): Generate a single optimized session. Leave the `day` field empty.
  • N = 2 to {max_training_days} (Short Plans): Generate exactly N unique days. Apply a logical split (e.g., N=3 is Push/Pull/Legs; N=4 is Upper/Lower; N=5 is Bro Split). DO NOT repeat the same exercises across all days. Include Rest/Active Recovery days if appropriate. You MUST populate the `day` field for every exercise (e.g., "Day 1 - Push", "Day 3 - Rest").
  • N > {max_training_days} (Long-term Plans): This is real gym programming. DO NOT generate N different workout days.
    STEP 1 — DETERMINE CYCLE LENGTH: Use your fitness expertise to select the optimal split cycle for the user's goal.
      Examples: Muscle Gain → PPL (6-day cycle) or Upper/Lower (4-day cycle)
               Fat Loss → Full Body (3-day cycle) or Circuit (4-day cycle)
               General Fitness → 3 or 4-day full body split
      The cycle length is YOUR decision based on fitness science — it is NOT fixed.
    STEP 2 — GENERATE THE CYCLE: Generate exactly that many unique days as a REPEATING MICROCYCLE.
    CRITICAL HARD LIMIT: YOU MUST STOP GENERATING AFTER THE BASE CYCLE. DO NOT generate Day 7, Day 8, Day 9, etc., if your cycle is only 6 days. DO NOT output exercises named "Repeat Day 1". The backend Python engine will handle the mathematical expansion and progressive overload automatically.
    STEP 3 — PROGRESSION PLAN: In `summary`, explain:
      - How many times to repeat this cycle to complete N days (e.g. ceil(N / cycle_length))
      - Week-by-week progressive overload: Week 1 = baseline, Week 2 = +2 reps, Week 3 = +weight, etc.
    STEP 4 — `tip`: State exactly: "Repeat this X-day cycle Y times over N days. Increase [metric] each week."
    You MUST populate the `day` field for every exercise (e.g., "Day 1 - Push (Cycle Day 1)").
- DYNAMIC REST DAYS: You are authorized to create Rest Days where the exercise name is "Rest" or "Light Stretching".
- DYNAMIC DAILY VOLUME: DO NOT just divide the retrieved exercises across the days. A single day MUST be a complete workout session on its own. Dynamically decide the number of exercises per day based on the split type (e.g., an intense Leg Day might need 5-7 exercises, while an Active Recovery day might only need 2-3 stretches). If the database didn't provide enough exercises for a complete daily session, use your expert knowledge to inject the missing exercises.
- ANATOMICAL BALANCE MANDATE: Every active workout day generated MUST be anatomically balanced. For example, if a day is "Full Body Strength", you MUST dynamically include at least one chest exercise, one back exercise, one shoulder exercise, and one core/mobility exercise. Do not spam a single region (like legs or lunges) while ignoring upper body.

INJURY-AWARE EXERCISE SELECTION (100% DYNAMIC — BIOMECHANICS SAFETY PROTOCOL):
- When the user reports ANY injury, pain, or medical condition, you MUST dynamically deduce the affected muscles, joints, and skeletal regions.
- CRITICAL: You are STRICTLY FORBIDDEN from including any exercise that loads, stresses, or impacts the deduced injured/painful areas.
  • If knee, leg, ankle, or hip pain is mentioned: You MUST dynamically ban all lunges, squats, leg presses, thigh-focused movements, and high-impact leg exercises. Forcefully replace them with seated exercises, upper body work, swimming, cycling, or active recovery mobility stretching.
  • If shoulder, elbow, or wrist pain/injury is mentioned: Dynamically ban overhead presses, heavy pushups, or loading wrist positions. Substitute with safe back/chest/core work.
- You MUST explicitly state in the `description` or `benefit` field how you adapted the selection to protect the injury (e.g., "Substituted with Seated Press to protect your injured joint").
- Always include a specific injury-safe warning in the `tip` field.

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

    def _expand_with_progression(self, workout_list: list, n_days: int) -> list:
        import math
        import re
        
        # 1. Group exercises by original day
        day_groups = []
        current_day_str = None
        current_group = []
        
        for ex in workout_list:
            ex_day = ex.get("day", "") if isinstance(ex, dict) else getattr(ex, "day", "")
            
            # FILTER OUT LLM HALLUCINATED STATIC REPEATS!
            if "repeat" in str(ex_day).lower() or "repeat" in str(ex.get("name", "") if isinstance(ex, dict) else getattr(ex, "name", "")).lower():
                continue
                
            if ex_day != current_day_str:
                if current_group:
                    day_groups.append(current_group)
                current_day_str = ex_day
                current_group = []
            current_group.append(ex)
            
        if current_group:
            day_groups.append(current_group)
            
        cycle_length = len(day_groups)
        if cycle_length == 0 or cycle_length >= n_days:
            return workout_list # No expansion needed
            
        expanded = []
        for target_day in range(1, n_days + 1):
            cycle_idx = (target_day - 1) % cycle_length
            cycle_num = math.ceil(target_day / cycle_length)
            original_group = day_groups[cycle_idx]
            
            for ex in original_group:
                new_ex = ex.model_dump() if hasattr(ex, "model_dump") else dict(ex)
                
                orig_day = new_ex.get("day", "")
                day_type = orig_day
                match = re.search(r'Day \d+\s*-\s*(.*)', orig_day, re.IGNORECASE)
                if match:
                    # Clean out any existing (Cycle X) string from LLM output
                    day_type = re.sub(r'\(Cycle.*?\)', '', match.group(1)).strip()
                    
                new_ex["day"] = f"Day {target_day} - {day_type} (Cycle {cycle_num})"
                
                # Apply Progressive Overload
                if cycle_num > 1 and "rest" not in str(new_ex.get("name", "")).lower() and "stretch" not in str(new_ex.get("name", "")).lower():
                    # Modify sets slightly
                    sets_str = str(new_ex.get("sets", "3"))
                    sets_match = re.search(r'(\d+)', sets_str)
                    if sets_match:
                        s_val = int(sets_match.group(1))
                        if cycle_num % 2 == 0:
                            s_val += 1
                        new_ex["sets"] = sets_str.replace(sets_match.group(1), str(min(s_val, 6)))
                        
                    # Modify reps slightly
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
                            
                    # Add overload tip
                    desc = new_ex.get("description", "")
                    if "Progressive Overload" not in desc:
                        new_ex["description"] = desc + f"\n\n🔥 **Progressive Overload (Cycle {cycle_num})**: Try to increase the weight by 2.5kg or push for extra reps compared to Cycle 1."
                        
                expanded.append(new_ex)
                
        return expanded

    async def run(self, state: AgentState) -> Dict[str, Any]:
        # Inject max_training_days into state so run_logic passes it to the prompt.
        safe_output_tokens = 3680   # gpt-4o-mini with 10% safety margin
        tokens_per_exercise = 150   # sets + reps + description + benefit (safer buffer)
        exercises_per_day   = 6     # typical session volume max
        max_days = max(1, safe_output_tokens // (tokens_per_exercise * exercises_per_day))
        # Store so run_logic picks it up via prompt_vars injection below
        state = dict(state)
        state.setdefault("_extra_prompt_vars", {})["max_training_days"] = max_days
        
        query = state['messages'][-1].content
        n_days = await self._detect_n_days(query)
        
        result = await self.run_logic(state, specialist_key="training", topic="fitness workout exercise")
        
        if n_days > 1 and "specialist_results" in result and "training" in result["specialist_results"]:
            training_data = result["specialist_results"]["training"]
            workout_list = training_data.get("workout", [])
            
            if workout_list:
                expanded_workout = self._expand_with_progression(workout_list, n_days)
                training_data["workout"] = expanded_workout
                
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
        gifs_dict = output.get("exercise_gifs", {})
        imgs_dict = output.get("exercise_images", {})
        
        # New approach: Extract from the structured 'workout' list directly!
        workout_list = output.get("workout", [])
        if isinstance(workout_list, list):
            for item in workout_list:
                if isinstance(item, dict):
                    name = item.get("name")
                    g_path = item.get("gif_path")
                    i_path = item.get("image_path")
                    if name and g_path and g_path in all_valid_gifs:
                        gifs_dict[name] = g_path
                    if name and i_path and i_path in all_valid_images:
                        imgs_dict[name] = i_path

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
        return output
