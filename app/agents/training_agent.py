from typing import Dict, Any, List
from app.core.state import AgentState
from app.tools.training_tools import TrainingRAGTool
from app.tools.web_search_tool import WebSearchTool
from app.agents.base import BaseRAGAgent
from pydantic import BaseModel, Field
from app.utils.logger import logger


class WorkoutExercise(BaseModel):
    name: str = Field(description="Name of the exercise.")
    target_muscle: List[str] = Field(description="List of target muscles.")
    benefit: str = Field(description="Benefit of this exercise.")
    description: str = Field(description="Step-by-step instructions on how to perform the exercise.")
    sets: str = Field(description="DYNAMIC: Recommended number of sets based on goal (e.g., '4', '3', '5').")
    reps: str = Field(description="DYNAMIC: Recommended reps or duration based on goal (e.g., '5-8' for strength, '15-20' for endurance, '60 seconds').")
    gif_path: str = Field(default="", description="Exact relative path to the GIF (e.g., videos/0044-XlZ4lAC.gif)")
    image_path: str = Field(default="", description="Exact relative path to the Image (e.g., images/0044-XlZ4lAC.jpg)")

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

INJURY-AWARE EXERCISE SELECTION (100% DYNAMIC):
- When the user reports ANY injury or medical condition, you MUST use your internal biomechanical knowledge to deduce which movements are unsafe.
- AUTOMATICALLY EXCLUDE or strictly modify any exercise that puts load, strain, or impact on the reported injured area.
- You MUST explicitly state in the `description` or `benefit` field how you modified the exercise for their specific injury (e.g., "Modified for your [Injury Name] by keeping the spine neutral").
- Always include a specific warning in the `tip` field addressing their exact injury and signs to stop.

Example JSON mapping: exercise_gifs = {{"Push-up": "videos/0662-I4hDWkc.gif"}}, exercise_images = {{"Push-up": "images/0662-I4hDWkc.jpg"}}.

USER DATA:
Goal: {goal}
Injuries/Medical: {injuries}
Dietary Preference: {diet_preference}
Current Context: {summary}
"""

        super().__init__(
            agent_name="Training Agent",
            rag_tool=TrainingRAGTool(),
            web_search_tool=WebSearchTool(),
            output_schema=TrainingAnalysis,
            system_prompt=system_prompt
        )

    async def run(self, state: AgentState) -> Dict[str, Any]:
        return await self.run_logic(state, specialist_key="training", topic="fitness workout exercise")

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
