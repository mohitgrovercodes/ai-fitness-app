from typing import Dict, Any, List
from app.core.state import AgentState
from app.tools.training_tools import TrainingRAGTool
from app.tools.web_search_tool import WebSearchTool
from app.agents.base import BaseRAGAgent
from pydantic import BaseModel, Field
from app.utils.logger import logger


class TrainingAnalysis(BaseModel):
    is_accurate: bool = Field(description="Are the retrieved exercises relevant and safe?")
    needs_web_search: bool = Field(description="True if exercise is unknown or local DB lacks info.")
    sub_queries: List[str] = Field(default=[], description="Alternative search terms for routines.")
    final_answer: str = Field(description="The professional coaching response detailing the workout.")
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
2. ACCURACY (CRAG Evaluator): If the retrieved exercises DO NOT match the user's specific request, set 'is_accurate' to false.
3. ADAPTABILITY: If the user has injuries, adapt the advice or suggest alternatives from the data.
4. MANDATORY MEDIA DICTIONARY: You MUST use the `exercise_gifs` and `exercise_images` JSON fields to store the media paths.
5. NO MEDIA IN TEXT: CRITICAL! NEVER print the words "GIF:" or "Image:" or include any file paths (like .gif or .jpg) inside the `final_answer` string. Put them ONLY in the JSON dictionary.
6. NO SYSTEM TALK: NEVER use phrases like "based on the retrieved data", "the database doesn't have", or "the retrieved exercises". Speak directly as an expert coach.

Example JSON mapping: exercise_gifs = {{"Push-up": "videos/0662-I4hDWkc.gif"}}, exercise_images = {{"Push-up": "images/0662-I4hDWkc.jpg"}}.

USER DATA:
Goal: {goal}
Injuries/Medical: {injuries}"""

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
    
    def _validate_output(self, output: Dict[str, Any], context: str) -> Dict[str, Any]:
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

        if not gifs_dict and final_answer:
            # Extract all gif/image paths from final_answer text
            gif_matches = re.findall(r'videos/[\w\-]+\.gif', final_answer)
            img_matches = re.findall(r'images/[\w\-]+\.jpg', final_answer)
            # Also extract markdown link labels: [ExerciseName GIF](videos/...)
            gif_labeled = re.findall(r'\[([^\]]+?)\s*GIF\]\((videos/[^)]+)\)', final_answer)
            img_labeled = re.findall(r'\[([^\]]+?)\s*Image\]\((images/[^)]+)\)', final_answer)

            for label, path in gif_labeled:
                if path in all_valid_gifs:
                    gifs_dict[label.strip()] = path
            for label, path in img_labeled:
                if path in all_valid_images:
                    imgs_dict[label.strip()] = path

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
