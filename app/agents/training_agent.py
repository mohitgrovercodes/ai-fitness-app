from typing import Dict, Any, List
from app.core.state import AgentState
from app.tools.training_tools import TrainingRAGTool
from app.tools.web_search_tool import WebSearchTool
from app.agents.base import BaseRAGAgent
from pydantic import BaseModel, Field


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
4. MEDIA: If the retrieved data includes a 'GIF Available' path, put it in 'exercise_gifs'. If it includes an 'Image Available' path, put it in 'exercise_images'. Use the EXACT name from the 'RETRIEVED DATA' as the key. 
Example: exercise_gifs = {{"Push-up": "videos/0662-I4hDWkc.gif"}}, exercise_images = {{"Push-up": "images/0662-I4hDWkc.jpg"}}.
DO NOT include media links in the 'final_answer' text.

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
        Ensures that media paths returned by the LLM were actually present in the context.
        Prevents hallucination of GIF/Image paths.
        """
        for field in ["exercise_gifs", "exercise_images"]:
            if field in output and isinstance(output[field], dict):
                validated_media = {}
                for name, path in output[field].items():
                    # Only keep the path if it actually appears in the retrieved context string
                    if path in context:
                        validated_media[name] = path
                    else:
                        logger.warning(f"⚠️ [Training Agent] Hallucination Blocked: '{path}' was not in context.")
                
                output[field] = validated_media
        return output
