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
4. MEDIA: If the retrieved data includes a 'GIF Available' path for an exercise you recommend, you MUST include that path in the 'exercise_gifs' dictionary. Use the EXACT name of the exercise as it appears in the 'RETRIEVED DATA' (e.g., 'Push-up: Incline') as the dictionary key.

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
            gif = r.get('gif_path')
            gif_info = f"\n  GIF Available: {gif}" if gif else "\n  GIF: Not available"
            lines.append(f"• {name} (Muscle: {muscle}, Equipment: {equip}){gif_info}\n  Prep: {prep}\n  Execution: {exe}")
        return "\n\n".join(lines)
