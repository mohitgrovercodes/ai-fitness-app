from typing import Dict, Any
from app.agents.base import BaseRAGAgent
from app.core.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.tools.web_search_tool import WebSearchTool
from app.utils.logger import logger

class DomainAgent(BaseRAGAgent):
    """
    Specialist Agent for General Fitness Knowledge.
    Handles topics like anatomy, physiology, hypertrophy science, BMR, etc.
    """
    def __init__(self):
        from app.core.config import settings
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0, 
            api_key=settings.OPENAI_API_KEY
        )
        self.web_search = WebSearchTool()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Senior Exercise Scientist at 'Agentic AI Gym'.
Your role is to provide scientifically accurate, evidence-based answers to general fitness and physiology questions.

TOPICS YOU HANDLE:
- Muscle anatomy and function.
- Exercise science (Hypertrophy, Strength, Endurance).
- Metabolic calculations (BMR, TDEE).
- Recovery, sleep, and supplementation science.
- Biomechanics and proper movement patterns.

STRICT GUIDELINES:
1. If the user asks for a specific workout routine, defer to the Training Agent.
2. If the user asks about specific foods or calories, defer to the Nutrition Agent.
3. Use the provided SEARCH CONTEXT if available to ensure your advice is up-to-date with current science.
4. Keep explanations clear, professional, and actionable.
5. DYNAMIC VERBOSITY: If the user asks a simple, factual question about their profile (e.g., "what is my weight?"), answer concisely in 1-2 sentences. DO NOT generate unsolicited coaching advice or extra paragraphs unless they specifically ask for a plan or advice.

SEARCH CONTEXT:
{context}

USER PROFILE:
Name: {full_name}
Age: {age} | Gender: {gender}
Weight: {weight_kg} kg | Height: {height_cm} cm
Activity Level: {activity_level}
TDEE & Calorie Targets:
  {tdee_str}
Goal: {goal}
Dietary Preference: {diet_pref}
Injuries/Medical: {injuries}
Medical Conditions: {medical}

USER QUERY: {user_input}"""),
            ("human", "{user_input}")
        ])

    async def run(self, state: AgentState) -> Dict[str, Any]:
        user_input = state['messages'][-1].content
        user_context = state.get("user_context", {})
        full_name      = user_context.get("full_name", "User")
        goal           = user_context.get("goal", "General Fitness")
        diet_pref      = user_context.get("diet_preference", "None")
        weight_kg      = user_context.get("weight_kg", "Unknown")
        height_cm      = user_context.get("height_cm", "Unknown")
        age            = user_context.get("age", "Unknown")
        gender         = user_context.get("gender", "Unknown")
        activity_level = user_context.get("activity_level", "Unknown")
        injuries       = ", ".join(user_context.get("injuries", [])) if user_context.get("injuries") else "None"
        medical        = ", ".join(user_context.get("medical_conditions", [])) if user_context.get("medical_conditions") else "None"

        cal_loss        = user_context.get("cal_loss", 0)
        cal_maintenance = user_context.get("cal_maintenance", 0)
        cal_gain        = user_context.get("cal_gain", 0)
        
        tdee_str = "Unknown — profile data incomplete."
        if cal_maintenance:
            tdee_str = (
                f"TDEE {cal_maintenance} kcal/day\n"
                f"  Weight-loss target  : {cal_loss} kcal\n"
                f"  Maintenance target  : {cal_maintenance} kcal\n"
                f"  Weight-gain target  : {cal_gain} kcal\n"
                f"  → Choose the target that matches the user's goal above."
            )
        logger.info(f"🧬 [Domain Agent] Analyzing general fitness query: '{user_input[:50]}...'")

        # Step 1: Attempt to find up-to-date info via Web Search if query is complex
        # We search if the query looks like it needs a factual check.
        context = "No additional context found."
        try:
            search_results = await self.web_search.search(f"fitness science {user_input}")
            if search_results and "results" in search_results:
                context = "\n".join([f"- {r['content']}" for r in search_results['results'][:3]])
        except Exception as e:
            logger.warning(f"[Domain Agent] Web search fallback failed: {e}")

        # Step 2: Generate Response
        chain = self.prompt | self.llm
        res = await chain.ainvoke({
            "context": context,
            "user_input": user_input,
            "full_name": full_name,
            "goal": goal,
            "diet_pref": diet_pref,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "age": age,
            "gender": gender,
            "activity_level": activity_level,
            "tdee_str": tdee_str,
            "injuries": injuries,
            "medical": medical
        })


        return {
            "specialist_results": {
                "domain": {
                    "answer": res.content,
                    "status": "success",
                    "sources": "web_search" if context != "No additional context found." else "internal_knowledge"
                }
            }
        }
