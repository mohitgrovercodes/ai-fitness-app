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
- Name: {full_name}
- Goal: {goal}
- Weight: {weight_kg} kg | Height: {height_cm} cm | Age: {age}
- Dietary Preference: {diet_pref}

USER QUERY: {user_input}"""),
            ("human", "{user_input}")
        ])

    async def run(self, state: AgentState) -> Dict[str, Any]:
        user_input = state['messages'][-1].content
        user_context = state.get("user_context", {})
        full_name = user_context.get("full_name", "User")
        goal = user_context.get("goal", "General Fitness")
        weight_kg = user_context.get("weight_kg", "Unknown")
        height_cm = user_context.get("height_cm", "Unknown")
        age = user_context.get("age", "Unknown")
        diet_pref = user_context.get("diet_preference", "None")
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
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "age": age,
            "diet_pref": diet_pref
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
