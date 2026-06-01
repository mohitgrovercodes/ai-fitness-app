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

    Includes a smart LLM-based topic gate that runs BEFORE the web search
    to reject off-topic queries (saving API cost) while accepting any
    legitimate fitness, nutrition, or exercise-science question.
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

    async def _is_relevant(self, query: str) -> bool:
        """
        Lightweight LLM gate (structured output, gpt-4o-mini, temp=0).
        Decides if a query falls within the fitness / nutrition / exercise-science
        domain.  Runs BEFORE the expensive web search to save API cost on junk.
        Returns True if relevant, False otherwise.
        """
        from pydantic import BaseModel, Field

        class RelevanceVerdict(BaseModel):
            is_relevant: bool = Field(
                description="True if the query is about fitness, exercise science, "
                            "nutrition science, sports medicine, body composition, "
                            "supplementation, recovery, anatomy for exercise, or "
                            "the user's personal fitness profile. False otherwise."
            )
            reason: str = Field(description="One-sentence justification")

        try:
            gate_llm = self.llm.with_structured_output(
                RelevanceVerdict, method="function_calling"
            )
            verdict: RelevanceVerdict = await gate_llm.ainvoke(
                f"""You are a strict topic-relevance classifier for a fitness and nutrition AI application.
The user query may be in English, Hindi (Devanagari), or Hinglish (Roman-script Hindi).

ACCEPT queries about:
- Exercise science, workout techniques, training principles, gym equipment
- Nutrition science, macronutrients, micronutrients, supplements, hydration
- Human anatomy and physiology AS IT RELATES TO exercise or fitness
- Sports medicine, injury prevention, recovery, flexibility, mobility
- Body composition, weight management, BMR, TDEE, metabolic science
- Sleep science AS IT RELATES TO fitness performance or muscle recovery
- The user's personal profile data (weight, height, BMI, goals, injuries)
- Mental health AS IT RELATES TO fitness motivation or exercise psychology
- History or origin of specific exercises, sports, or fitness practices

REJECT queries about:
- Geography, politics, history (non-fitness), entertainment, pop culture
- General science NOT related to fitness (physics, chemistry, astronomy)
- Technology, programming, business, finance, law
- Random trivia, fun facts, riddles, jokes
- Any topic that has NO reasonable connection to fitness, health, or nutrition

USER QUERY: "{query}"

Classify this query strictly."""
            )
            logger.info(
                f"🔬 [Domain Gate] LLM verdict: relevant={verdict.is_relevant} | "
                f"reason='{verdict.reason}'"
            )
            return verdict.is_relevant
        except Exception as e:
            # Fail-open: if the gate itself errors, let the query through
            logger.warning(f"[Domain Gate] LLM relevance check failed: {e} — allowing query as fallback.")
            return True

    async def run(self, state: AgentState) -> Dict[str, Any]:
        user_input = state['messages'][-1].content

        # ── TOPIC GATE: LLM Relevance Check ──────────────────────────────────
        # Runs BEFORE the web search call to avoid wasting API cost on
        # off-topic queries.  Cheap (~150 tokens, gpt-4o-mini).
        if not await self._is_relevant(user_input):
            logger.info(f"🚫 [Domain Agent] Query rejected by topic gate: '{user_input[:60]}...'")
            return self._rejection_response()

        # ── PASSED GATE — proceed with normal flow ───────────────────────────
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

        # Step 1: Web Search for up-to-date context
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

    @staticmethod
    def _rejection_response() -> Dict[str, Any]:
        """Standard response when a query is outside the fitness/nutrition domain."""
        return {
            "specialist_results": {
                "domain": {
                    "answer": (
                        "That's an interesting question, but it falls outside my area of expertise! "
                        "I'm your AI Exercise Scientist — I specialize in fitness science, nutrition, "
                        "exercise physiology, and everything gym-related. "
                        "Feel free to ask me anything about workouts, diet, supplements, recovery, "
                        "or how your body works during training! 💪"
                    ),
                    "status": "rejected",
                    "sources": "topic_gate"
                }
            }
        }
