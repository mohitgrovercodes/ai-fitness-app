from typing import Dict, Any
from app.agents.base import BaseAgent
from app.core.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.tools.web_tools import WebSearchTool
from app.utils.logger import logger

class DomainAgent(BaseAgent):
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

SEARCH CONTEXT:
{context}

USER QUERY: {user_input}"""),
            ("human", "{user_input}")
        ])

    async def run(self, state: AgentState) -> Dict[str, Any]:
        user_input = state['messages'][-1].content
        logger.info(f"🧬 [Domain Agent] Analyzing general fitness query: '{user_input[:50]}...'")

        # Step 1: Attempt to find up-to-date info via Web Search if query is complex
        # We search if the query looks like it needs a factual check.
        context = "No additional context found."
        try:
            search_results = await self.web_search.search(f"fitness science {user_input}")
            if search_results:
                context = "\n".join([f"- {r['content']}" for r in search_results[:3]])
        except Exception as e:
            logger.warning(f"[Domain Agent] Web search fallback failed: {e}")

        # Step 2: Generate Response
        chain = self.prompt | self.llm
        res = await chain.ainvoke({
            "context": context,
            "user_input": user_input
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
