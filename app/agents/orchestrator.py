from typing import Dict, Any
from app.core.state import AgentState
from app.schema.orchestration import IntentResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.utils.logger import logger

class Orchestrator:
    """
    Step 3 & 4: PRODUCTION ORCHESTRATOR
    Uses Structured Output for 100% reliable intent classification.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        from app.core.config import settings
        self.model = ChatOpenAI(
            model=model_name, 
            temperature=0, 
            api_key=settings.OPENAI_API_KEY
        ).with_structured_output(IntentResponse, method="function_calling")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Orchestrator for 'Agentic AI Gym'.
Your task is to classify user intent based on the current message and the conversation context.

CONTEXT SUMMARY: {summary}

CATEGORIES:
- 'workout': Training, exercises, routines.
- 'nutrition': Food, calories, meal plans.
- 'progress': Logging metrics, history.
- 'image': User provided an image.
- 'general': Fitness knowledge.
- 'out_of_scope': Non-fitness topics.

If the user says "tell me more" or "how many calories in that?", use the SUMMARY to determine the intent."""),
            ("human", "{input}")
        ])

    async def run(self, state: AgentState) -> Dict[str, Any]:
        last_message = state['messages'][-1].content
        summary = state.get('conversation_summary', "No previous context.")
        
        # Invoke with structured output
        chain = self.prompt | self.model
        res: IntentResponse = await chain.ainvoke({
            "input": last_message,
            "summary": summary
        })
        
        # Logic for domain checking
        route = "agent_router" if res.is_fitness_domain else "out_of_scope_handler"
        
        logger.info(f"✅ [Orchestrator] Intents: {res.intents} | Domain: {res.is_fitness_domain}")
        
        return {
            "intent": res.intents,
            "is_fitness_domain": res.is_fitness_domain,
            "next_node": route
        }
