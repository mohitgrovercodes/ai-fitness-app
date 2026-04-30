from typing import Dict, Any
from app.core.state import AgentState
from app.schema.orchestration import IntentResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class Orchestrator:
    """
    Step 3 & 4: PRODUCTION ORCHESTRATOR
    Uses Structured Output for 100% reliable intent classification.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        # We use .with_structured_output to force the LLM to follow our Pydantic schema
        self.model = ChatOpenAI(model=model_name, temperature=0).with_structured_output(IntentResponse)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Orchestrator for 'Agentic AI Gym'.
Your task is to classify user intent.

CATEGORIES:
- 'workout': Training, exercises, routines.
- 'nutrition': Food, calories, meal plans.
- 'progress': Logging metrics, history.
- 'image': User provided an image (even if not explicitly mentioned, if text refers to 'this' or 'that' with an image).
- 'general': Fitness knowledge.
- 'out_of_scope': Non-fitness topics.

Check if the query is in the fitness domain."""),
            ("human", "{input}")
        ])

    async def run(self, state: AgentState) -> Dict[str, Any]:
        last_message = state['messages'][-1].content
        
        # Invoke with structured output - No more parsing errors!
        chain = self.prompt | self.model
        res: IntentResponse = await chain.ainvoke({"input": last_message})
        
        # Logic for domain checking
        route = "agent_router" if res.is_fitness_domain else "out_of_scope_handler"
        
        print(f"✅ [Orchestrator] Intent: {res.intent} | Domain: {res.is_fitness_domain}")
        
        return {
            "intent": res.intent,
            "is_fitness_domain": res.is_fitness_domain,
            "next_node": route
        }
