from typing import Dict, Any
from app.core.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class Orchestrator:
    """
    Step 3: ORCHESTRATOR
    Central Controller & Flow Manager.
    Uses an Intent Classifier to decide the path.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.parser = JsonOutputParser()
        
        # Intent Classification Prompt (Step 4 & 5 in Diagram)
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Orchestrator for 'Fit Bot', an AI Gym Assistant.
Your goal is to classify the user's intent and determine if it's within the fitness domain.

INTENTS:
1. 'workout': Exercise instructions, routines, or physical training.
2. 'nutrition': Food, calories, diets, or recipes.
3. 'progress': Tracking weight, logging activities, or analytics.
4. 'image': The user has provided an image for identification.
5. 'general': Fitness-related general knowledge.
6. 'out_of_scope': Anything not related to fitness, health, or nutrition.

RESPONSE FORMAT (JSON):
{{
  "intent": "one of the above",
  "is_fitness_domain": true/false,
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}"""),
            ("human", "{input}")
        ])

    async def run(self, state: AgentState) -> Dict[str, Any]:
        """
        Processes the input to classify intent and route to the Agent Router.
        """
        last_message = state['messages'][-1].content
        
        # Step 4: Intent Classifier
        chain = self.intent_prompt | self.llm | self.parser
        classification = await chain.ainvoke({"input": last_message})
        
        intent = classification.get("intent", "out_of_scope")
        is_fitness = classification.get("is_fitness_domain", False)
        
        # Step 5 & 6: Routing Logic
        if not is_fitness:
            next_node = "out_of_scope_handler"
        else:
            # Map intent to specialized agents
            next_node = "agent_router"
            
        print(f"🧠 Orchestrator: Intent='{intent}', Route='{next_node}'")
        
        return {
            "intent": intent,
            "is_fitness_domain": is_fitness,
            "next_node": next_node
        }
