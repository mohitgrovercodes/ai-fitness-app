from typing import Dict, Any, List
from app.core.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.utils.logger import logger

class SafetyResult(BaseModel):
    is_safe: bool
    reason: str
    suggested_response: str

class SafetyGuardrail:
    """
    Step 2: SAFETY GUARDRAILS (Production Grade)
    Ensures no harmful or illegal medical advice is processed.
    """
    def __init__(self):
        from app.core.config import settings
        self.model = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0, 
            api_key=settings.OPENAI_API_KEY
        ).with_structured_output(SafetyResult, method="function_calling")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Fit Bot Safety Officer. Assess if the user input or the proposed response is safe.
            
STRICT POLICIES:
1. No medical diagnosis or disease treatment.
2. No pro-eating disorder or extreme starvation content.
3. CULTURAL POLICY: DO NOT recommend BEEF in any diet or recipe. If the user asks for beef or if a response suggests it, flag it as unsafe."""),
            ("human", "{input}")
        ])

    async def check(self, state: AgentState) -> Dict[str, Any]:
        last_message = state['messages'][-1].content
        chain = self.prompt | self.model
        res = await chain.ainvoke({"input": last_message})
        
        logger.info(f"🛡️ [Safety] Safe: {res.is_safe} | Reason: {res.reason}")
        
        return {
            "is_safe": res.is_safe,
            "safety_reason": res.reason,
            "safety_response": res.suggested_response,
            "next_node": "orchestrator" if res.is_safe else "safe_response_node"
        }

class AgentRouter:
    """
    Step 6: PRODUCTION ROUTER
    Supports complex intent mapping and parallel tasking.
    """
    def __init__(self):
        self.intent_map = {
            "workout": "training_agent",
            "nutrition": "nutrition_agent",
            "image": "vision_agent",
            "progress": "progress_agent",
            "general": "domain_agent"
        }

    def route(self, state: AgentState) -> Dict[str, Any]:
        intents = state.get("intent", [])
        
        # We now use a single specialists_node to manage parallel execution internally
        # this avoids the LangGraph parallel join issue.
        target_nodes = ["specialists_node"]
            
        logger.info(f"🔀 [Router] Routing to: {target_nodes}")
        
        return {
            "next_node": target_nodes
        }
