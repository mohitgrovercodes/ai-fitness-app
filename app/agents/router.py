from typing import Dict, Any, List
from app.core.state import AgentState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(SafetyResult)
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
        
        print(f"🛡️ [Safety] Safe: {res.is_safe} | Reason: {res.reason}")
        
        return {
            "is_safe": res.is_safe,
            "safety_reason": res.reason,
            "next_node": "orchestrator" if res.is_safe else "safe_response_node"
        }

class AgentRouter:
    """
    Step 6: PRODUCTION ROUTER
    Supports complex intent mapping.
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
        intent = state.get("intent")
        
        # Parallel Tasking Loophole Fix:
        # If the user asks about food AND exercise, we could return a list of nodes.
        # For now, we fetch the primary specialist.
        target_node = self.intent_map.get(intent, "domain_agent")
        
        print(f"🔀 [Router] Redirecting to: {target_node}")
        
        return {
            "next_node": target_node
        }
