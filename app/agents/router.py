from typing import Dict, Any, List,Optional
from app.core.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from app.utils.logger import logger

class SafetyResult(BaseModel):
    is_safe: bool
    reason:  Optional[str]
    suggested_response: str = ""

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
The input may be in English, Hindi (Devanagari script), or Hinglish (Hindi grammar/vocabulary written in Roman/Latin script).

STRICT POLICIES:
1. No medical diagnosis or disease treatment.
2. No pro-eating disorder or extreme starvation content.
3. CULTURAL POLICY (CRITICAL): DO NOT recommend BEEF in any diet or recipe. If the user asks for beef or if a response suggests it, flag it as unsafe. This includes all language/script variations:
   - English: "beef", "cow meat"
   - Hindi (Devanagari): "बीफ", "गाय का मांस", "गौ मांस", "बड़े का मीट"
   - Hinglish (Roman script): "beef", "gay ka meat", "cow meat", "gau maans", "bade ka meat"
4. WHITELIST (CRITICAL): It is 100% safe for the user to ask about their own weight, height, BMI, fitness goals, or profile details (e.g., "What is my weight?", "Mera weight kitna hai?"). DO NOT flag these as eating disorder issues. They are safe.
5. WHITELIST (CRITICAL): All food and diet advisory questions are 100% SAFE and within the core scope of this fitness chatbot. This includes ANY of the following — "should I eat this?", "kya main ye kha sakta hu?", "kya ye weight loss ke liye accha hai?", or ANY vague short query like "should I eat it?" that implies the user is asking about a food item. NEVER flag these as unsafe. They are the primary purpose of this app.
6. WHITELIST (CRITICAL): General workout, exercise, and fitness questions are always safe.
7. If you flag something as unsafe, you MUST provide a polite `suggested_response` explaining why it cannot be answered. The rejection response MUST be in the exact same language and script as the input (e.g. if the input is in Devanagari Hindi, reject in Devanagari Hindi; if in Hinglish, reject in Hinglish; if in English, reject in English)."""),
            ("human", "{input}")
        ])

    async def check(self, state: AgentState) -> Dict[str, Any]:
        """Checks the USER input for safety."""
        last_message = state['messages'][-1].content
        return await self._run_check(last_message, "orchestrator", "safe_response_node")

    async def check_response(self, state: AgentState) -> Dict[str, Any]:
        """Checks the AGENT response for safety before sending to user."""
        last_message = state['messages'][-1].content
        # If unsafe, we redirect to safe_response_node which will provide the suggested_response
        return await self._run_check(last_message, "END", "safe_response_node")

    async def _run_check(self, text: str, safe_node: str, unsafe_node: str) -> Dict[str, Any]:
        chain = self.prompt | self.model
        res = await chain.ainvoke({"input": text})
        
        logger.info(f"🛡️ [Safety] Safe: {res.is_safe} | Reason: {res.reason}")
        
        return {
            "is_safe": res.is_safe,
            "safety_reason": res.reason,
            "safety_response": res.suggested_response,
            "next_node": safe_node if res.is_safe else unsafe_node
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
