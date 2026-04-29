from typing import Dict, Any
from app.core.state import AgentState

class AgentRouter:
    """
    Step 6: AGENT ROUTER
    Routes to the most relevant specialized agent(s).
    """
    def __init__(self):
        # Map intents to graph nodes
        self.intent_map = {
            "workout": "training_agent",
            "nutrition": "nutrition_agent",
            "image": "vision_agent",
            "progress": "progress_agent",
            "general": "domain_agent"
        }

    def route(self, state: AgentState) -> Dict[str, Any]:
        """
        Determines the next specialist agent based on the orchestrator's intent.
        """
        intent = state.get("intent")
        next_node = self.intent_map.get(intent, "domain_agent") # Default to domain agent
        
        print(f"🔀 Agent Router: Routing '{intent}' to '{next_node}'")
        
        return {
            "next_node": next_node
        }
