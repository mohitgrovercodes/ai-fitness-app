from langgraph.graph import StateGraph, END
from app.core.state import AgentState
from app.agents.orchestrator import Orchestrator
from app.agents.router import AgentRouter

# Initialize Nodes
orchestrator = Orchestrator()
router = AgentRouter()

def build_graph():
    """
    Step-by-Step Construction of the Fit Bot DAG.
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("orchestrator", orchestrator.run)
    workflow.add_node("agent_router", router.route)
    
    # Placeholders for specialists (we will build these next)
    # workflow.add_node("vision_agent", vision_agent.run)
    # ... etc

    # 2. Set Entry Point
    workflow.set_entry_point("orchestrator")

    # 3. Define Edges (Routing)
    def route_after_orchestrator(state: AgentState):
        return state["next_node"]

    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "agent_router": "agent_router",
            "out_of_scope_handler": END # For now
        }
    )
    
    # Define route after Router
    def route_after_router(state: AgentState):
        return state["next_node"]

    # This will be updated as we add specialists
    workflow.add_conditional_edges(
        "agent_router",
        route_after_router,
        {
            # Specialists will be added here
            "domain_agent": END,
            "vision_agent": END,
            "nutrition_agent": END,
            "training_agent": END,
            "progress_agent": END
        }
    )

    return workflow.compile()

# Example Usage:
# graph = build_graph()
