from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.core.state import AgentState
from app.agents.orchestrator import Orchestrator
from app.agents.router import AgentRouter, SafetyGuardrail
from app.agents.nutrition_agent import NutritionAgent
from app.agents.training_agent import TrainingAgent
from langchain_core.messages import AIMessage

from app.agents.memory_agent import MemoryManager

# Initialize Nodes
orchestrator = Orchestrator()
router = AgentRouter()
nutrition_agent = NutritionAgent()
training_agent = TrainingAgent()
safety_guardrail = SafetyGuardrail()
memory_manager = MemoryManager()

async def synthesis_node(state: AgentState):
    """
    Step 8: SYNTHESIS LAYER
    Merges outputs from all active agents into a unified response.
    """
    results = state.get("specialist_results", {})
    nutrition_out = results.get("nutrition", {}).get("answer", "")
    training_out = results.get("training", {}).get("answer", "")
    
    # Merge outputs from multiple agents if both were triggered
    responses = []
    if nutrition_out:
        responses.append(nutrition_out)
    if training_out:
        responses.append(training_out)
        
    final_response = "\n\n---\n\n".join(responses) if responses else "I've analyzed your request but couldn't find a specific answer in my database."
    
    print("✨ [Synthesis] Finalizing response.")
    return {
        "messages": [AIMessage(content=final_response)],
        "next_node": END
    }

async def out_of_scope_handler(state: AgentState):
    """Handles general/non-fitness queries."""
    msg = "I'm your AI Fitness & Nutrition coach. I specialize in workouts and diet. Could we stick to those topics?"
    return {
        "messages": [AIMessage(content=msg)],
        "next_node": END
    }

async def safe_response_node(state: AgentState):
    """Handles queries blocked by the safety guardrail."""
    msg = state.get("safety_response", "I cannot fulfill this request due to safety or policy guidelines.")
    return {
        "messages": [AIMessage(content=msg)],
        "next_node": END
    }

# --- Placeholder Nodes ---
async def dummy_vision_agent(state: AgentState):
    print("🚧 [Vision Agent] Placeholder reached.")
    return {"specialist_results": {"vision": {"answer": "Vision Agent is under construction."}}}

async def dummy_progress_agent(state: AgentState):
    print("🚧 [Progress Agent] Placeholder reached.")
    return {"specialist_results": {"progress": {"answer": "Progress Agent is under construction."}}}

async def dummy_domain_agent(state: AgentState):
    print("🚧 [Domain Agent] Placeholder reached.")
    return {"specialist_results": {"domain": {"answer": "Domain Agent is under construction."}}}

def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("safety_guardrail", safety_guardrail.check)
    workflow.add_node("safe_response_node", safe_response_node)
    workflow.add_node("orchestrator", orchestrator.run)
    workflow.add_node("agent_router", router.route)
    workflow.add_node("nutrition_agent", nutrition_agent.run)
    workflow.add_node("synthesis_layer", synthesis_node)
    workflow.add_node("memory_manager", memory_manager.run)
    workflow.add_node("out_of_scope_handler", out_of_scope_handler)
    
    # Placeholders
    workflow.add_node("training_agent", training_agent.run)
    workflow.add_node("vision_agent", dummy_vision_agent)
    workflow.add_node("progress_agent", dummy_progress_agent)
    workflow.add_node("domain_agent", dummy_domain_agent)

    # 2. Set Entry Point
    workflow.set_entry_point("safety_guardrail")

    # 3. Define Edges (Routing)
    def route_after_safety(state: AgentState):
        return state["next_node"]

    workflow.add_conditional_edges(
        "safety_guardrail",
        route_after_safety,
        {
            "orchestrator": "orchestrator",
            "safe_response_node": "safe_response_node"
        }
    )

    def route_after_orchestrator(state: AgentState):
        return state["next_node"]

    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "agent_router": "agent_router",
            "out_of_scope_handler": "out_of_scope_handler"
        }
    )
    
    def route_after_router(state: AgentState):
        # Router returns a list of nodes for parallel tasks
        return state["next_node"]

    workflow.add_conditional_edges(
        "agent_router",
        route_after_router,
        {
            "nutrition_agent": "nutrition_agent",
            "training_agent": "training_agent",
            "vision_agent": "vision_agent",
            "progress_agent": "progress_agent",
            "domain_agent": "domain_agent",
            "end": END
        }
    )

    # Specialists move to Synthesis
    workflow.add_edge("nutrition_agent", "synthesis_layer")
    workflow.add_edge("training_agent", "synthesis_layer")
    workflow.add_edge("vision_agent", "synthesis_layer")
    workflow.add_edge("progress_agent", "synthesis_layer")
    workflow.add_edge("domain_agent", "synthesis_layer")
    
    workflow.add_edge("synthesis_layer", "memory_manager")
    workflow.add_edge("out_of_scope_handler", "memory_manager")
    workflow.add_edge("safe_response_node", "memory_manager")
    workflow.add_edge("memory_manager", END)

    # Compile with MemorySaver
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Example Usage:
# graph = build_graph()
