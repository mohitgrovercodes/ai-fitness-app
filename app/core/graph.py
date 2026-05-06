import asyncio
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.core.state import AgentState
from app.agents.orchestrator import Orchestrator
from app.agents.router import AgentRouter, SafetyGuardrail
# Specialist imports moved inside nodes to prevent binary extension conflicts
from langchain_core.messages import AIMessage
from app.utils.logger import logger

from app.agents.memory_agent import MemoryManager

# Node functions will initialize agents lazily

async def specialists_node(state: AgentState):
    """
    NEW: Specialists Coordinator
    Runs active agents in parallel using asyncio.gather.
    This ensures a single join point for the synthesis layer.
    """
    intents = state.get("intent", [])
    tasks = []
    
    from app.agents.nutrition_agent import NutritionAgent
    from app.agents.training_agent import TrainingAgent
    from app.agents.vision_agent import VisionAgent
    from app.agents.domain_agent import DomainAgent

    # Map intents to agent run methods (Lazy Init)
    intent_map = {
        "nutrition": NutritionAgent().run,
        "workout": TrainingAgent().run,
        "image": VisionAgent().run,
        "general": DomainAgent().run,
        "progress": dummy_progress_agent
    }
    
    # Collect tasks for all detected intents
    for intent in intents:
        if intent in intent_map:
            tasks.append(intent_map[intent](state))
            
    # Default to domain agent if no specific intent found
    if not tasks:
        tasks.append(dummy_domain_agent(state))
        
    # Re-enforce Sequential execution to prevent Segfaults with ChromaDB
    logger.info(f"🧬 [Specialists] Running {len(tasks)} agents sequentially for stability...")
    results = []
    for task in tasks:
        results.append(await task)
    
    # Merge all specialist results into a single update
    merged_results = {}
    for r in results:
        if "specialist_results" in r:
            merged_results.update(r["specialist_results"])
            
    return {"specialist_results": merged_results}

async def synthesis_node(state: AgentState):
    """
    Step 8: INTELLIGENT SYNTHESIS LAYER
    Weaves multiple specialist responses into a single, cohesive coaching message.
    """
    from langchain_openai import ChatOpenAI
    from app.core.config import settings
    
    results = state.get("specialist_results", {})
    if not results:
        return {"messages": [AIMessage(content="I've analyzed your request but couldn't find a specific answer.")]}

    # Format agent outputs for the Master Coach
    agent_outputs = []
    for agent_name, data in results.items():
        ans = data.get("answer") or data # Handle different return types
        if ans:
            agent_outputs.append(f"[{agent_name.upper()}]: {ans}")

    if not agent_outputs:
         return {"messages": [AIMessage(content="I've analyzed your request but couldn't find a specific answer.")]}

    context_str = "\n\n".join(agent_outputs)
    
    # Master Coach LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=settings.OPENAI_API_KEY)
    
    prompt = f"""You are the Lead Fitness Coach at 'Agentic AI Gym'.
Your task is to take the specialized advice from your team (provided below) and weave it into a single, fluent, and encouraging response for the user.

RULES:
- Do NOT just list the points. Integrate them.
- If both workout and nutrition advice are provided, explain how they complement each other.
- Maintain a warm, expert, and highly professional tone.
- Ensure the most important information is clear and actionable.
- Format the final response using clean Markdown.

SPECIALIST ADVICE:
{context_str}

FINAL COHESIVE RESPONSE:"""

    logger.info("✨ [Synthesis] Weaving specialist responses into a master plan.")
    res = await llm.ainvoke(prompt)
    
    return {
        "messages": [AIMessage(content=res.content)]
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
    logger.info("🚧 [Vision Agent] Placeholder reached.")
    return {"specialist_results": {"vision": {"answer": "Vision Agent is under construction."}}}

async def dummy_progress_agent(state: AgentState):
    logger.info("🚧 [Progress Agent] Placeholder reached.")
    return {"specialist_results": {"progress": {"answer": "Progress Agent is under construction."}}}

async def dummy_domain_agent(state: AgentState):
    logger.info("🚧 [Domain Agent] Placeholder reached.")
    return {"specialist_results": {"domain": {"answer": "Domain Agent is under construction."}}}

def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes (Lazy Initialization)
    async def _safety(state): return await SafetyGuardrail().check(state)
    async def _orch(state): return await Orchestrator().run(state)
    async def _router(state): return AgentRouter().route(state)
    async def _memory(state): return await MemoryManager().run(state)

    workflow.add_node("safety_guardrail", _safety)
    workflow.add_node("safe_response_node", safe_response_node)
    workflow.add_node("orchestrator", _orch)
    workflow.add_node("agent_router", _router)
    workflow.add_node("specialists_node", specialists_node)
    workflow.add_node("synthesis_layer", synthesis_node)
    workflow.add_node("memory_manager", _memory)
    workflow.add_node("out_of_scope_handler", out_of_scope_handler)
    
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
        return state["next_node"]

    workflow.add_conditional_edges(
        "agent_router",
        route_after_router,
        {
            "specialists_node": "specialists_node",
            "end": END
        }
    )

    # Specialists move to Synthesis
    workflow.add_edge("specialists_node", "synthesis_layer")
    workflow.add_edge("synthesis_layer", "memory_manager")
    workflow.add_edge("out_of_scope_handler", "memory_manager")
    workflow.add_edge("safe_response_node", "memory_manager")
    workflow.add_edge("memory_manager", END)

    # Compile WITHOUT MemorySaver for stability testing
    return workflow.compile()

# Example Usage:
# graph = build_graph()
