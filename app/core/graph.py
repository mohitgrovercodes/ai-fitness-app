import asyncio
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
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
    
    # Map intents to agent classes (Lazy Import)
    for intent in intents:
        if intent == "nutrition":
            from app.agents.nutrition_agent import NutritionAgent
            tasks.append(NutritionAgent().run(state))
        elif intent == "workout":
            from app.agents.training_agent import TrainingAgent
            tasks.append(TrainingAgent().run(state))
        elif intent == "image":
            from app.agents.vision_agent import VisionAgent
            tasks.append(VisionAgent().run(state))
        elif intent == "general":
            from app.agents.domain_agent import DomainAgent
            tasks.append(DomainAgent().run(state))
        elif intent == "progress":
            tasks.append(dummy_progress_agent(state))
            
    # Default to domain agent if no specific intent found
    if not tasks:
        from app.agents.domain_agent import DomainAgent
        tasks.append(DomainAgent().run(state))
        
    # RE-ENABLED: Parallel execution using asyncio.gather
    import time
    start_time = time.time()
    logger.info(f"🧬 [Specialists] Running {len(tasks)} agents in parallel...")
    
    try:
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        logger.info(f"✅ [Specialists] Parallel run completed in {duration:.2f}s.")
    except Exception as e:
        logger.error(f"❌ [Specialists] Critical failure during parallel execution: {e}")
        # Return empty list to prevent crash; downstream nodes will handle missing results
        results = []
    
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
    media_attachments = []
    
    for agent_name, data in results.items():
        if isinstance(data, dict):
            ans = data.get("answer")
            
            # Extract media if present
            gifs = data.get("exercise_gifs", {})
            for name, url in gifs.items():
                if url: media_attachments.append(f"![{name}]({url})")
                
            imgs = data.get("exercise_images", {})
            for name, url in imgs.items():
                if url and name not in gifs: # Avoid duplicate image if GIF exists
                    media_attachments.append(f"![{name}]({url})")
        else:
            ans = data
            
        if ans:
            agent_outputs.append(f"[{agent_name.upper()}]: {ans}")

    if not agent_outputs:
         return {"messages": [AIMessage(content="I've analyzed your request but couldn't find a specific answer.")]}

    context_str = "\n\n".join(agent_outputs)
    
    # Master Coach LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=settings.OPENAI_API_KEY)
    
    prompt = f"""You are the Lead Fitness Coach at 'Agentic AI Gym'.
Your task is to take the specialized advice from your team (provided below) and weave it into a single, cohesive coaching response.

RULES:
- Do NOT just list the points. Integrate them into natural paragraphs.
- If both workout and nutrition advice are provided, explain briefly how they complement each other.
- Maintain a warm, expert, and professional tone.
- Ensure the most important information is clear and actionable.

CRITICAL — NUMBERS & DATA ACCURACY:
- The structured meal/workout data in the API response fields (meals, workout, daily_totals) is the GROUND TRUTH.
- Your job is to write a motivating narrative — do NOT invent or repeat specific calorie/macro/protein numbers in the response text.
- Do NOT say things like "your plan has 1500 kcal" or "120g protein" — those numbers are shown separately in the structured data fields.
- If you reference nutrition, say things like "your meal plan is designed to create a healthy calorie deficit" instead of stating specific numbers.

STRICT INSTRUCTIONS FOR IMAGE-BASED REQUESTS:
- If a [VISION] result is provided, your response MUST focus primarily on the food description and its nutritional breakdown.
- You MUST present the Nutritional Breakdown exactly point-wise with NUMERIC values (e.g., `- **Protein**: 15g`). Do NOT use vague terms.
- NEVER add a "Complementary Aspects" or generic tip section after a vision response.

SPECIALIST ADVICE:
{context_str}

FINAL RESPONSE:"""

    logger.info("✨ [Synthesis] Weaving specialist responses into a master plan.")
    res = await llm.ainvoke(prompt)
    
    final_content = res.content
    if media_attachments:
        final_content += "\n\n### Exercise Demonstrations:\n" + "\n\n".join(media_attachments)
    
    return {
        "messages": [AIMessage(content=final_content)],
        "next_node": "output_safety"
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

async def global_error_handler(state: AgentState):
    """Fallback node for unexpected graph failures."""
    logger.error("🚨 [Global Error] An unexpected error occurred during graph execution.")
    return {
        "messages": [AIMessage(content="I'm sorry, I encountered an unexpected error while processing your request. Please try again in a moment or ask something else! 🛠️")],
        "next_node": END
    }


def build_graph():
    workflow = StateGraph(AgentState)
    
    # ── 1. RETRY POLICY (Production Grade) ───────────────────
    # Handles transient API failures, rate limits, and network hiccups
    api_retry = RetryPolicy(
        initial_interval=2.0,
        max_interval=30.0,
        backoff_factor=2.0,
        max_attempts=3
    )

    # ── 2. NODES (Lazy Initialization) ───────────────────────
    async def _safety(state): return await SafetyGuardrail().check(state)
    async def _orch(state): return await Orchestrator().run(state)
    async def _safety_out(state): return await SafetyGuardrail().check_response(state)
    async def _router(state): return AgentRouter().route(state)
    async def _memory(state): return await MemoryManager().run(state)

    workflow.add_node("safety_guardrail", _safety, retry=api_retry)
    workflow.add_node("output_safety", _safety_out, retry=api_retry)
    workflow.add_node("safe_response_node", safe_response_node)
    workflow.add_node("orchestrator", _orch, retry=api_retry)
    workflow.add_node("agent_router", _router)
    workflow.add_node("specialists_node", specialists_node) # Already has internal retries/sequential logic
    workflow.add_node("synthesis_layer", synthesis_node, retry=api_retry)
    workflow.add_node("memory_manager", _memory)
    workflow.add_node("out_of_scope_handler", out_of_scope_handler)
    
    # ── 3. EDGES (Routing) ───────────────────────────────────
    workflow.set_entry_point("safety_guardrail")

    workflow.add_conditional_edges(
        "safety_guardrail",
        lambda s: s["next_node"],
        {"orchestrator": "orchestrator", "safe_response_node": "safe_response_node"}
    )

    workflow.add_conditional_edges(
        "orchestrator",
        lambda s: s["next_node"],
        {"agent_router": "agent_router", "out_of_scope_handler": "out_of_scope_handler"}
    )
    
    workflow.add_conditional_edges(
        "agent_router",
        lambda s: s["next_node"],
        {"specialists_node": "specialists_node", "end": END}
    )

    workflow.add_edge("specialists_node", "synthesis_layer")
    
    workflow.add_conditional_edges(
        "synthesis_layer",
        lambda state: state.get("next_node", "output_safety"),
        {"output_safety": "output_safety", "memory_manager": "memory_manager"}
    )
    
    workflow.add_conditional_edges(
        "output_safety",
        lambda s: s["next_node"],
        {"memory_manager": "memory_manager", "safe_response_node": "safe_response_node"}
    )

    workflow.add_edge("out_of_scope_handler", "memory_manager")
    workflow.add_edge("safe_response_node", "memory_manager")
    workflow.add_edge("memory_manager", END)

    # ── 4. PERSISTENCE (Production Grade) ───────────────────
    # Use MemorySaver for immediate production reliability on this host.
    # Note: Transition to PostgresSaver/SqliteSaver for multi-host scaling.
    return workflow.compile(checkpointer=MemorySaver())

# Example Usage:
# graph = build_graph()
