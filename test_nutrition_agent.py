"""
============================================================
  FIT BOT - Agent Test Runner
  Run: python test_nutrition_agent.py
============================================================
"""
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage

# ── Import the compiled graph ─────────────────────────────
from app.core.graph import build_graph

# ── Sample User Context ───────────────────────────────────
SAMPLE_USER_CONTEXT = {
    "goal": "Fat Loss",
    "age": 25,
    "weight_kg": 80,
    "height_cm": 175,
    "injuries": [],
    "medical_conditions": []
}

# ── Test Queries ──────────────────────────────────────────
TEST_CASES = [
    {
        "name": "✅ TEST 1: Simple Nutrition Query",
        "query": "How many calories are in chicken breast?",
        "expect": "nutrition_agent"
    },
    {
        "name": "✅ TEST 2: Complex Nutrition Query",
        "query": "Give me high protein vegetarian dinner options",
        "expect": "nutrition_agent"
    },
    {
        "name": "🛡️ TEST 3: Beef Restriction Policy",
        "query": "I want to add beef to my meal plan",
        "expect": "safety_block"
    },
    {
        "name": "🚫 TEST 4: Out-of-Scope Query",
        "query": "What is the capital of France?",
        "expect": "out_of_scope_handler"
    },
    {
        "name": "👽 TEST 5: Missing Data / Fictional Food",
        "query": "I just ate a 'Quantum Turbo-Charge Protein Bar'. How many calories are in that?",
        "expect": "nutrition_agent"
    }
]

async def run_test(graph, test_case: dict):
    """Run a single test case through the agent."""
    print(f"\n{'='*60}")
    print(f"  {test_case['name']}")
    print(f"  Query: \"{test_case['query']}\"")
    print(f"{'='*60}")
    
    initial_state = {
        "messages": [HumanMessage(content=test_case["query"])],
        "user_context": SAMPLE_USER_CONTEXT,
        "intent": [],
        "is_fitness_domain": True,
        "next_node": "orchestrator",
        "specialist_results": {},
        "is_safe": True,
        "safety_reason": None,
        "safety_response": None
    }
    
    try:
        config = {"configurable": {"thread_id": "test_thread_1"}}
        result = await graph.ainvoke(initial_state, config)
        
        # Print the final response
        final_messages = result.get("messages", [])
        if final_messages:
            last_message = final_messages[-1].content
            print(f"\n🤖 AGENT RESPONSE:\n{last_message}")
        
        # Print diagnostic info
        print(f"\n📊 DIAGNOSTICS:")
        print(f"   Intent: {result.get('intent', 'N/A')}")
        print(f"   Is Safe: {result.get('is_safe', 'N/A')}")
        print(f"   Domain: {result.get('is_fitness_domain', 'N/A')}")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")

async def main():
    print("\n" + "🏋️ "*20)
    print("     FIT BOT AGENT TEST SUITE")
    print("🏋️ "*20)
    
    # Build the graph
    print("\n⏳ Initializing the Agentic Graph...")
    graph = build_graph()
    print("✅ Graph compiled successfully!\n")
    
    # Run tests
    for test_case in TEST_CASES:
        await run_test(graph, test_case)
        await asyncio.sleep(1) # Small delay between tests
    
    print(f"\n{'='*60}")
    print("  ✅ ALL TESTS COMPLETED")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    asyncio.run(main())
