import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

# Add app to path
import os
sys.path.append(os.getcwd())

from app.agents.base import BaseRAGAgent
from app.agents.nutrition_agent import NutritionAgent, NutritionAnalysis
from app.core.state import AgentState

async def test_base_rag_agent_fallback():
    print("Testing BaseRAGAgent fallback logic...")
    
    # Mock dependencies
    mock_rag = AsyncMock()
    mock_rag.search.return_value = []
    mock_rag.multi_query_search.return_value = []
    
    mock_web = MagicMock()
    mock_web.is_available = False
    
    class TestAgent(BaseRAGAgent):
        def _format_context(self, results):
            return ""

    agent = TestAgent(
        agent_name="Test Agent",
        rag_tool=mock_rag,
        web_search_tool=mock_web,
        output_schema=NutritionAnalysis,
        system_prompt="Test prompt"
    )
    
    # Mock LLM
    agent.llm = AsyncMock()
    agent.llm.ainvoke.return_value = NutritionAnalysis(
        is_accurate=False,
        needs_web_search=True,
        final_answer="Fallback test"
    )

    state = {
        "messages": [MagicMock(content="test query")],
        "user_context": {},
        "conversation_summary": "Test summary"
    }

    try:
        result = await agent.run_logic(state, "test", "test")
        print("✅ BaseRAGAgent fallback check passed (No KeyError).")
    except Exception as e:
        print(f"❌ BaseRAGAgent fallback check failed: {e}")

async def test_nutrition_multiplier():
    print("\nTesting NutritionAgent multiplier logic...")
    agent = NutritionAgent()
    
    # Mock tools
    agent.rag_tool = AsyncMock()
    agent.rag_tool.search.return_value = [{"food_name": "Chicken", "calories": 100}]
    agent.rag_tool.multi_query_search.return_value = [{"food_name": "Chicken", "calories": 200}]
    
    # Mock LLM to simulate Phase 1 analysis finding a multiplier
    agent.llm = AsyncMock()
    # First call: returns multiplier and says not accurate (triggers Phase 2)
    agent.llm.ainvoke.side_effect = [
        NutritionAnalysis(
            is_accurate=False, 
            needs_web_search=False, 
            quantity_multiplier=2.0, 
            sub_queries=["200g chicken"],
            final_answer="Wait..."
        ),
        # Second call (after Phase 2 search with multiplier)
        NutritionAnalysis(
            is_accurate=True, 
            needs_web_search=False, 
            quantity_multiplier=2.0, 
            final_answer="Here is your 200g chicken info."
        )
    ]

    state = {
        "messages": [MagicMock(content="200g chicken")],
        "user_context": {},
        "conversation_summary": "Summary"
    }

    await agent.run(state)
    
    # Verify Phase 2 search was called with the multiplier
    agent.rag_tool.multi_query_search.assert_called_with("200g chicken", ["200g chicken"], multiplier=2.0)
    print("✅ NutritionAgent multiplier propagation passed.")

if __name__ == "__main__":
    asyncio.run(test_base_rag_agent_fallback())
    asyncio.run(test_nutrition_multiplier())
