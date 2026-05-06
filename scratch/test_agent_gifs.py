import asyncio
import os
from dotenv import load_dotenv
from app.core.state import AgentState
from app.agents.training_agent import TrainingAgent
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

async def test_training_agent():
    agent = TrainingAgent()
    state: AgentState = {
        "messages": [HumanMessage(content="Give me a workout for chest including push-ups.")],
        "user_context": {
            "goal": "Build muscle",
            "injuries": []
        },
        "conversation_summary": "User wants a chest workout."
    }
    
    print("🚀 Running Training Agent...")
    db_results = await agent.rag_tool.search(state['messages'][-1].content)
    context = agent._format_context(db_results)
    print("\n--- Retrieved Context ---")
    print(context)
    
    result = await agent.run(state)
    
    print("\n--- Agent Result ---")
    training_results = result.get("specialist_results", {}).get("training", {})
    print(f"Answer: {training_results.get('answer')[:200]}...")
    print(f"Status: {training_results.get('status')}")
    print(f"GIFs: {training_results.get('exercise_gifs')}")

if __name__ == "__main__":
    asyncio.run(test_training_agent())
