import asyncio
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add the project root to sys.path so that 'app' can be imported
sys.path.append(str(Path(__file__).parent.parent))

from app.agents.training_agent import TrainingAgent
from langchain_core.messages import HumanMessage
from app.utils.logger import logger
import logging

# Suppress noisy logs for a cleaner CLI experience
logging.getLogger("fit_bot").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

async def interactive_test():
    agent = TrainingAgent()
    print("\n" + "="*50)
    print("      FIT BOT — EXERCISE GIF TESTER")
    print("="*50)
    print("Type your workout query below (e.g., 'Chest workout', 'Leg day')")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("🤔 Your Query: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break

        state = {
            "messages": [HumanMessage(content=query)],
            "user_context": {"goal": "General Fitness", "injuries": []},
            "conversation_summary": ""
        }
        
        print("\n⏳ Processing...")
        try:
            result = await agent.run(state)
            training_results = result.get("specialist_results", {}).get("training", {})
            
            print("\n" + "-"*30)
            print("🤖 AGENT RESPONSE:")
            print("-"*30)
            print(training_results.get('answer'))
            
            gifs = training_results.get('exercise_gifs', {})
            if gifs:
                print("\n📽️ ATTACHED GIFS:")
                for name, path in gifs.items():
                    print(f"  • {name}: {path}")
            else:
                print("\nℹ️ No GIFs found for these exercises.")
            print("-"*30 + "\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(interactive_test())
