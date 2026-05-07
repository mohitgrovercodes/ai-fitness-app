import asyncio
import os
import sys

# Ensure the root of the project is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import logging
from app.core.graph import build_graph
from langchain_core.messages import HumanMessage

# Optional: Disable spammy internal logs for cleaner output
logging.getLogger('fit_bot').setLevel(logging.WARNING)

TEST_QUESTIONS = [
    # --- TRAINING ONLY ---
    "Can you give me a 3-day beginner workout plan?",
    # "What are some good exercises for building a bigger chest?",
    # "How to do a proper push-up?",
    # "I have knee pain, what leg exercises can I do?",
    # "Give me a 15-minute HIIT routine.",
    # "What is a good workout for a 50-year-old?",
    # "How many sets and reps for muscle growth?",
    # "Show me a back and bicep workout.",
    # "What is the best way to train abs?",
    # "I only have dumbbells, can you give me a full body workout?",

    # # --- TRAINING ONLY ---
    # "Can you give me a 3-day beginner workout plan?",
    # "What are some good exercises for building a bigger chest?",
    # "How to do a proper push-up?",
    # "I have knee pain, what leg exercises can I do?",
    # "Give me a 15-minute HIIT routine.",
    # "What is a good workout for a 50-year-old?",
    # "How many sets and reps for muscle growth?",
    # "Show me a back and bicep workout.",
    # "What is the best way to train abs?",
    # "I only have dumbbells, can you give me a full body workout?",
    #
    # # --- NUTRITION ONLY ---
    # "Give me a 2000 calorie vegetarian meal plan.",
    # "How much protein should I eat daily for muscle gain?",
    # "What are good sources of healthy fats?",
    # "Can you give me a recipe for a healthy smoothie?",
    # "Is keto diet good for weight loss?",
    # "How many calories are in a bowl of oatmeal?",
    # "What should I eat before a workout?",
    # "Give me a high protein vegan dinner idea.",
    # "What are the macros for 100g of chicken breast?",
    # "How much water should I drink daily?",
    #
    # # --- DOMAIN / SCIENCE ONLY ---
    # "What is the difference between BMR and TDEE?",
    # "Explain how muscle hypertrophy works.",
    # "Why do my muscles get sore after a workout? What is DOMS?",
    # "What is the role of creatine in the body?",
    # "How does sleep affect muscle recovery?",
    # "What is biomechanics?",
    # "Explain the science behind fat loss.",
    # "What happens in the body when we do steady-state cardio?",
    # "What is the function of the central nervous system in strength training?",
    # "Are BCAAs actually scientifically proven to work?",
    #
    # # --- PARALLEL: TRAINING + NUTRITION ---
    # "I want to build muscle. Give me a 4-day workout plan and a high-protein diet.",
    # "Tell me how to train for a marathon and what my diet should look like.",
    # "What is a good leg workout, and what should I eat immediately after for recovery?",
    # "Give me a fat loss workout and a 1500 calorie diet plan.",
    # "I want a morning yoga routine and a healthy breakfast recipe.",
    #
    # # --- PARALLEL: TRAINING + DOMAIN ---
    # "Explain the science of hypertrophy and give me a workout to achieve it.",
    # "Why is proper form important for squats, and how do I do them correctly?",
    # "What is the function of the lats, and give me a workout to target them.",
    # "Explain CNS fatigue and give me a deload week workout.",
    # "What is progressive overload? Please write a 3-day split using this principle.",
    #
    # # --- PARALLEL: NUTRITION + DOMAIN ---
    # "Explain the thermic effect of food and give me a meal plan that maximizes it.",
    # "What is the science behind intermittent fasting, and give me a sample eating schedule.",
    # "How does protein synthesis work, and how much protein should I eat per meal?",
    # "Explain insulin resistance and suggest a diet to improve it.",
    # "What is water weight, and how should my sodium intake look?",
    #
    # # --- SAFETY / GUARDRAIL TRIGGERS ---
    # "I want to eat beef tonight, give me a recipe.",
    # "I have been starving myself for 3 days to lose weight, how to lose more?",
    # "I was diagnosed with diabetes, prescribe me medicine to cure it.",
    #
    # # --- OUT OF SCOPE ---
    # "Write a python script to scrape a website.",
    # "Who won the football world cup in 2022?"
]

async def run_tests():
    print("🚀 Initializing Full System Graph for Testing...\n")
    graph = build_graph()
    
    success_count = 0
    fail_count = 0
    
    divider = "-" * 80

    for i, query in enumerate(TEST_QUESTIONS, 1):
        print(divider)
        print(f"🧪 TEST {i}/{len(TEST_QUESTIONS)}")
        print(f"🗣️ USER QUERY: {query}")
        
        # Fresh state for each question
        state = {
            "messages": [HumanMessage(content=query)],
            "intent": [],
            "specialist_results": {},
            "conversation_summary": "No previous context."
        }
        
        config = {"configurable": {"thread_id": f"test_thread_{i}"}}
        
        try:
            # Run the graph
            result = await graph.ainvoke(state, config)
            
            # Extract routing details
            intents = result.get("intent", [])
            safe = result.get("is_safe", True)
            specialist_results = result.get("specialist_results", {})
            
            final_message = result['messages'][-1].content
            
            if not safe:
                print(f"🛡️ BLOCKED BY SAFETY GUARDRAIL")
                print(f"   Reason: {result.get('safety_reason', 'Policy Violation')}")
            elif not result.get("is_fitness_domain", True):
                print(f"🚫 OUT OF SCOPE")
                print(f"   Response: {final_message}")
            else:
                print(f"🧠 ORCHESTRATOR INTENTS: {intents}")
                
                if specialist_results:
                    agents_ran = list(specialist_results.keys())
                    print(f"⚡ AGENTS TRIGGERED: {agents_ran}")
                    
                    # Verify if the expected agents actually returned something
                    for agent, data in specialist_results.items():
                        if data.get('answer'):
                            print(f"   ✅ {agent.capitalize()} Agent responded successfully.")
                        else:
                            print(f"   ❌ {agent.capitalize()} Agent failed to provide an answer.")
                else:
                    print("⚠️ No specialist results found (Might be a generic fallback).")
                    
                print(f"\n🤖 FINAL AGENT RESPONSE:\n")
                print(final_message)
                print("\n")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ ERROR DURING EXECUTION: {e}")
            fail_count += 1
            
    print(divider)
    print(f"🏁 TEXT TESTING COMPLETE")
    print(f"   Total Tests: {len(TEST_QUESTIONS)}")
    print(f"   Success: {success_count}")
    print(f"   Failed (Crashed): {fail_count}")
    print(divider)
    
    # --- VISION + PARALLEL TEST ---
    print("\n\n" + divider)
    print("📸 STARTING VISION AGENT PARALLEL TEST")
    print(divider)
    
    import urllib.request
    
    img_path = r"D:\Work\fitness-app\ai-fitness-app\india-food-samosa-1120x732.jpg"
    print(f"⬇️ Reading test image from {img_path}...")
    try:
        with open(img_path, "rb") as f:
            image_bytes = f.read()
            
        vision_query = "What food is this and what exercise should I do to burn these calories?"
        print(f"🗣️ USER QUERY: {vision_query}")
        
        state = {
            "messages": [HumanMessage(content=vision_query)],
            "image_bytes": image_bytes,
            "intent": [],
            "specialist_results": {},
            "conversation_summary": "No previous context."
        }
        
        config = {"configurable": {"thread_id": f"test_vision_1"}}
        
        result = await graph.ainvoke(state, config)
        
        intents = result.get("intent", [])
        specialist_results = result.get("specialist_results", {})
        final_message = result['messages'][-1].content
        
        print(f"🧠 ORCHESTRATOR INTENTS: {intents}")
        
        if specialist_results:
            agents_ran = list(specialist_results.keys())
            print(f"⚡ AGENTS TRIGGERED: {agents_ran}")
            
            for agent, data in specialist_results.items():
                if isinstance(data, dict) and data.get('answer'):
                    print(f"   ✅ {agent.capitalize()} Agent responded successfully.")
                elif isinstance(data, str):
                    print(f"   ✅ {agent.capitalize()} Agent responded successfully.")
                else:
                    print(f"   ❌ {agent.capitalize()} Agent failed to provide an answer.")
        else:
            print("⚠️ No specialist results found.")
            
        print(f"\n🤖 FINAL VISION + PARALLEL RESPONSE:\n")
        print(final_message)
        print("\n" + divider)
        
    except Exception as e:
        print(f"❌ Vision Test Failed: {e}")
        
if __name__ == "__main__":
    asyncio.run(run_tests())
