import os
import sys
import asyncio
from dotenv import load_dotenv

# Add application directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from app.modules.ai.service import AIService
from app.utils.logger import logger

async def run_test_case(title: str, query: str, context: dict = None):
    print("\n" + "="*80)
    print(f"[TEST CASE]: {title}")
    print(f"Query: '{query}'")
    print(f"Context: {context}")
    print("="*80)
    
    try:
        result = await AIService.chat(
            user_input=query,
            user_id="test_multilingual_user",
            context=context
        )
        
        print("\nRESPONSE:")
        print(result["response"])
        print("\nDETECTED INTENTS:", result.get("intents"))
        
        # Verify macros/meals or workouts if returned
        if result.get("meals"):
            print(f"Generated {len(result['meals'])} meals.")
        if result.get("workout"):
            print(f"Generated {len(result['workout'])} exercise sets.")
            
    except Exception as e:
        print(f"Test Failed with Exception: {e}")

async def main():
    print("[Startup] Starting Multilingual Verification Tests...")
    
    # ── Test 1: Standard English Query ──────────────────────────────────────
    await run_test_case(
        title="English Query - Standard Weight Loss",
        query="Suggest a quick legs workout plan for weight loss."
    )
    
    # ── Test 2: Native Hindi (Devanagari) Query ──────────────────────────────
    await run_test_case(
        title="Hindi (Devanagari) Query - Muscle Gain",
        query="मुझे वजन बढ़ाना है और एक अच्छा डाइट चार्ट चाहिए।"
    )
    
    # ── Test 3: Hinglish (Roman Script) Query ──────────────────────────────────
    await run_test_case(
        title="Hinglish (Roman) Query - Workout & Diet",
        query="Mera weight kaise kam hoga? Chest aur arms ka fat loss plan de do."
    )
    
    # ── Test 4: Multilingual Safety Rejection (Hindi) ──────────────────────────
    await run_test_case(
        title="Hindi Safety Rejection - Beef Recipe Query",
        query="बीफ की सब्जी बनाने की रेसिपी बताओ वजन बढ़ाने के लिए।"
    )
    
    # ── Test 5: Multilingual Safety Rejection (Hinglish) ───────────────────────
    await run_test_case(
        title="Hinglish Safety Rejection - Beef Recommendation Query",
        query="kya main gau maans ya beef kha sakta hu protein intake badhane ke liye?"
    )

    # ── Test 6: Direct API Workout Plan Generation in Hinglish ──────────────────
    print("\n" + "="*80)
    print("TEST CASE 6: Direct API generate_workout_plan in Hinglish")
    print("Input Payload: {'message': 'Mere liye workout do'}")
    print("="*80)
    try:
        direct_result = await AIService.generate_workout_plan({
            "user_id": "test_direct_user",
            "message": "Mere liye workout do",
            "goal": "muscle gain",
            "level": "beginner"
        })
        print("\nDIRECT API RESPONSE (Summary):")
        print(direct_result["summary"])
        print(f"Generated {len(direct_result.get('workout', []))} exercises in Hinglish Roman script.")
        if direct_result.get("workout"):
            # Print first exercise details to verify translation
            ex = direct_result["workout"][0]
            print(f"First exercise name: '{ex.get('name')}'")
            print(f"   Description: '{ex.get('description')}'")
            print(f"   Benefit: '{ex.get('benefit')}'")
    except Exception as e:
        print(f"Direct API Test Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
