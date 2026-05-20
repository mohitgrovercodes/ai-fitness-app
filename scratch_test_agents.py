import asyncio
import sys
import os

# Ensure the app module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.getcwd())

# Force stdout to utf-8 if possible, or just don't use emojis
sys.stdout.reconfigure(encoding='utf-8')

from app.modules.ai.service import AIService
from app.utils.logger import logger

async def run_tests():
    print("\n" + "="*70)
    print("[START] EXTENSIVE AI USE CASE EVALUATION")
    print("="*70 + "\n")

    results = []

    # ---------------------------------------------------------
    # TEST 1: Nutrition - Vegan Weight Loss
    # ---------------------------------------------------------
    print("[TEST 1] Nutrition (Vegan, Female, 65kg, 160cm, Sedentary, Weight Loss)")
    t1_payload = {
        "user_id": "eval_user_1",
        "goal": "lose weight safely",
        "diet_type": "vegan",
        "allergies": ["peanuts"],
        "weight": 65,
        "height": 160,
        "gender": "female",
        "age": 32,
        "activity_level": "SEDENTARY",
        "duration": "1 day"
    }
    
    try:
        t1_res = await AIService.generate_diet_plan(t1_payload)
        summary = t1_res.get('summary', '')
        print("[SUCCESS]")
        print(f"Summary: {summary}")
        results.append(f"### Test 1: Vegan Weight Loss\n**Payload:** {t1_payload}\n**Summary:** {summary}\n**Meals:** {t1_res.get('meals')}\n")
    except Exception as e:
        print(f"[ERROR]: {e}")

    # ---------------------------------------------------------
    # TEST 2: Nutrition - Keto Muscle Gain
    # ---------------------------------------------------------
    print("\n[TEST 2] Nutrition (Keto, Male, 90kg, 185cm, Highly Active, Muscle Gain)")
    t2_payload = {
        "user_id": "eval_user_2",
        "goal": "gain muscle mass",
        "diet_type": "keto",
        "allergies": [],
        "weight": 90,
        "height": 185,
        "gender": "male",
        "age": 28,
        "activity_level": "VERY_ACTIVE",
        "duration": "1 day"
    }
    
    try:
        t2_res = await AIService.generate_diet_plan(t2_payload)
        summary = t2_res.get('summary', '')
        print("[SUCCESS]")
        print(f"Summary: {summary}")
        results.append(f"### Test 2: Keto Muscle Gain\n**Payload:** {t2_payload}\n**Summary:** {summary}\n**Meals:** {t2_res.get('meals')}\n")
    except Exception as e:
        print(f"[ERROR]: {e}")

    # ---------------------------------------------------------
    # TEST 3: Training - Beginner, Back Pain
    # ---------------------------------------------------------
    print("\n[TEST 3] Training (Beginner Female, 70kg, 165cm, No Equipment, Back Pain)")
    t3_payload = {
        "user_id": "eval_user_3",
        "goal": "improve general fitness and tone up",
        "level": "beginner",
        "duration": "1 day",
        "injuries": ["lower back pain", "sciatica"],
        "weight": 70,
        "height": 165,
        "gender": "female",
        "age": 45
    }
    
    try:
        t3_res = await AIService.generate_workout_plan(t3_payload)
        summary = t3_res.get('summary', '')
        tip = t3_res.get('tip', '')
        print("[SUCCESS]")
        print(f"Summary: {summary}\nTip: {tip}")
        results.append(f"### Test 3: Beginner Training w/ Back Pain\n**Payload:** {t3_payload}\n**Summary:** {summary}\n**Tip:** {tip}\n**Workout:** {t3_res.get('workout')}\n")
    except Exception as e:
        print(f"[ERROR]: {e}")

    # ---------------------------------------------------------
    # TEST 4: Training - Advanced, Knee Injury
    # ---------------------------------------------------------
    print("\n[TEST 4] Training (Advanced Male, 85kg, 175cm, Powerlifting, Knee Injury)")
    t4_payload = {
        "user_id": "eval_user_4",
        "goal": "powerlifting strength focus",
        "level": "advanced",
        "duration": "1 day",
        "injuries": ["torn ACL left knee", "patellar tendonitis"],
        "weight": 85,
        "height": 175,
        "gender": "male",
        "age": 30
    }
    
    try:
        t4_res = await AIService.generate_workout_plan(t4_payload)
        summary = t4_res.get('summary', '')
        tip = t4_res.get('tip', '')
        print("[SUCCESS]")
        print(f"Summary: {summary}\nTip: {tip}")
        results.append(f"### Test 4: Advanced Training w/ Knee Injury\n**Payload:** {t4_payload}\n**Summary:** {summary}\n**Tip:** {tip}\n**Workout:** {t4_res.get('workout')}\n")
    except Exception as e:
        print(f"[ERROR]: {e}")

    # ---------------------------------------------------------
    # TEST 5: Orchestrator - Complex Recomp
    # ---------------------------------------------------------
    print("\n[TEST 5] Orchestrator (Skinny Fat Recomp)")
    t5_msg = "I'm skinny fat. I want to lose my belly but get bigger arms. What should I do?"
    
    try:
        t5_res = await AIService.chat(
            user_input=t5_msg,
            user_id="eval_user_5",
            context={"weight_kg": 75, "height_cm": 180, "gender": "male", "goal": "body recomposition"}
        )
        intents = t5_res.get('intents', [])
        print("[SUCCESS]")
        print(f"Intents Detected: {intents}")
        results.append(f"### Test 5: Orchestrator Intent Routing\n**Message:** {t5_msg}\n**Intents:** {intents}\n**Response:** {t5_res.get('response')[:200]}...\n")
    except Exception as e:
        print(f"[ERROR]: {e}")

    # Write output artifact
    with open("C:/Users/mogr1/.gemini/antigravity/brain/8d893a5b-b2dd-4b20-9924-fe2298f030e4/test_results.md", "w", encoding="utf-8") as f:
        f.write("# Extensive AI Use Case Evaluation\n\n")
        f.write("\n---\n".join(results))
    print("\n[COMPLETE] Saved full results to test_results.md")

if __name__ == "__main__":
    asyncio.run(run_tests())
