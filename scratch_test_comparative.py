"""
Comparative / Parametric Test Suite (Direct Service Layer — No Server Required)
Tests pairs of inputs where ONE variable changes while all others are held constant.
Verifies that the AI response changes in the CORRECT direction.

Run: $env:PYTHONUTF8="1"; venv\python.exe scratch_test_comparative.py
"""

import sys
import os
import asyncio

sys.path.append(os.getcwd())
sys.stdout.reconfigure(encoding='utf-8')

RESULTS_PATH = "C:/Users/mogr1/.gemini/antigravity/brain/8d893a5b-b2dd-4b20-9924-fe2298f030e4/Comparative_Test_Results.md"

from app.modules.ai.service import AIService

def extract_diet_calories(body):
    daily = body.get("daily_totals", {})
    if daily and daily.get("calories"):
        return float(str(daily.get("calories", 0)))
    meals = body.get("meals", [])
    if meals:
        return sum(float(m.get("calories", 0)) for m in meals)
    return 0.0

def extract_diet_carbs(body):
    meals = body.get("meals", [])
    total = 0.0
    for m in meals:
        carbs_str = str(m.get("carbs", "0")).replace("g", "").strip()
        try:
            total += float(carbs_str)
        except:
            pass
    return total

def extract_workout_names(body):
    return [w.get("name", "?") for w in body.get("workout", [])]

results = []

def compare(label, var_name, val_a, val_b, metric_a, metric_b, expected_direction, notes=""):
    passed = False
    try:
        fa = float(str(metric_a).replace(",", "").replace("g carbs","").replace(" kcal","").split()[0])
        fb = float(str(metric_b).replace(",", "").replace("g carbs","").replace(" kcal","").split()[0])
        if expected_direction == "A>B":
            passed = fa > fb
        elif expected_direction == "B>A":
            passed = fb > fa
        elif expected_direction == "DIFFERENT":
            passed = metric_a != metric_b
    except:
        if expected_direction == "DIFFERENT":
            passed = str(metric_a).strip() != str(metric_b).strip()

    status = "PASS" if passed else "FAIL"
    icon = "[PASS]" if passed else "[FAIL]"
    print(f"\n{icon} {label}")
    print(f"  Changed: {var_name}  A={val_a}  vs  B={val_b}")
    print(f"  Metric A: {metric_a}")
    print(f"  Metric B: {metric_b}")
    print(f"  Expected: {expected_direction}  ->  {status}")
    if notes:
        print(f"  Note: {notes}")

    results.append({
        "label": label,
        "var_name": var_name,
        "val_a": str(val_a),
        "val_b": str(val_b),
        "metric_a": str(metric_a)[:200],
        "metric_b": str(metric_b)[:200],
        "expected": expected_direction,
        "status": status,
        "notes": notes
    })

# =============================================================================
# BASE PROFILE (held constant across all tests unless noted)
# =============================================================================
BASE = {
    "user_id": "comp_eval_user",
    "weight": 80,
    "height": 175,
    "gender": "male",
    "age": 30,
    "activity_level": "SEDENTARY",
    "goal": "weight loss",
    "diet_type": "standard",
    "allergies": [],
    "duration": "1 day",
    "level": "intermediate",
    "injuries": []
}

async def run():

    # ================================================================
    # EXPERIMENT 1: Activity Level — Sedentary vs Very Active
    # Hypothesis: Higher activity = higher calories needed
    # ================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: Activity Level (SEDENTARY vs VERY_ACTIVE)")
    print("="*60)
    a1 = await AIService.generate_diet_plan({**BASE, "activity_level": "SEDENTARY"})
    b1 = await AIService.generate_diet_plan({**BASE, "activity_level": "VERY_ACTIVE"})
    compare(
        "Daily Calories: Sedentary vs Very Active",
        "activity_level", "SEDENTARY", "VERY_ACTIVE",
        f"{extract_diet_calories(a1):.0f} kcal",
        f"{extract_diet_calories(b1):.0f} kcal",
        "B>A",
        "Very Active TDEE ~1.72x Sedentary. Diet should reflect significantly more calories."
    )

    # ================================================================
    # EXPERIMENT 2: Body Weight — 60kg vs 100kg
    # Hypothesis: Heavier = higher BMR = higher calories
    # ================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Body Weight (60kg vs 100kg)")
    print("="*60)
    a2 = await AIService.generate_diet_plan({**BASE, "weight": 60})
    b2 = await AIService.generate_diet_plan({**BASE, "weight": 100})
    compare(
        "Daily Calories: 60kg vs 100kg",
        "weight", "60kg", "100kg",
        f"{extract_diet_calories(a2):.0f} kcal",
        f"{extract_diet_calories(b2):.0f} kcal",
        "B>A",
        "Each extra kg adds ~10 kcal to BMR per Mifflin-St Jeor. 100kg user needs ~400 more kcal."
    )

    # ================================================================
    # EXPERIMENT 3: Gender — Female vs Male (identical stats)
    # Hypothesis: Male BMR is 166 kcal higher than female at same stats
    # ================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: Gender (Female vs Male, same height/weight/age)")
    print("="*60)
    a3 = await AIService.generate_diet_plan({**BASE, "gender": "female"})
    b3 = await AIService.generate_diet_plan({**BASE, "gender": "male"})
    compare(
        "Daily Calories: Female vs Male",
        "gender", "female", "male",
        f"{extract_diet_calories(a3):.0f} kcal",
        f"{extract_diet_calories(b3):.0f} kcal",
        "B>A",
        "Mifflin-St Jeor: Male = +166 kcal constant over female."
    )

    # ================================================================
    # EXPERIMENT 4: Age — 25 yr vs 60 yr
    # Hypothesis: Younger person has higher BMR
    # ================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 4: Age (25yr vs 60yr)")
    print("="*60)
    a4 = await AIService.generate_diet_plan({**BASE, "age": 25})
    b4 = await AIService.generate_diet_plan({**BASE, "age": 60})
    compare(
        "Daily Calories: Age 25 vs Age 60",
        "age", "25yr", "60yr",
        f"{extract_diet_calories(a4):.0f} kcal",
        f"{extract_diet_calories(b4):.0f} kcal",
        "A>B",
        "Mifflin-St Jeor: each extra year subtracts ~5 kcal. A 35-year gap = ~175 kcal difference."
    )

    # ================================================================
    # EXPERIMENT 5: Height — 155cm vs 195cm
    # Hypothesis: Taller person has higher BMR
    # ================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 5: Height (155cm vs 195cm)")
    print("="*60)
    a5 = await AIService.generate_diet_plan({**BASE, "height": 155})
    b5 = await AIService.generate_diet_plan({**BASE, "height": 195})
    compare(
        "Daily Calories: 155cm vs 195cm",
        "height", "155cm", "195cm",
        f"{extract_diet_calories(a5):.0f} kcal",
        f"{extract_diet_calories(b5):.0f} kcal",
        "B>A",
        "Mifflin-St Jeor: +6.25 kcal per cm. 40cm height gap = ~250 kcal difference."
    )

    # ================================================================
    # EXPERIMENT 6: Diet Type — Standard vs Keto (carb comparison)
    # Hypothesis: Keto should have dramatically fewer carbs
    # ================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 6: Diet Type (Standard vs Keto) — Carb Check")
    print("="*60)
    a6 = await AIService.generate_diet_plan({**BASE, "diet_type": "standard"})
    b6 = await AIService.generate_diet_plan({**BASE, "diet_type": "keto"})
    compare(
        "Total Daily Carbs: Standard vs Keto",
        "diet_type", "standard", "keto",
        f"{extract_diet_carbs(a6):.0f}g carbs",
        f"{extract_diet_carbs(b6):.0f}g carbs",
        "A>B",
        "Keto requires <50g carbs/day. Standard diet has 200-300g. Should be dramatically different."
    )

    # ================================================================
    # EXPERIMENT 7: Fitness Goal — Muscle Gain vs Flexibility
    # Hypothesis: Exercises should be completely different
    # ================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 7: Workout Goal (Muscle Gain vs Flexibility)")
    print("="*60)
    a7 = await AIService.generate_workout_plan({**BASE, "goal": "muscle gain"})
    b7 = await AIService.generate_workout_plan({**BASE, "goal": "improve flexibility and mobility"})
    names_a7 = extract_workout_names(a7)
    names_b7 = extract_workout_names(b7)
    compare(
        "Exercises: Muscle Gain vs Flexibility Goal",
        "goal", "muscle gain", "flexibility",
        str(names_a7),
        str(names_b7),
        "DIFFERENT",
        "Muscle gain should include compound lifts/push exercises. Flexibility should include stretches/yoga."
    )

    # ================================================================
    # EXPERIMENT 8: Injury Impact — No Injury vs Knee Injury
    # Hypothesis: Same profile with a knee injury should produce different/modified workout
    # ================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 8: Injury Impact (No Injury vs Torn ACL)")
    print("="*60)
    a8 = await AIService.generate_workout_plan({**BASE, "goal": "general fitness", "injuries": []})
    b8 = await AIService.generate_workout_plan({**BASE, "goal": "general fitness", "injuries": ["torn ACL", "knee pain"]})
    names_a8 = extract_workout_names(a8)
    names_b8 = extract_workout_names(b8)
    tip_a8 = a8.get("tip", "")[:120]
    tip_b8 = b8.get("tip", "")[:120]
    compare(
        "Exercises: No Injury vs Torn ACL + Knee Pain",
        "injuries", "none", "torn ACL + knee pain",
        str(names_a8),
        str(names_b8),
        "DIFFERENT",
        f"TipA: {tip_a8} || TipB: {tip_b8}"
    )

    # ================================================================
    # WRITE REPORT
    # ================================================================
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("# Comparative / Parametric Test Report\n\n")
        f.write("> **Methodology:** Each experiment changes exactly ONE variable while all others are held constant.\n")
        f.write("> The AI response is expected to change in a predictable, correct direction.\n")
        f.write("> This validates that the deterministic TDEE math and LLM prompts are working correctly.\n\n")
        f.write(f"**Total: {len(results)} experiments | PASS: {pass_count} | FAIL: {fail_count}**\n\n")
        f.write("---\n\n")

        f.write("## Summary Table\n\n")
        f.write("| # | Experiment | Changed Variable | Value A | Value B | Expected | Metric A | Metric B | Result |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(results, 1):
            icon = "✅" if r["status"] == "PASS" else "❌"
            ma = r["metric_a"][:40]
            mb = r["metric_b"][:40]
            f.write(f"| {i} | {r['label'][:35]} | `{r['var_name']}` | {r['val_a']} | {r['val_b']} | {r['expected']} | {ma} | {mb} | {icon} |\n")

        f.write("\n---\n\n## Detailed Findings\n\n")
        for i, r in enumerate(results, 1):
            icon = "✅" if r["status"] == "PASS" else "❌"
            f.write(f"### {icon} Experiment {i}: {r['label']}\n\n")
            f.write(f"| Field | Value |\n|---|---|\n")
            f.write(f"| **Changed Variable** | `{r['var_name']}` |\n")
            f.write(f"| **Value A** | `{r['val_a']}` |\n")
            f.write(f"| **Value B** | `{r['val_b']}` |\n")
            f.write(f"| **Metric A** | `{r['metric_a']}` |\n")
            f.write(f"| **Metric B** | `{r['metric_b']}` |\n")
            f.write(f"| **Expected Direction** | `{r['expected']}` |\n")
            f.write(f"| **Result** | `{r['status']}` |\n")
            if r["notes"]:
                f.write(f"\n**Analysis:** {r['notes']}\n")
            f.write("\n---\n\n")

    print(f"\n{'='*60}")
    print(f"COMPLETE: {pass_count} PASS | {fail_count} FAIL out of {len(results)} experiments")
    print(f"Report saved to Comparative_Test_Results.md")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(run())
