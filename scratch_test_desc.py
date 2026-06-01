import asyncio
import os
import sys

# Ensure the root directory is in the python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.modules.ai.service import AIService
from app.models.user import User, GoalType, ActivityLevel, Gender

async def test_desc():
    service = AIService()
    user = User(
        id="test_user",
        full_name="John Doe",
        age=30,
        gender=Gender.MALE,
        weight_kg=80,
        height_cm=180,
        goal=GoalType.MUSCLE_GAIN,
        activity_level=ActivityLevel.MODERATE,
        injuries="Neck pain",
        medical_conditions="",
        diet_preference="None"
    )
    res = await service.chat(
        user_id="test_user",
        message="Generate a 5 day workout",
        user_data=user
    )
    print("----- AI Response -----")
    if "workout" in res:
        for w in res["workout"]:
            print(f"Name: {w.get('name')}")
            print(f"Benefit: {w.get('benefit')}")
            print(f"Description: {w.get('description')}")
            print("-" * 20)
    else:
        print("No workout in response:", res)

if __name__ == "__main__":
    asyncio.run(test_desc())
