
from app.core.graph import build_graph
from langchain_core.messages import HumanMessage
import uuid

# Global variable for lazy init
_fitness_graph = None

def get_graph():
    global _fitness_graph
    if _fitness_graph is None:
        from app.core.graph import build_graph
        _fitness_graph = build_graph()
    return _fitness_graph

class AIService:



    @staticmethod
    async def chat(user_input: str, user_id: str, context: dict = None, image_bytes: bytes = None):
        """
        Main entry point for the Agentic AI Gym Chatbot.
        Processes user input through the LangGraph Multi-Agent system.
        """
        config = {"configurable": {"thread_id": user_id}}
        
        # 1. Fetch Profile from Database if available
        from app.core.sql_db import SessionLocal
        from app.modules.profile.service import ProfileService
        
        db = SessionLocal()
        try:
            profile = ProfileService.get_profile(db, user_id)
            db_context = {}
            if profile:
                db_context = {
                    "full_name": profile.full_name,
                    "age": profile.age,
                    "gender": profile.gender.value if profile.gender else None,
                    "weight_kg": profile.weight,
                    "height_cm": profile.height,
                    "goal": profile.goal or None,
                    "activity_level": profile.activity_level.value if profile.activity_level else None,
                    "diet_preference": profile.diet_preference,
                    "injuries": profile.injuries if isinstance(profile.injuries, list) else [],
                    "medical_conditions": profile.medical_conditions if isinstance(profile.medical_conditions, list) else []
                }
            
            # Merge: Incoming context overrides DB context
            merged_context = {**db_context, **(context or {})}
        finally:
            db.close()
            
        # Initial state for the graph
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": merged_context,
            "conversation_summary": "", # This will be updated by the memory_manager
            "user_id": user_id,
            "specialist_results": {"__clear__": True},  # Wipe old zombie data from previous turns
            "image_bytes": image_bytes  # Always set explicitly — None clears old image from MemorySaver
        }
            
        # Run the graph
        graph = get_graph()
        final_state = await graph.ainvoke(initial_state, config=config)
        
        # Extract the last AI message
        last_msg = final_state["messages"][-1]
        
        # Extract media and structured data from specialist results if available
        gifs = {}
        imgs = {}
        workouts = []
        meals = []
        daily_totals = {}
        
        specialists = final_state.get("specialist_results", {})
        for name, data in specialists.items():
            if isinstance(data, dict):
                if "exercise_gifs" in data:
                    gifs.update(data["exercise_gifs"])
                if "exercise_images" in data:
                    imgs.update(data["exercise_images"])
                if "workout" in data and isinstance(data["workout"], list):
                    workouts.extend(data["workout"])
                if "meals" in data and isinstance(data["meals"], list):
                    meals.extend(data["meals"])
                if "daily_totals" in data and isinstance(data["daily_totals"], dict):
                    daily_totals.update(data["daily_totals"])
        
        out_data = {
            "response": last_msg.content,
            "workout": workouts,
            "meals": meals,
            "intents": final_state.get("intent", [])
        }
        
        if gifs:
            out_data["exercise_gifs"] = gifs
        if imgs:
            out_data["exercise_images"] = imgs
        if daily_totals:
            out_data["daily_totals"] = daily_totals
            
        return out_data


    @staticmethod
    async def generate_workout_plan(data: dict):
        """Directly calls the TrainingAgent with structured data from a button/form."""
        from app.agents.training_agent import TrainingAgent
        from app.core.sql_db import SessionLocal
        from app.modules.profile.service import ProfileService
        
        user_id = data.get("user_id", "default")
        goal = data.get("goal", "")
        level = data.get("level", "")
        duration = data.get("duration", "")
        injuries = data.get("injuries", [])
        
        # Fetch real user profile from DB
        db = SessionLocal()
        try:
            profile = ProfileService.get_profile(db, user_id)
            db_context = {}
            if profile:
                db_context = {
                    "full_name": profile.full_name,
                    "age": profile.age,
                    "gender": profile.gender.value if profile.gender else None,
                    "weight_kg": profile.weight,
                    "height_cm": profile.height,
                    "goal": profile.goal or goal,
                    "activity_level": profile.activity_level.value if profile.activity_level else None,
                    "diet_preference": profile.diet_preference,
                    "injuries": profile.injuries if isinstance(profile.injuries, list) else [],
                    "medical_conditions": profile.medical_conditions if isinstance(profile.medical_conditions, list) else []
                }
        finally:
            db.close()
        
        # Merge: request data overrides DB context
        merged_context = {**db_context, "goal": goal or db_context.get("goal", ""), "level": level, "injuries": injuries or db_context.get("injuries", [])}
        
        # Flexibility: if the frontend sends a specific message, use it. Otherwise, build one.
        user_input = data.get("message")
        if not user_input:
            parts = ["Create a structured workout plan"]
            if goal: parts.append(f"for {goal}")
            if level: parts.append(f"at a {level} fitness level")
            if duration: parts.append(f"for a duration of {duration}")
            user_input = " ".join(parts) + "."
        
        state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": merged_context,
            "conversation_summary": "Direct API Generation Request",
            "user_id": user_id
        }
        result = await TrainingAgent().run(state)
        output = result.get("specialist_results", {}).get("training", {})
        return {
            "summary": output.get("summary", ""),
            "workout": output.get("workout", []),
            "tip": output.get("tip", ""),
            "response": output.get("answer", "Could not generate plan."),
        }

    @staticmethod
    def _calculate_tdee(weight_kg, height_cm, age, gender, activity_level) -> dict:
        try:
            w = float(weight_kg or 70)
            h = float(height_cm or 170)
            a = float(age or 25)
        except (ValueError, TypeError):
            w, h, a = 70, 170, 25

        if str(gender).upper() in ("MALE", "M"):
            bmr = (10 * w) + (6.25 * h) - (5 * a) + 5
        else:
            bmr = (10 * w) + (6.25 * h) - (5 * a) - 161

        multipliers = {
            "SEDENTARY": 1.2,
            "LIGHTLY_ACTIVE": 1.375,
            "MODERATELY_ACTIVE": 1.55,
            "VERY_ACTIVE": 1.725,
            "EXTRA_ACTIVE": 1.9,
        }
        factor = multipliers.get(str(activity_level).upper(), 1.2)
        tdee = round(bmr * factor)

        return {
            "bmr": round(bmr),
            "tdee": tdee,
            "weight_loss_target": round(max(bmr, tdee * 0.80)),  # 20% deficit, never below BMR
            "weight_gain_target": round(tdee * 1.15),            # 15% surplus
            "maintenance_target": tdee,
        }

    @staticmethod
    async def generate_diet_plan(data: dict):
        """Directly calls the NutritionAgent with structured data from a button/form."""
        from app.agents.nutrition_agent import NutritionAgent
        from app.core.sql_db import SessionLocal
        from app.modules.profile.service import ProfileService
        
        user_id = data.get("user_id", "default")
        goal = data.get("goal", "")
        diet_type = data.get("diet_type", "")
        allergies = data.get("allergies", [])

        # Fetch real user profile from DB
        db = SessionLocal()
        try:
            profile = ProfileService.get_profile(db, user_id)
            db_context = {}
            if profile:
                tdee_data = AIService._calculate_tdee(
                    profile.weight, profile.height, profile.age,
                    profile.gender.value if profile.gender else "male",
                    profile.activity_level.value if profile.activity_level else "SEDENTARY"
                )
                
                # Determine target calories based on goal
                current_goal = profile.goal or goal
                if "loss" in current_goal.lower() or "lose" in current_goal.lower():
                    target_cal = tdee_data["weight_loss_target"]
                elif "gain" in current_goal.lower() or "bulk" in current_goal.lower():
                    target_cal = tdee_data["weight_gain_target"]
                else:
                    target_cal = tdee_data["maintenance_target"]
                    
                db_context = {
                    "full_name": profile.full_name,
                    "age": profile.age,
                    "gender": profile.gender.value if profile.gender else None,
                    "weight_kg": profile.weight,
                    "height_cm": profile.height,
                    "goal": current_goal,
                    "activity_level": profile.activity_level.value if profile.activity_level else None,
                    "diet_preference": diet_type or profile.diet_preference,
                    "injuries": profile.injuries if isinstance(profile.injuries, list) else [],
                    "medical_conditions": profile.medical_conditions if isinstance(profile.medical_conditions, list) else [],
                    "allergies": allergies,
                    "target_calories": target_cal  # Crucial for Auto-Scaler
                }
        finally:
            db.close()

        # Merge: request data overrides DB context
        merged_context = {**db_context, "goal": goal or db_context.get("goal", "")}
        
        # Flexibility: if the frontend sends a specific message, use it. Otherwise, build one.
        user_input = data.get("message")
        if not user_input:
            parts = ["Create a structured time-based daily meal plan"]
            if goal: parts.append(f"for {goal}")
            if diet_type: parts.append(f"with a {diet_type} dietary preference")
            if allergies: parts.append(f". Avoid these allergies: {', '.join(allergies)}")
            user_input = " ".join(parts) + "."
        
        state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": merged_context,
            "conversation_summary": "Direct API Diet Request",
            "user_id": user_id
        }
        result = await NutritionAgent().run(state)
        output = result.get("specialist_results", {}).get("nutrition", {})
        return {
            "summary": output.get("summary", ""),
            "meals": output.get("meals", []),
            "daily_totals": output.get("daily_totals", {}),
            "tip": output.get("tip", ""),
            "response": output.get("answer", "Could not generate plan.")
        }

    @staticmethod
    async def ask_domain_agent(data: dict):
        """Directly calls the DomainAgent with structured data from a button/form."""
        from app.agents.domain_agent import DomainAgent
        from app.core.sql_db import SessionLocal
        from app.modules.profile.service import ProfileService
        
        user_id = data.get("user_id", "default")
        user_input = data.get("message", "What is muscle hypertrophy?")

        # Fetch real user profile from DB to give DomainAgent some context
        db = SessionLocal()
        db_context = {}
        try:
            profile = ProfileService.get_profile(db, user_id)
            if profile:
                db_context = {
                    "goal": profile.goal,
                    "diet_preference": profile.diet_preference
                }
        finally:
            db.close()

        state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": db_context,
            "conversation_summary": "Direct API Domain Query"
        }
        
        result = await DomainAgent().run(state)
        output = result.get("specialist_results", {}).get("domain", {})
        
        return {
            "response": output.get("answer", "Could not answer the query."),
            "sources": output.get("sources", "internal_knowledge")
        }