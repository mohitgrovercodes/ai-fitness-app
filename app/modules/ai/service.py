
from app.core.graph import build_graph
from langchain_core.messages import HumanMessage
import uuid
from app.agents.base import _compute_tdee

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
                tdee_data =_compute_tdee(
                    profile.weight, profile.height, profile.age,
                    profile.gender.value if profile.gender else "male",
                    profile.activity_level.value if profile.activity_level else "SEDENTARY"
                )
                current_goal = profile.goal or ""
                if "loss" in current_goal.lower() or "lose" in current_goal.lower() or "decrease" in current_goal.lower():
                    target_cal = tdee_data["cal_loss"]
                elif "gain" in current_goal.lower() or "bulk" in current_goal.lower() or "increase" in current_goal.lower():
                    target_cal = tdee_data["cal_gain"]
                else:
                    target_cal = tdee_data["cal_maintenance"]

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
                    "medical_conditions": profile.medical_conditions if isinstance(profile.medical_conditions, list) else [],
                    "target_calories": target_cal  # ✅ Activates Auto-Scaler in _validate_output
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
        per_day_totals = {}
        
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
                if "per_day_totals" in data and isinstance(data["per_day_totals"], dict):
                    per_day_totals.update(data["per_day_totals"])
        
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
        if per_day_totals:
            out_data["per_day_totals"] = per_day_totals
            
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
        import json
        from app.core.redis_client import redis_manager
        try:
            if redis_manager.is_available():
                redis_key = f"chat_history:{user_id}"
                human_msg = {"type": "human", "content": user_input}
                ai_msg = {
                    "type": "ai",
                    "structured_data": {"training": output},
                    "intents": ["workout"]
                }
                redis_manager.client.rpush(redis_key, json.dumps(human_msg))
                redis_manager.client.rpush(redis_key, json.dumps(ai_msg))
        except Exception as e:
            print(f"Redis save error in generate_workout: {e}")
        return {
            "summary": output.get("summary", ""),
            "workout": output.get("workout", []),
            "tip": output.get("tip", ""),
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
                # Inline Mifflin-St Jeor TDEE (same formula as base.py — no external utility).
                # All three calorie targets stored in user_context; the LLM picks the right
                # one based on the goal text it reads in its system prompt.
                try:
                    w   = float(profile.weight or 0)
                    h   = float(profile.height or 0)
                    a   = float(profile.age or 0)
                    g   = (profile.gender.value if profile.gender else "male").upper()
                    act = (profile.activity_level.value if profile.activity_level else "SEDENTARY").upper()
                    _multipliers = {
                        "SEDENTARY": 1.2, "LIGHTLY_ACTIVE": 1.375,
                        "MODERATELY_ACTIVE": 1.55, "VERY_ACTIVE": 1.725, "EXTRA_ACTIVE": 1.9,
                    }
                    if w > 0 and h > 0 and a > 0:
                        bmr  = (10*w + 6.25*h - 5*a + 5) if g in ("MALE", "M") else (10*w + 6.25*h - 5*a - 161)
                        tdee = bmr * _multipliers.get(act, 1.2)
                        cal_loss        = round(max(bmr, tdee * 0.80))
                        cal_maintenance = round(tdee)
                        cal_gain        = round(tdee * 1.15)
                    else:
                        cal_loss = cal_maintenance = cal_gain = 0
                except (ValueError, TypeError):
                    cal_loss = cal_maintenance = cal_gain = 0

                current_goal = profile.goal or goal
                db_context = {
                    "full_name":          profile.full_name,
                    "age":                profile.age,
                    "gender":             profile.gender.value if profile.gender else None,
                    "weight_kg":          profile.weight,
                    "height_cm":          profile.height,
                    "goal":               current_goal,
                    "activity_level":     profile.activity_level.value if profile.activity_level else None,
                    "diet_preference":    diet_type or profile.diet_preference,
                    "injuries":           profile.injuries if isinstance(profile.injuries, list) else [],
                    "medical_conditions": profile.medical_conditions if isinstance(profile.medical_conditions, list) else [],
                    "allergies":          allergies,
                    # Three calorie targets — LLM selects the right one based on the goal
                    "cal_loss":           cal_loss,
                    "cal_maintenance":    cal_maintenance,
                    "cal_gain":           cal_gain,
                }
        finally:
            db.close()

        # Merge: request data overrides DB context
        merged_context = {**db_context, "goal": goal or db_context.get("goal", "")}
        
        # Flexibility: if the frontend sends a specific message, use it. Otherwise, build one.
        user_input = data.get("message")
        if not user_input:
            duration = data.get("duration", "")  # e.g. "daily", "weekly", "monthly", "10 days", "45 days"
            parts = ["Create a structured meal plan"]
            if duration:
                parts.append(f"for {duration}")
            else:
                parts.append("for today (daily)")
            if goal:
                parts.append(f"for the goal: {goal}")
            if diet_type:
                parts.append(f"with a {diet_type} dietary preference")
            if allergies:
                parts.append(f". Avoid these allergens: {', '.join(allergies)}")
            user_input = " ".join(parts) + "."
        
        state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": merged_context,
            "conversation_summary": "Direct API Diet Request",
            "user_id": user_id
        }
        result = await NutritionAgent().run(state)
        output = result.get("specialist_results", {}).get("nutrition", {})
        import json
        from app.core.redis_client import redis_manager
        try:
            if redis_manager.is_available():
                redis_key = f"chat_history:{user_id}"
                human_msg = {"type": "human", "content": user_input}
                ai_msg = {
                    "type": "ai",
                    "structured_data": {"training": output},
                    "intents": ["workout"]
                }
                redis_manager.client.rpush(redis_key, json.dumps(human_msg))
                redis_manager.client.rpush(redis_key, json.dumps(ai_msg))
        except Exception as e:
            print(f"Redis save error in generate_workout: {e}")
        
        return {
            "summary": output.get("summary", ""),
            "meals": output.get("meals", []),
            "daily_totals": output.get("daily_totals", {}),
            "per_day_totals": output.get("per_day_totals", {}),
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