
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

def _is_null_or_empty(val):
    if val is None:
        return True
    if isinstance(val, str):
        v_stripped = val.strip().lower()
        return v_stripped in ("", "null", "none", "undefined")
    if isinstance(val, (list, set, tuple)):
        return len(val) == 0 or all(_is_null_or_empty(item) for item in val)
    return False

def _resolve_numeric(payload_val, db_val, default_val=None):
    if not _is_null_or_empty(payload_val):
        try:
            return float(payload_val)
        except (ValueError, TypeError):
            pass
    if not _is_null_or_empty(db_val):
        try:
            return float(db_val)
        except (ValueError, TypeError):
            pass
    return default_val

def _resolve_string(payload_val, db_val, default_val=None):
    if not _is_null_or_empty(payload_val):
        return str(payload_val).strip()
    if not _is_null_or_empty(db_val):
        return str(db_val).strip()
    return default_val

def _resolve_list(payload_val, db_val, default_val=None):
    if default_val is None:
        default_val = []
        
    def to_clean_list(val):
        if val is None:
            return None
        if isinstance(val, str):
            parts = [item.strip() for item in val.split(",") if item.strip()]
            return parts if parts else None
        if isinstance(val, list):
            parts = []
            for item in val:
                if item is not None:
                    s = str(item).strip()
                    if s and s.lower() not in ("null", "none", "undefined"):
                        parts.append(s)
            return parts if parts else None
        return None

    clean_payload = to_clean_list(payload_val)
    if clean_payload is not None:
        return clean_payload
        
    clean_db = to_clean_list(db_val)
    if clean_db is not None:
        return clean_db
        
    return default_val

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
            
            # ── Normalization: Extract and align keys from incoming payload context ──
            raw_context = context or {}
            payload_weight = raw_context.get("weight") if raw_context.get("weight") is not None else raw_context.get("weight_kg")
            payload_height = raw_context.get("height") if raw_context.get("height") is not None else raw_context.get("height_cm")
            payload_age    = raw_context.get("age")
            payload_gender = raw_context.get("gender")
            payload_activity = raw_context.get("activity_level")
            payload_goal = raw_context.get("goal")
            payload_diet = raw_context.get("diet_preference")
            payload_injuries = raw_context.get("injuries")
            payload_med = raw_context.get("medical_conditions")

            # Priority 1: Manual payload values (with fallback if null/empty). Priority 2: DB profile values. Priority 3: safe defaults.
            w_val = _resolve_numeric(payload_weight, profile.weight if profile else None, None)
            h_val = _resolve_numeric(payload_height, profile.height if profile else None, None)
            a_val = _resolve_numeric(payload_age, profile.age if profile else None, None)
            
            inj = _resolve_list(payload_injuries, profile.injuries if profile else None, [])
            med = _resolve_list(payload_med, profile.medical_conditions if profile else None, [])
            g_val = _resolve_string(payload_goal, profile.goal if profile else None, None)
            d_val = _resolve_string(payload_diet, profile.diet_preference if profile else None, None)

            # Resolve gender and activity level
            req_gender = _resolve_string(payload_gender, profile.gender.value if profile and profile.gender else None, "male")
            req_activity = _resolve_string(payload_activity, profile.activity_level.value if profile and profile.activity_level else None, "SEDENTARY")
            
            gender_str = req_gender.lower()
            activity_str = req_activity.upper()

            # ── Dynamic TDEE Math ──
            # Recompute TDEE and target calories dynamically based on the final resolved biometric state
            if w_val and h_val and a_val:
                tdee_data = _compute_tdee(w_val, h_val, a_val, gender_str, activity_str)
                current_goal = g_val or ""
                if "loss" in current_goal.lower() or "lose" in current_goal.lower() or "decrease" in current_goal.lower():
                    target_cal = tdee_data["cal_loss"]
                elif "gain" in current_goal.lower() or "bulk" in current_goal.lower() or "increase" in current_goal.lower():
                    target_cal = tdee_data["cal_gain"]
                else:
                    target_cal = tdee_data["cal_maintenance"]
            else:
                tdee_data = {"bmr": 0, "tdee": 0, "cal_loss": 0, "cal_maintenance": 0, "cal_gain": 0}
                target_cal = 0

            merged_context = {
                "full_name":          profile.full_name if profile else "User",
                "age":                a_val,
                "gender":             gender_str.upper(),
                "weight_kg":          w_val,
                "height_cm":          h_val,
                "goal":               g_val,
                "activity_level":     activity_str,
                "diet_preference":    d_val,
                "injuries":           inj if isinstance(inj, list) else [],
                "medical_conditions": med if isinstance(med, list) else [],
                "target_calories":    target_cal,
                "cal_loss":           tdee_data.get("cal_loss", 0),
                "cal_maintenance":    tdee_data.get("cal_maintenance", 0),
                "cal_gain":           tdee_data.get("cal_gain", 0),
            }
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
            
            # ── Normalization: Extract and align keys from incoming payload data ──
            payload_weight = data.get("weight") if data.get("weight") is not None else data.get("weight_kg")
            payload_height = data.get("height") if data.get("height") is not None else data.get("height_cm")
            payload_age    = data.get("age")
            payload_gender = data.get("gender")
            payload_activity = data.get("activity_level") or level
            payload_goal = data.get("goal") or goal
            payload_diet = data.get("diet_preference")
            payload_injuries = data.get("injuries") or injuries
            payload_med = data.get("medical_conditions")

            # Priority 1: Manual payload values (with fallback if null/empty). Priority 2: DB profile values. Priority 3: safe defaults.
            w_val = _resolve_numeric(payload_weight, profile.weight if profile else None, None)
            h_val = _resolve_numeric(payload_height, profile.height if profile else None, None)
            a_val = _resolve_numeric(payload_age, profile.age if profile else None, None)
            
            inj = _resolve_list(payload_injuries, profile.injuries if profile else None, [])
            med = _resolve_list(payload_med, profile.medical_conditions if profile else None, [])
            g_val = _resolve_string(payload_goal, profile.goal if profile else None, None)
            d_val = _resolve_string(payload_diet, profile.diet_preference if profile else None, None)

            # Resolve gender and activity level
            req_gender = _resolve_string(payload_gender, profile.gender.value if profile and profile.gender else None, "male")
            req_activity = _resolve_string(payload_activity, profile.activity_level.value if profile and profile.activity_level else None, "SEDENTARY")
            
            gender_str = req_gender.lower()
            activity_str = req_activity.upper()

            merged_context = {
                "full_name":          profile.full_name if profile else "User",
                "age":                a_val,
                "gender":             gender_str.upper(),
                "weight_kg":          w_val,
                "height_cm":          h_val,
                "goal":               g_val,
                "activity_level":     activity_str,
                "diet_preference":    d_val,
                "injuries":           inj if isinstance(inj, list) else [],
                "medical_conditions": med if isinstance(med, list) else [],
            }
        finally:
            db.close()

        # Flexibility: if the frontend sends a specific message, use it. Otherwise, build one.
        user_input = data.get("message", "").strip() if data.get("message") else ""
        if user_input:
            from app.utils.logger import logger as _log
            _log.info(f"📩 [generate_workout] Using user message: '{user_input}'")
        else:
            parts = ["Create a structured workout plan"]
            if goal:     parts.append(f"for {goal}")
            if level:    parts.append(f"at a {level} fitness level")
            if duration: parts.append(f"for a duration of {duration}")
            user_input = " ".join(parts) + "."
            from app.utils.logger import logger as _log
            _log.info(f"📩 [generate_workout] No message in body — auto-built: '{user_input}'")
        
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

        # ── Step 1: Load DB profile (for supplemental info) ───────────────────
        db = SessionLocal()
        try:
            profile = ProfileService.get_profile(db, user_id)
            
            # ── Normalization: Extract and align keys from incoming payload data ──
            payload_weight = data.get("weight") if data.get("weight") is not None else data.get("weight_kg")
            payload_height = data.get("height") if data.get("height") is not None else data.get("height_cm")
            payload_age    = data.get("age")
            payload_gender = data.get("gender")
            payload_activity = data.get("activity_level")
            payload_goal = data.get("goal")
            payload_diet = data.get("diet_preference") or diet_type
            payload_injuries = data.get("injuries")
            payload_med = data.get("medical_conditions")

            # Priority 1: Manual payload values (with fallback if null/empty). Priority 2: DB profile values. Priority 3: safe defaults.
            w_val = _resolve_numeric(payload_weight, profile.weight if profile else None, 0.0)
            h_val = _resolve_numeric(payload_height, profile.height if profile else None, 0.0)
            a_val = _resolve_numeric(payload_age, profile.age if profile else None, 0.0)
            
            inj = _resolve_list(payload_injuries, profile.injuries if profile else None, [])
            med = _resolve_list(payload_med, profile.medical_conditions if profile else None, [])
            g_val = _resolve_string(payload_goal, profile.goal if profile else None, None)
            d_val = _resolve_string(payload_diet, profile.diet_preference if profile else None, None)

            # Resolve gender and activity level
            req_gender = _resolve_string(payload_gender, profile.gender.value if profile and profile.gender else None, "male")
            req_activity = _resolve_string(payload_activity, profile.activity_level.value if profile and profile.activity_level else None, "SEDENTARY")
            
            gender_str = req_gender.lower()
            activity_str = req_activity.upper()
        finally:
            db.close()

        # ── Step 3: Compute TDEE unconditionally ──────────────────────────────
        # Always runs — even if there is no DB profile — as long as payload has the numbers.
        _multipliers = {
            "SEDENTARY": 1.2, "LIGHTLY_ACTIVE": 1.375,
            "MODERATELY_ACTIVE": 1.55, "VERY_ACTIVE": 1.725, "EXTRA_ACTIVE": 1.9,
        }
        try:
            if w_val > 0 and h_val > 0 and a_val > 0:
                bmr  = (10*w_val + 6.25*h_val - 5*a_val + 5) if gender_str in ("male", "m") else (10*w_val + 6.25*h_val - 5*a_val - 161)
                tdee = bmr * _multipliers.get(activity_str, 1.2)
                cal_loss        = round(max(bmr, tdee * 0.80))   # −20 % deficit, never below BMR
                cal_maintenance = round(tdee)
                cal_gain        = round(tdee * 1.15)             # +15 % surplus
            else:
                bmr = tdee = cal_loss = cal_maintenance = cal_gain = 0
        except (ValueError, TypeError):
            bmr = tdee = cal_loss = cal_maintenance = cal_gain = 0

        # Goal-based calorie selection
        current_goal = g_val or "General Fitness"
        target_cal = cal_maintenance
        if current_goal:
            g_lower = current_goal.lower()
            if "loss" in g_lower or "lose" in g_lower or "decrease" in g_lower:
                target_cal = cal_loss
            elif "gain" in g_lower or "bulk" in g_lower or "increase" in g_lower:
                target_cal = cal_gain

        merged_context = {
            "full_name":          profile.full_name if profile else "User",
            "age":                a_val if a_val > 0 else None,
            "gender":             gender_str.upper(),
            "weight_kg":          w_val if w_val > 0 else None,
            "height_cm":          h_val if h_val > 0 else None,
            "goal":               current_goal,
            "activity_level":     activity_str,
            "diet_preference":    d_val,
            "injuries":           inj if isinstance(inj, list) else [],
            "medical_conditions": med if isinstance(med, list) else [],
            "allergies":          allergies,
            "cal_loss":           cal_loss,
            "cal_maintenance":    cal_maintenance,
            "cal_gain":           cal_gain,
            "target_calories":    target_cal,
        }
        
        # Flexibility: if the frontend sends a specific message, use it. Otherwise, build one.
        user_input = data.get("message", "").strip() if data.get("message") else ""
        if user_input:
            from app.utils.logger import logger as _log
            _log.info(f"📩 [generate_diet] Using user message: '{user_input}'")
        else:
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
            from app.utils.logger import logger as _log
            _log.info(f"📩 [generate_diet] No message in body — auto-built: '{user_input}'")
        
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
                    "structured_data": {"Nutrition": output},
                    "intents": ["nutrition"]
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