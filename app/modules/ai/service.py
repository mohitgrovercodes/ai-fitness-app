
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
    if isinstance(val, (int, float)) and val == 0:
        return True

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
            parts = []
            for item in val.split(","):
                s = item.strip()
                if s and s.lower() not in ("null", "none", "undefined"):
                    parts.append(s)
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
def _extract_specialist_text(specialist_results: dict) -> str:
    """Dynamically extract first text value from any specialist's response."""
    text_parts = []
    for agent_name, data in specialist_results.items():
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, str) and len(val) > 10:
                    text_parts.append(val)
                    break
    return " | ".join(text_parts) if text_parts else ""
async def _detect_and_translate_query(user_input: str) -> tuple[str, str]:
    if not user_input:
        return "english", ""

    from langchain_openai import ChatOpenAI
    from app.core.config import settings
    from pydantic import BaseModel, Field
    
    class TranslationResult(BaseModel):
        detected_language: str = Field(description="Must be 'english', 'hindi', or 'hinglish'")
        english_translation: str = Field(description="The English translation of the query")
        
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0, 
            api_key=settings.OPENAI_API_KEY
        ).with_structured_output(TranslationResult, method="function_calling")
        
        prompt = f"""Analyze this user request from a fitness app:
"{user_input}"

Detect the language ('english', 'hindi' for Devanagari script, or 'hinglish' for Hindi grammar/vocabulary in Roman alphabet) and translate it to clean English for fitness planning.

Return the result matching the required schema."""
        
        res = await llm.ainvoke(prompt)
        return res.detected_language, res.english_translation
    except Exception:
        return "english", user_input

async def _translate_structured_output(output: dict, target_lang: str) -> dict:
    if not output or not target_lang or target_lang.strip().lower() == "english":
        return output
        
    from langchain_openai import ChatOpenAI
    from app.core.config import settings
    import json
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)
        
        prompt = f"""You are a dynamic translation engine for a fitness application.
Your task is to translate all user-facing natural language text fields inside the provided JSON object into '{target_lang.upper()}'.

STRICT RULES:
1. Maintain the EXACT same JSON keys, array sizes, and structure. Do NOT add or remove any keys.
2. Maintain all numeric values, sets, reps, and URLs exactly as-is (e.g. keep gif_path and image_path URLs unmodified).
3. If the target language is 'HINDI', translate into high-quality Devanagari script (e.g., "नमस्ते", "वर्कआउट").
4. If the target language is 'HINGLISH', translate into natural, social-media-style transliterated Hindi/English blend written in the Roman/Latin alphabet (e.g., "Aapka plan ready hai").
5. Only translate natural language fields (such as 'summary', 'tip', 'description', 'benefit', 'day', 'type', 'name' if applicable). Keep technical labels, IDs, or schemas identical.

JSON OBJECT TO TRANSLATE:
{json.dumps(output, ensure_ascii=False)}

TRANSLATED JSON:"""
        
        res = await llm.ainvoke(prompt)
        content = res.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception as e:
        from app.utils.logger import logger
        logger.error(f"Error during structured translation of direct API output: {e}")
        return output

async def _translate_plain_text(text: str, target_lang: str) -> str:
    if not text or not target_lang or target_lang.strip().lower() == "english":
        return text
        
    from langchain_openai import ChatOpenAI
    from app.core.config import settings
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)
        
        prompt = f"""You are a dynamic translation engine for a fitness application.
Translate the following fitness-related explanation or text into '{target_lang.upper()}'.

STRICT RULES:
1. If the target language is 'HINDI', translate into encouraging, natural Hindi in Devanagari script. Keep basic fitness words phonetically written in Devanagari (e.g., write "वर्कआउट", "कैलरी", "मसल" rather than complex formal translation).
2. If the target language is 'HINGLISH', translate into natural, social-media-style transliterated Hindi/English blend written in the Roman alphabet (e.g. "Aapka chest workout ready hai").
3. Keep the translation professional, accurate, and motivating.
4. Return ONLY the translated text, with no extra annotations, prefixes, or markdown blocks.

TEXT TO TRANSLATE:
{text}

TRANSLATION:"""
        
        res = await llm.ainvoke(prompt)
        return res.content.strip()
    except Exception as e:
        from app.utils.logger import logger
        logger.error(f"Error during plain text translation: {e}")
        return text

async def _resolve_goal_category(goal_str: str) -> str:
    if not goal_str:
        return "maintenance"

    from langchain_openai import ChatOpenAI
    from app.core.config import settings
    from pydantic import BaseModel, Field

    class GoalCategoryResult(BaseModel):
        category: str = Field(description="Must be 'loss', 'gain', or 'maintenance'")

    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=settings.OPENAI_API_KEY
        ).with_structured_output(GoalCategoryResult, method="function_calling")

        prompt = f"""Classify this fitness goal string:
"{goal_str}"

Map it to one of these categories:
- 'loss': If it represents losing weight, fat loss, cutting, body recomposition (loss focus), decreasing weight, or leaning out.
- 'gain': If it represents building muscle, gaining weight, bulking, increasing strength/mass, or muscle hypertrophy.
- 'maintenance': If it represents maintaining weight, general health, flexibility, recovery, or endurance.

Return the result matching the required schema."""

        res = await llm.ainvoke(prompt)
        return res.category.strip().lower()
    except Exception:
        return "maintenance"


async def _build_calorie_targets(
    w_val,
    h_val,
    a_val,
    gender_str: str,
    activity_str: str,
    goal_str: str,
) -> dict:
    """
    Single source of truth for TDEE-based calorie targets used by every
    AI endpoint (/chat, /generate-workout, /generate-diet).

    Always delegates the arithmetic to `_compute_tdee` (Mifflin-St Jeor)
    in app/agents/base.py — no duplicated inline math. Goal classification
    is handled by `_resolve_goal_category` so that "weight loss", "fat loss",
    "cutting", etc. all map consistently across endpoints.

    Returns a dict with exactly the four keys downstream agents expect:
      - target_calories   (loss / maintenance / gain — selected by goal)
      - cal_loss          (−20 % deficit, never below BMR)
      - cal_maintenance   (TDEE)
      - cal_gain          (+15 % surplus)

    If biometrics are missing/invalid, _compute_tdee returns all zeros and
    target_calories defaults to 0 — same behavior across all endpoints.
    """
    tdee_data = _compute_tdee(w_val, h_val, a_val, gender_str, activity_str)

    # If TDEE is zero (incomplete profile), all targets are zero — no LLM call needed.
    if not tdee_data.get("tdee"):
        return {
            "target_calories": 0,
            "cal_loss":        tdee_data.get("cal_loss", 0),
            "cal_maintenance": tdee_data.get("cal_maintenance", 0),
            "cal_gain":        tdee_data.get("cal_gain", 0),
        }

    # Goal-based selection. Defaults to maintenance for empty / unknown goals.
    if goal_str:
        goal_cat = await _resolve_goal_category(goal_str)
    else:
        goal_cat = "maintenance"

    if goal_cat == "loss":
        target_cal = tdee_data["cal_loss"]
    elif goal_cat == "gain":
        target_cal = tdee_data["cal_gain"]
    else:
        target_cal = tdee_data["cal_maintenance"]

    return {
        "target_calories": target_cal,
        "cal_loss":        tdee_data["cal_loss"],
        "cal_maintenance": tdee_data["cal_maintenance"],
        "cal_gain":        tdee_data["cal_gain"],
    }


class AIService:

    @staticmethod
    async def initialize_request(
        user_id: str,
        user_input: str,
        payload_context: dict = None,
        image_bytes: bytes = None
    ) -> dict:
        """
        Unified Centralized Request Intake Layer.
        Consolidates profile loading, manual payload merges, dynamic TDEE calculations,
        and runs language detection EXACTLY ONCE at the application entry boundary.
        """
        from app.core.sql_db import SessionLocal
        from app.modules.profile.service import ProfileService

        db = SessionLocal()
        try:
            profile = ProfileService.get_profile(db, user_id)
            raw_context = payload_context or {}
            
            # Extract biometric and context details safely
            payload_weight = raw_context.get("weight") if raw_context.get("weight") is not None else raw_context.get("weight_kg")
            payload_height = raw_context.get("height") if raw_context.get("height") is not None else raw_context.get("height_cm")
            payload_age    = raw_context.get("age")
            payload_gender = raw_context.get("gender")
            payload_activity = raw_context.get("activity_level") or raw_context.get("level")
            payload_goal = raw_context.get("goal")
            payload_diet = raw_context.get("diet_preference") or raw_context.get("diet_type")
            payload_injuries = raw_context.get("injuries")
            payload_med = raw_context.get("medical_conditions")
            payload_allergies = raw_context.get("allergies")
            payload_language = raw_context.get("language") or raw_context.get("preferred_language")

            w_val = _resolve_numeric(payload_weight, profile.weight if profile else None, None)
            h_val = _resolve_numeric(payload_height, profile.height if profile else None, None)
            a_val = _resolve_numeric(payload_age, profile.age if profile else None, None)
            
            inj = _resolve_list(payload_injuries, profile.injuries if profile else None, [])
            med = _resolve_list(payload_med, profile.medical_conditions if profile else None, [])
            allg = _resolve_list(payload_allergies, None, [])
            g_val = _resolve_string(payload_goal, profile.goal if profile else None, None)
            d_val = _resolve_string(payload_diet, profile.diet_preference if profile else None, None)

            req_gender = _resolve_string(payload_gender, profile.gender.value if profile and profile.gender else None, "male")
            req_activity = _resolve_string(payload_activity, profile.activity_level.value if profile and profile.activity_level else None, "SEDENTARY")
            
            gender_str = req_gender.lower()
            activity_str = req_activity.upper()

            # ── Unified TDEE / Calorie Targets (single source of truth) ──
            cal_ctx = await _build_calorie_targets(
                w_val, h_val, a_val, gender_str, activity_str, g_val or ""
            )

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
                "allergies":          allg if isinstance(allg, list) else [],
                **cal_ctx,
                "language":           _resolve_string(payload_language, None, None)
            }
        finally:
            db.close()

        # ── Multilingual Input Resolution (Exactly Once) ──
        detected_lang, translated_input = await _detect_and_translate_query(user_input)
        target_lang = (merged_context.get("language") or detected_lang).strip().lower()
        
        # Override language in context with the resolved language
        merged_context["language"] = target_lang
        
        original_query = user_input
        translated_query = None
        if target_lang != "english" and translated_input:
            translated_query = translated_input

        return {
            "user_context":     merged_context,
            "language":         target_lang,
            "original_query":   original_query,
            "translated_query": translated_query,
            "image_bytes":      image_bytes
        }



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
            payload_language = raw_context.get("language") or raw_context.get("preferred_language")

            # Priority 1: Manual payload values. Priority 2: DB profile values. Priority 3: safe defaults.
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

            # ── Unified TDEE / Calorie Targets (single source of truth) ──
            cal_ctx = await _build_calorie_targets(
                w_val, h_val, a_val, gender_str, activity_str, g_val or ""
            )

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
                **cal_ctx,
                "language":           _resolve_string(payload_language, None, None)
            }
        finally:
            db.close()
            
        # ── Fetch Existing Summary from Redis (Added) ──
        from app.core.redis_client import redis_manager
        existing_summary = ""
        try:
            if redis_manager.is_available():
                summary_bytes = redis_manager.client.get(f"chat_summary:{user_id}")
                if summary_bytes:
                    existing_summary = summary_bytes.decode("utf-8")
        except Exception as e:
            from app.utils.logger import logger
            logger.error(f"Error fetching summary: {e}")

        # Initial state for the graph
        from langchain_core.messages import HumanMessage
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": merged_context,
            "conversation_summary": existing_summary, # INJECTED SUMMARY
            "user_id": user_id,
            "specialist_results": {"__clear__": True}, 
            "image_bytes": image_bytes,
            "language": None,
            "original_query": None,
            "translated_query": None
        }
            
        # Run the graph
        graph = get_graph()
        final_state = await graph.ainvoke(initial_state, config=config)
        
        # Extract the last AI message
        last_msg = final_state["messages"][-1]
        
        # Extract media and structured data from specialist results
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
        
        if gifs: out_data["exercise_gifs"] = gifs
        if imgs: out_data["exercise_images"] = imgs
        if daily_totals: out_data["daily_totals"] = daily_totals
        if per_day_totals: out_data["per_day_totals"] = per_day_totals
            
        # ── Multi-Language Redis Saving & Summarization (Moved to service.py) ──
        import json
        try:
            if redis_manager.is_available():
                redis_key = f"chat_history:{user_id}"
                summary_key = f"chat_summary:{user_id}"
                index_key = f"chat_summary_index:{user_id}"
                
                original_query = final_state.get("original_query") or user_input
                translated_query = final_state.get("translated_query")
                
                # 1. Save Original Language Turn
                human_msg_original = {"type": "human", "content": original_query}
                ai_msg_translated = {
                    "type": "ai",
                    "content": last_msg.content,  
                    # "structured_data": final_state.get("specialist_results", {}), 
                    # "intents": final_state.get("intent", [])
                }
                redis_manager.client.rpush(redis_key, json.dumps(human_msg_original))
                redis_manager.client.rpush(redis_key, json.dumps(ai_msg_translated))
                
                # 2. Save Translated Language Turn (If translation happened)
                if translated_query and translated_query != original_query:
                    human_msg_translated = {"type": "human", "content": translated_query}
                    ai_msg_original = {
                        "type": "ai",
                        "content": _extract_specialist_text(final_state.get("specialist_results", {})),
                        "structured_data": final_state.get("specialist_results", {}),
                        "intents": final_state.get("intent", [])
                    }
                    redis_manager.client.rpush(redis_key, json.dumps(human_msg_translated))
                    redis_manager.client.rpush(redis_key, json.dumps(ai_msg_original))

                # 3. Summarization Trigger (Runs every 20 messages)
                total_messages = redis_manager.client.llen(redis_key)
                last_index_str = redis_manager.client.get(index_key)
                last_index = int(last_index_str) if last_index_str else 0

                if total_messages - last_index >= 20:
                    from app.utils.logger import logger
                    logger.info(f"🧠 [Service Memory] Threshold reached. Summarizing...")
                    
                    raw_msgs = redis_manager.client.lrange(redis_key, last_index, total_messages - 1)
                    parsed_msgs = []
                    for raw in raw_msgs:
                        try:
                            parsed = json.loads(raw)
                            parsed_msgs.append(f"{parsed.get('type', '')}: {parsed.get('content', '')}")
                        except: pass
                    
                    msg_str = "\n".join(parsed_msgs)
                    if msg_str.strip():
                        from langchain_openai import ChatOpenAI
                        from langchain_core.prompts import ChatPromptTemplate
                        from app.core.config import settings

                        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY)
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", "Summarize the conversation history into EXACTLY TWO LINES. Include user goals, injuries, specific foods discussed. EXISTING SUMMARY: {existing_summary}"),
                            ("human", "NEW MESSAGES:\n{messages_str}")
                        ])
                        
                        chain = prompt | llm
                        res = await chain.ainvoke({"existing_summary": existing_summary, "messages_str": msg_str})
                        
                        new_summary = res.content.strip()
                        redis_manager.client.set(summary_key, new_summary)
                        redis_manager.client.set(index_key, total_messages)
                        logger.info(f"🧠 [Service Memory] New summary created!")
                    
        except Exception as e:
            from app.utils.logger import logger
            logger.error(f"Redis save/summary error in chat API: {e}")
            
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

            # ── Unified TDEE / Calorie Targets (single source of truth) ──
            cal_ctx = await _build_calorie_targets(
                w_val, h_val, a_val, gender_str, activity_str, g_val or ""
            )

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
                **cal_ctx,
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
        
        # ── Multilingual Input Resolution ──
        payload_language = data.get("language") or data.get("preferred_language")
        detected_lang, translated_input = await _detect_and_translate_query(user_input)
        target_lang = (payload_language or detected_lang).strip().lower()
        
        original_workout_query = user_input
        if target_lang != "english" and translated_input:
            from app.utils.logger import logger as _log
            _log.info(f"🌐 [generate_workout] Translating Hinglish/Hindi query '{user_input}' to English: '{translated_input}'")
            user_input = translated_input

        state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": merged_context,
            "conversation_summary": "Direct API Generation Request",
            "user_id": user_id
        }
        result = await TrainingAgent().run(state)
        output = result.get("specialist_results", {}).get("training", {})
        ai_generated_msg=output
        # ── Multilingual Output Rendering ──
        if target_lang != "english":
            from app.utils.logger import logger as _log
            _log.info(f"🌐 [generate_workout] Rendering structured plan output in target language: {target_lang}")
            output = await _translate_structured_output(output, target_lang)

        import json
        from app.core.redis_client import redis_manager
        try:
            if redis_manager.is_available():
                redis_key = f"chat_history:{user_id}"
                human_msg = {"type": "human", "content": original_workout_query}
                ai_msg = {
                    "type": "ai",
                    "structured_data": {"training": output},
                    "intents": ["workout"]
                }
                human_msg_translated = {"type": "human", "content": translated_input}
                ai_msg_orignal = {
                    "type": "ai",
                    "structured_data": {"training": ai_generated_msg},
                    "intents": ["workout"]
                }
                redis_manager.client.rpush(redis_key, json.dumps(human_msg))
                redis_manager.client.rpush(redis_key, json.dumps(ai_msg))
                redis_manager.client.rpush(redis_key, json.dumps(human_msg_translated))
                redis_manager.client.rpush(redis_key, json.dumps(ai_msg_orignal))
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

        # ── Unified TDEE / Calorie Targets (single source of truth) ──
        # Delegates to _compute_tdee (Mifflin-St Jeor) — identical math to /chat
        # and /generate-workout. No duplicated inline arithmetic here.
        current_goal = g_val or "General Fitness"
        cal_ctx = await _build_calorie_targets(
            w_val, h_val, a_val, gender_str, activity_str, current_goal
        )

        merged_context = {
            "full_name":          profile.full_name if profile else "User",
            "age":                a_val if a_val and a_val > 0 else None,
            "gender":             gender_str.upper(),
            "weight_kg":          w_val if w_val and w_val > 0 else None,
            "height_cm":          h_val if h_val and h_val > 0 else None,
            "goal":               current_goal,
            "activity_level":     activity_str,
            "diet_preference":    d_val,
            "injuries":           inj if isinstance(inj, list) else [],
            "medical_conditions": med if isinstance(med, list) else [],
            "allergies":          allergies,
            **cal_ctx,
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
        
        # ── Multilingual Input Resolution ──
        payload_language = data.get("language") or data.get("preferred_language")
        detected_lang, translated_input = await _detect_and_translate_query(user_input)
        target_lang = (payload_language or detected_lang).strip().lower()
        
        original_diet_query = user_input
        if target_lang != "english" and translated_input:
            from app.utils.logger import logger as _log
            _log.info(f"🌐 [generate_diet] Translating Hinglish/Hindi query '{user_input}' to English: '{translated_input}'")
            user_input = translated_input

        state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": merged_context,
            "conversation_summary": "Direct API Diet Request",
            "user_id": user_id
        }
        result = await NutritionAgent().run(state)
        output = result.get("specialist_results", {}).get("nutrition", {})
        ai_msg_translated=output

        # ── Multilingual Output Rendering ──
        if target_lang != "english":
            from app.utils.logger import logger as _log
            _log.info(f"🌐 [generate_diet] Rendering structured plan output in target language: {target_lang}")
            output = await _translate_structured_output(output, target_lang)

        import json
        from app.core.redis_client import redis_manager
        try:
            if redis_manager.is_available():
                redis_key = f"chat_history:{user_id}"
                human_msg_original = {"type": "human", "content": original_diet_query}
                ai_msg_original = {
                    "type": "ai",
                    "structured_data": {"Nutrition": output},
                    "intents": ["nutrition"]
                }
                human_msg_translated = {"type": "human", "content": user_input}
                ai_msg = {
                    "type": "ai",
                    "structured_data": {"Nutrition": ai_msg_translated},
                    "intents": ["nutrition"]
                }
                redis_manager.client.rpush(redis_key, json.dumps(human_msg_original))
                redis_manager.client.rpush(redis_key, json.dumps(ai_msg_original))
                redis_manager.client.rpush(redis_key, json.dumps(human_msg_translated))
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
        original_input=user_input
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

        # ── Multilingual Input Resolution ──
        payload_language = data.get("language") or data.get("preferred_language")
        detected_lang, translated_input = await _detect_and_translate_query(user_input)
        target_lang = (payload_language or detected_lang).strip().lower()

        if target_lang != "english" and translated_input:
            from app.utils.logger import logger as _log
            _log.info(f"🌐 [ask_domain] Translating Hinglish/Hindi query '{user_input}' to English: '{translated_input}'")
            user_input = translated_input

        state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": db_context,
            "conversation_summary": "Direct API Domain Query"
        }
        
        result = await DomainAgent().run(state)
        output = result.get("specialist_results", {}).get("domain", {})
        
        response_text = output.get("answer", "Could not answer the query.")
        ai_msg=response_text
        # ── Multilingual Output Rendering ──
        if target_lang != "english" and response_text:
            from app.utils.logger import logger as _log
            _log.info(f"🌐 [ask_domain] Translating response text to target language: {target_lang}")
            response_text = await _translate_plain_text(response_text, target_lang)

        import json
        from app.core.redis_client import redis_manager
        try:
            if redis_manager.is_available():
                redis_key = f"chat_history:{user_id}"
                human_msg_original = {"type": "human", "content": original_input}
                ai_msg_translated = {
                    "type": "ai",
                    "structured_data": {"Domain": response_text},
                    "intents": ["domain intent"]
                }
                human_msg_translated = {"type": "human", "content": user_input}
                ai_msg_original = {
                    "type": "ai",
                    "structured_data": {"Domain": ai_msg},
                    "intents": ["domain intent"]
                }
                redis_manager.client.rpush(redis_key, json.dumps(human_msg_original))
                redis_manager.client.rpush(redis_key, json.dumps(ai_msg_translated))
                redis_manager.client.rpush(redis_key, json.dumps(human_msg_translated))
                redis_manager.client.rpush(redis_key, json.dumps(ai_msg_original))
        except Exception as e:
            print(f"Redis save error in generate_workout: {e}")
        return {
            "response": response_text,
            "sources": output.get("sources", "internal_knowledge")
        }