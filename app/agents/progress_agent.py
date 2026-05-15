import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.state import AgentState
from app.core.redis_client import redis_manager
from app.utils.logger import logger


# ── Output Schema ──────────────────────────────────────────────────────────────

class ProgressInsight(BaseModel):
    category: str = Field(description="Category: 'Nutrition', 'Workout', 'Consistency', or 'Goal'.")
    observation: str = Field(description="A single, specific insight (1-2 sentences).")

class ProgressAnalysis(BaseModel):
    final_answer: str = Field(
        description=(
            "A warm, personalized 3-4 sentence progress summary. "
            "Reference the user's actual goal and what they've been working on. "
            "Be motivational and specific — NOT generic."
        )
    )
    insights: List[ProgressInsight] = Field(
        default=[],
        description="2-4 specific insights derived from the user's actual history."
    )
    goal_alignment: str = Field(
        default="",
        description="1 sentence: how well the user's recent activity aligns with their stated goal."
    )
    next_steps: List[str] = Field(
        default=[],
        description="2-3 short, actionable next steps tailored to the user's goal and recent history."
    )
    sessions_analyzed: int = Field(
        default=0, description="Total number of AI interaction sessions reviewed."
    )
    nutrition_sessions: int = Field(
        default=0, description="Number of sessions where the user engaged with nutrition/diet topics."
    )
    workout_sessions: int = Field(
        default=0, description="Number of sessions where the user engaged with workout/training topics."
    )


# ── Progress Agent ─────────────────────────────────────────────────────────────

class ProgressAgent:
    """
    Step 7.5: PROGRESS AGENT
    Analyzes the user's journey using:
      - Redis chat history (structured meal/workout data from every past AI turn)
      - User Profile (SQL) — baseline weight, goal, diet preference
    No separate Progress DB required.
    """

    def __init__(self):
        from app.core.config import settings
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=settings.OPENAI_API_KEY,
        ).with_structured_output(ProgressAnalysis, method="function_calling")

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Progress Analyst for 'Agentic AI Gym'.
Your role is to review the user's fitness journey and provide a personalized progress report.

USER PROFILE:
- Name: {full_name}
- Goal: {goal}
- Weight: {weight_kg} kg | Height: {height_cm} cm | Age: {age}
- Diet Preference: {diet_preference}
- Activity Level: {activity_level}
- Medical/Injuries: {injuries}

RULES:
- Only reference topics the user has actually discussed (do NOT invent data).
- If history is sparse, acknowledge it warmly and encourage the user to keep going.
- Be specific: reference their actual goal (e.g. "weight loss", "muscle gain").
- next_steps must be actionable and directly tied to their goal and history.
- Do NOT produce generic gym advice — every sentence should feel personal."""),
            ("human", """CONVERSATION SUMMARY: {conversation_summary}

JOURNEY JOURNAL (parsed from {sessions_analyzed} sessions):
{journey_journal}

The user just asked: "{query}"

Generate a full progress report.""")
        ])

    # ── Internal: Parse Redis History ──────────────────────────────────────────

    def _parse_redis_history(self, user_id: str) -> Dict[str, Any]:
        """
        Pulls and parses the user's full chat history from Redis.
        Extracts structured meal/workout data stored by MemoryManager.
        Returns a structured journey summary.
        """
        redis_key = f"chat_history:{user_id}"
        journal_lines = []
        nutrition_sessions = 0
        workout_sessions = 0
        total_sessions = 0

        if not redis_manager.is_available():
            logger.warning("⚠️ [ProgressAgent] Redis unavailable — proceeding with profile data only.")
            return {
                "journal": "No session history available (Redis offline).",
                "sessions_analyzed": 0,
                "nutrition_sessions": 0,
                "workout_sessions": 0,
            }

        try:
            raw_messages = redis_manager.client.lrange(redis_key, 0, -1)
        except Exception as e:
            logger.error(f"❌ [ProgressAgent] Redis read error: {e}")
            return {
                "journal": "Could not retrieve session history.",
                "sessions_analyzed": 0,
                "nutrition_sessions": 0,
                "workout_sessions": 0,
            }

        session_idx = 0
        for raw in raw_messages:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("type") != "ai":
                continue  # Only analyze AI turns (which hold structured_data)

            session_idx += 1
            total_sessions += 1
            intents = msg.get("intents", [])
            structured = msg.get("structured_data", {})

            line_parts = [f"Session {session_idx}:"]

            # --- Nutrition data ---
            nutrition_data = structured.get("nutrition", {})
            if nutrition_data or "nutrition" in intents:
                nutrition_sessions += 1
                meals = nutrition_data.get("meals", [])
                daily = nutrition_data.get("daily_totals", {})
                if meals:
                    meal_names = [m.get("name", "") for m in meals[:3] if m.get("name")]
                    line_parts.append(
                        f"Nutrition plan discussed — {len(meals)} meals "
                        f"(e.g. {', '.join(meal_names)}). "
                        f"Daily target: {daily.get('calories', '?')} kcal, "
                        f"{daily.get('protein', '?')} protein."
                    )
                elif "nutrition" in intents:
                    line_parts.append("Nutrition/diet question discussed.")

            # --- Workout data ---
            training_data = structured.get("training", {})
            if training_data or "workout" in intents:
                workout_sessions += 1
                workouts = training_data.get("workout", [])
                if workouts:
                    exercise_names = [w.get("exercise", w.get("name", "")) for w in workouts[:3] if w]
                    line_parts.append(
                        f"Workout plan discussed — {len(workouts)} exercises "
                        f"(e.g. {', '.join(filter(None, exercise_names))})."
                    )
                elif "workout" in intents:
                    line_parts.append("Workout/training question discussed.")

            # --- Vision / image analysis ---
            vision_data = structured.get("vision", {})
            if vision_data or "image" in intents:
                food_desc = vision_data.get("food_description", "")
                line_parts.append(
                    f"Image analyzed{f': {food_desc[:80]}' if food_desc else ' (food/exercise photo)'}."
                )

            # --- General domain question ---
            if "general" in intents and not any(k in intents for k in ["nutrition", "workout", "image"]):
                line_parts.append("General fitness/science question discussed.")

            if len(line_parts) > 1:  # Only add if there's meaningful content
                journal_lines.append(" ".join(line_parts))

        journal_text = (
            "\n".join(journal_lines)
            if journal_lines
            else "No detailed session history found yet — user may be new."
        )

        logger.info(
            f"📊 [ProgressAgent] Parsed {total_sessions} sessions for '{user_id}' | "
            f"Nutrition: {nutrition_sessions} | Workout: {workout_sessions}"
        )

        return {
            "journal": journal_text,
            "sessions_analyzed": total_sessions,
            "nutrition_sessions": nutrition_sessions,
            "workout_sessions": workout_sessions,
        }

    # ── Main Run ───────────────────────────────────────────────────────────────

    async def run(self, state: AgentState) -> Dict[str, Any]:
        query = state["messages"][-1].content
        user_id = state.get("user_id", "default")
        user_context = state.get("user_context", {}) or {}
        summary = state.get("conversation_summary", "No previous context.")

        # 1. Parse Redis history
        history = self._parse_redis_history(user_id)

        # 2. Extract user profile fields
        full_name = user_context.get("full_name") or "there"
        goal = user_context.get("goal") or "General Fitness"
        weight_kg = user_context.get("weight_kg") or "Unknown"
        height_cm = user_context.get("height_cm") or "Unknown"
        age = user_context.get("age") or "Unknown"
        diet_pref = user_context.get("diet_preference") or "No preference"
        activity = user_context.get("activity_level") or "Unknown"
        injuries_raw = user_context.get("injuries", [])
        injuries = ", ".join(injuries_raw) if isinstance(injuries_raw, list) and injuries_raw else "None"

        logger.info(
            f"📈 [ProgressAgent] Running for '{user_id}' | Goal: '{goal}' | "
            f"Sessions: {history['sessions_analyzed']}"
        )

        # 3. Build prompt and invoke LLM
        chain = self.prompt | self.llm
        analysis: ProgressAnalysis = await chain.ainvoke({
            "full_name": full_name,
            "goal": goal,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "age": age,
            "diet_preference": diet_pref,
            "activity_level": activity,
            "injuries": injuries,
            "conversation_summary": summary,
            "sessions_analyzed": history["sessions_analyzed"],
            "journey_journal": history["journal"],
            "query": query,
        })

        # 4. Enrich with counts from Redis parsing (LLM may not see exact numbers)
        analysis_dict = analysis.model_dump()
        analysis_dict["sessions_analyzed"] = history["sessions_analyzed"]
        analysis_dict["nutrition_sessions"] = history["nutrition_sessions"]
        analysis_dict["workout_sessions"] = history["workout_sessions"]

        logger.info(
            f"✅ [ProgressAgent] Report generated | "
            f"Insights: {len(analysis_dict.get('insights', []))} | "
            f"Next steps: {len(analysis_dict.get('next_steps', []))}"
        )

        return {
            "specialist_results": {
                "progress": {
                    "answer": analysis_dict["final_answer"],
                    "insights": analysis_dict.get("insights", []),
                    "goal_alignment": analysis_dict.get("goal_alignment", ""),
                    "next_steps": analysis_dict.get("next_steps", []),
                    "sessions_analyzed": analysis_dict["sessions_analyzed"],
                    "nutrition_sessions": analysis_dict["nutrition_sessions"],
                    "workout_sessions": analysis_dict["workout_sessions"],
                    "status": "success",
                }
            }
        }
