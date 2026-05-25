from typing import Dict, Any
from app.core.state import AgentState
from app.schema.orchestration import IntentResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.utils.logger import logger

class Orchestrator:
    """
    Step 3 & 4: PRODUCTION ORCHESTRATOR
    Uses Structured Output for 100% reliable intent classification.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        from app.core.config import settings
        self.model = ChatOpenAI(
            model=model_name, 
            temperature=0, 
            api_key=settings.OPENAI_API_KEY
        ).with_structured_output(IntentResponse, method="function_calling")
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Orchestrator for 'Agentic AI Gym'.
Your task is to:
1. Classify user intent based on the current message and the conversation context.
2. Detect the language of the incoming message:
   - 'english': If written in English (e.g. "I want a chest workout plan").
   - 'hindi': If written in Hindi grammar and in the native Devanagari script (e.g. "मुझे वजन कम करने के लिए डाइट चार्ट चाहिए").
   - 'hinglish': If written in Hindi grammar/vocabulary but transliterated into the Latin/Roman script (e.g. "Mera weight loss plan bana do", "kya main weight loss ke dauran rice kha sakta hu?").
3. Translate the query to English:
   - Translate all Hindi or Hinglish queries accurately into grammatically correct English so that downstream fitness/diet RAG agents can perform highly accurate search queries.
   - If the user query is already in English, keep the `english_translation` identical to the original message.

CONTEXT SUMMARY: {summary}

CATEGORIES:
- 'workout': Exercise routines, form, gym equipment, training plans.
- 'nutrition': Diet, calories, macros, meal plans, or specific foods.
- 'image': Analyzing an uploaded photo of food or exercise.
- 'general': Scientific questions, or personal profile questions (e.g., 'what is my weight', 'what is my goal').
- 'progress': Tracking weight, strength gains, or history.
- 'out_of_scope': Non-fitness topics.

MULTI-INTENT RULES (CRITICAL):
- You MUST return MULTIPLE intents when the user asks for multiple things.
- Weight loss / weight gain / body transformation goals ALWAYS require BOTH 'workout' AND 'nutrition' intents — the user needs both a workout plan AND a diet plan to achieve their goal.
- If the user says 'diet plan' AND mentions a fitness goal (lose/gain weight, get fit), return ['workout', 'nutrition'].
- If the user only asks a single specific question (e.g., 'how many calories in an apple?'), return just ['nutrition'].
- If the user asks about their own profile (name, weight, height, goal), return ['general'] so the Domain Agent can answer it. DO NOT classify as 'out_of_scope'.
- If the user says 'tell me more' or 'how many calories in that?', use the SUMMARY to determine the intent."""),
            ("human", "{input}")
        ])

    async def run(self, state: AgentState) -> Dict[str, Any]:
        last_message = state['messages'][-1].content
        summary = state.get('conversation_summary', "No previous context.")
        
        # Check if an image was uploaded
        has_image = state.get('image_bytes') is not None
        input_text = last_message
        if has_image:
            input_text += "\n\n[SYSTEM NOTE: The user has attached an image file to this request. You MUST include the 'image' intent in your classification!]"
            
        # Invoke with structured output
        chain = self.prompt | self.model
        res: IntentResponse = await chain.ainvoke({
            "input": input_text,
            "summary": summary
        })
        
        # Logic for domain checking
        route = "agent_router" if res.is_fitness_domain else "out_of_scope_handler"
        
        # Cross-Agent Intelligence:
        # LLM itself detected if this is a body transformation goal needing BOTH agents
        final_intents = res.intents
        if res.has_body_transformation_goal:
            if "nutrition" in final_intents and "workout" not in final_intents:
                final_intents.append("workout")
                logger.info("🧠 [Orchestrator] Auto-added 'workout' — body transformation goal detected by LLM.")
            elif "workout" in final_intents and "nutrition" not in final_intents:
                final_intents.append("nutrition")
                logger.info("🧠 [Orchestrator] Auto-added 'nutrition' — body transformation goal detected by LLM.")

        # Enforce strict 'image' handling to prevent generic fluff
        if "image" in final_intents and "nutrition" in final_intents:
            logger.info("🧠 [Orchestrator] Stripping 'nutrition' intent because an image is present. Vision Agent will handle the nutritional breakdown.")
            final_intents.remove("nutrition")

        # ── Multilingual Translation & Substitution ─────────────────────────
        detected_lang = res.detected_language.strip().lower() if res.detected_language else "english"
        if detected_lang not in ("english", "hindi", "hinglish"):
            detected_lang = "english"  # Fallback safety

        # Read optional override language from payload context
        payload_context = state.get("user_context", {}) or {}
        resolved_lang = payload_context.get("language") or state.get("language") or detected_lang
        resolved_lang = resolved_lang.strip().lower()

        logger.info(f"🌐 [Orchestrator] Detected query language: {detected_lang} | Resolved target language: {resolved_lang}")

        # Substitute the last message in LangGraph active memory with the translated English text
        # this ensures downstream specialized agents query their RAG in English with high precision.
        current_messages = list(state.get("messages", []))
        if current_messages and resolved_lang != "english":
            from langchain_core.messages import HumanMessage
            logger.info(f"📝 [Orchestrator] Substituting query with English translation: '{res.english_translation}'")
            current_messages[-1] = HumanMessage(content=res.english_translation)

        return {
            "intent": final_intents,
            "is_fitness_domain": res.is_fitness_domain,
            "next_node": route,
            "language": resolved_lang,
            "original_query": last_message,
            "translated_query": res.english_translation,
            "messages": {"type": "replace", "messages": current_messages}
        }
