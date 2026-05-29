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
- If the user is asking ANY question about their own personal data stored in this app, return ['general']. (For context, the user's stored profile currently contains these fields: {profile_keys})
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
            
                # Extract dynamic profile keys
        profile_keys_str = ", ".join(list(state.get("user_context", {}).keys()))
        
        # Invoke with structured output
        chain = self.prompt | self.model
        res: IntentResponse = await chain.ainvoke({
            "input": input_text,
            "summary": summary,
            "profile_keys": profile_keys_str
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

        # ── Language: Trust the centralized initialize_request() ─────────────
        # Language detection & translation already happened ONCE in AIService.initialize_request().
        # state["language"], state["original_query"], and state["translated_query"] are already set.
        # We only fall back to the Orchestrator's LLM detection if state has no language (shouldn't happen).
        resolved_lang = state.get("language") or (
            res.detected_language.strip().lower() if res.detected_language else "english"
        )
        if resolved_lang not in ("english", "hindi", "hinglish"):
            resolved_lang = "english"

        logger.info(f"🌐 [Orchestrator] Using centralized target language: {resolved_lang}")

        # Ensure downstream agents always receive the English translation in message history.
        # (initialize_request already feeds translated_query into the HumanMessage,
        #  but this is a safety net for edge cases like English queries.)
        current_messages = list(state.get("messages", []))
        if current_messages and resolved_lang != "english" and res.english_translation:
            from langchain_core.messages import HumanMessage
            logger.info(f"📝 [Orchestrator] Ensuring English message for agents: '{res.english_translation}'")
            current_messages[-1] = HumanMessage(content=res.english_translation)

        return {
            "intent": final_intents,
            "is_fitness_domain": res.is_fitness_domain,
            "next_node": route,
            "language": resolved_lang,
            # Preserve the REAL original query set by initialize_request, don't overwrite with English.
            "original_query": state.get("original_query") or last_message,
            "translated_query": state.get("translated_query") or res.english_translation,
            "messages": {"type": "replace", "messages": current_messages}
        }
