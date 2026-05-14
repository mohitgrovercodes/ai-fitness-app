import json
from typing import Dict, Any, List
from app.core.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from app.utils.logger import logger
from app.core.redis_client import redis_manager

class MemoryManager:
    """
    Production Memory Manager with Optimized Redis Persistence.
    - Stores full history in Redis using RPUSH.
    - Summarizes history when messages exceed 20.
    - Maintains a 2-line summary + last 6 messages in active context.
    """
    def __init__(self, summary_trigger: int = 20, keep_recent: int = 6):
        from app.core.config import settings
        self.summary_trigger = summary_trigger
        self.keep_recent = keep_recent
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0, 
            api_key=settings.OPENAI_API_KEY
        )
        
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are managing the conversation memory for an AI Fitness Coach.
Your task is to summarize the conversation history provided below into EXACTLY TWO LINES.
Include key details: user goals, injuries, specific foods discussed, and current workout phase.

EXISTING SUMMARY: {existing_summary}"""),
            ("human", "NEW MESSAGES TO SUMMARIZE:\n{messages_str}")
        ])

    async def run(self, state: AgentState) -> Dict[str, Any]:
        print(f"🐛 [DEBUG MemoryAgent] Keys in state: {list(state.keys())}")
        messages = list(state.get('messages', []))
        user_id = state.get("user_id", "default_user") 
        print(f"🐛 [DEBUG MemoryAgent] Extracted user_id: {user_id}") 
        redis_key = f"chat_history:{user_id}"
        summary_key = f"chat_summary:{user_id}"

        # 1. Save the new messages to Redis
        # A single chat turn adds exactly 2 messages: 1 Human, 1 AI. We save both.
        if redis_manager.is_available() and messages:
            try:
                msgs_to_save = messages[-2:] if len(messages) >= 2 else messages
                for msg in msgs_to_save:
                    data_to_save = {
                        "type": msg.type, 
                        "content": msg.content
                    }
                    
                    # If this is the AI's final message of the current turn, attach the full JSON payload
                    if msg.type == "ai" and msg == messages[-1]:
                        data_to_save["structured_data"] = state.get("specialist_results", {})
                        data_to_save["intents"] = state.get("intent", [])
                        
                    serialized_msg = json.dumps(data_to_save)
                    redis_manager.client.rpush(redis_key, serialized_msg)
            except Exception as e:
                logger.error(f"❌ [Memory] Redis Save Error: {e}")

        # 2. Check for Summary Trigger
        # Use Redis LIST count (not LangGraph RAM count) so trigger survives server restarts.
        if redis_manager.is_available():
            total_messages = redis_manager.client.llen(redis_key)
        else:
            total_messages = len(messages)  # fallback if Redis is down
        existing_summary = state.get("conversation_summary", "")
        
        # Load summary from Redis if not in state
        if not existing_summary and redis_manager.is_available():
            existing_summary = redis_manager.client.get(summary_key) or ""

        # Logic: If messages > 20, we summarize and prune
        if total_messages > self.summary_trigger:
            logger.info(f"🧠 [Memory] Threshold reached ({total_messages}). Summarizing...")
            
            # Keep only the last 6 messages
            msgs_to_summarize = messages[:-self.keep_recent]
            active_messages = messages[-self.keep_recent:]
            
            msg_str = "\n".join([f"{m.type}: {m.content}" for m in msgs_to_summarize])
            
            chain = self.summary_prompt | self.llm
            res = await chain.ainvoke({
                "existing_summary": existing_summary,
                "messages_str": msg_str
            })
            
            new_summary = res.content.strip()
            # Force exactly 2 lines if possible, or at least very short
            logger.info(f"🧠 [Memory] New 2-line summary created: {new_summary[:50]}...")
            
            if redis_manager.is_available():
                redis_manager.client.set(summary_key, new_summary)
            
            return {
                "conversation_summary": new_summary,
                "messages": {"type": "replace", "messages": active_messages}
            }

        return {
            "conversation_summary": existing_summary
        }
