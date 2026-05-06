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
    Production Memory Manager with Redis Persistence.
    - Stores full history in Redis.
    - Summarizes history every 10 messages.
    - Maintains the last 20 messages in active context.
    """
    def __init__(self, summary_trigger: int = 10, keep_recent: int = 20):
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
Your task is to summarize the conversation history provided below.
Include key details: goals, injuries, specific foods discussed, workouts planned.
If there is an existing summary, update it with the new information.
Keep it concise but comprehensive.

EXISTING SUMMARY: {existing_summary}"""),
            ("human", "NEW MESSAGES TO SUMMARIZE:\n{messages_str}")
        ])

    async def run(self, state: AgentState) -> Dict[str, Any]:
        messages = state.get('messages', [])
        # We assume thread_id is available in state or can be derived
        # For now, we'll use a placeholder if not found, but in production, this comes from config
        user_id = state.get("user_id", "default_user") 
        redis_key = f"chat_history:{user_id}"
        summary_key = f"chat_summary:{user_id}"

        # 1. Save latest message to Redis if not already there
        if redis_manager.is_available():
            try:
                # Store all messages as JSON strings in a Redis list
                serialized_msgs = [json.dumps({"type": m.type, "content": m.content}) for m in messages]
                # In a real scenario, we might only push the NEWEST message, 
                # but for simplicity in this node, we sync the state.
                redis_manager.client.delete(redis_key)
                if serialized_msgs:
                    redis_manager.client.rpush(redis_key, *serialized_msgs)
            except Exception as e:
                logger.error(f"❌ [Memory] Redis Save Error: {e}")

        # 2. Check for Summary Trigger (every 10 messages)
        # We track 'message_count' in state or Redis to trigger summarization
        total_messages = len(messages)
        existing_summary = state.get("conversation_summary", "")
        
        # If we haven't loaded the summary from Redis yet, do it now
        if not existing_summary and redis_manager.is_available():
            existing_summary = redis_manager.client.get(summary_key) or ""

        # Summary trigger logic: Every 10 messages
        # To maintain a 20-message context window, we only prune when total > 20
        # but we can update the summary every 10.
        if total_messages % self.summary_trigger == 0 and total_messages > 0:
            logger.info(f"🧠 [Memory] Updating summary at {total_messages} messages.")
            
            # If we exceed the window, we prune. Otherwise we just summarize the whole history so far.
            if total_messages > self.keep_recent:
                msgs_to_summarize = messages[:-self.keep_recent]
                active_messages = messages[-self.keep_recent:]
            else:
                msgs_to_summarize = messages
                active_messages = messages # Keep all for now

            if msgs_to_summarize:
                msg_str = "\n".join([f"{m.type}: {m.content}" for m in msgs_to_summarize])
                
                chain = self.summary_prompt | self.llm
                res = await chain.ainvoke({
                    "existing_summary": existing_summary,
                    "messages_str": msg_str
                })
                
                new_summary = res.content
                logger.info("🧠 [Memory] Summarization updated.")
                
                if redis_manager.is_available():
                    redis_manager.client.set(summary_key, new_summary)
                
                # If we pruned, return the new list
                if len(active_messages) < total_messages:
                    return {
                        "conversation_summary": new_summary,
                        "messages": {"type": "replace", "messages": active_messages}
                    }
                return {"conversation_summary": new_summary}

        return {
            "conversation_summary": existing_summary
        }
