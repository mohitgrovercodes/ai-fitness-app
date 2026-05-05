from typing import Dict, Any
from app.core.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.utils.logger import logger

class MemoryManager:
    """
    Production Memory Pruning.
    Summarizes older messages and truncates the history to prevent token limit errors.
    """
    def __init__(self, max_messages: int = 15, keep_recent: int = 5):
        from app.core.config import settings
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0, 
            api_key=settings.OPENAI_API_KEY
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are managing the conversation memory for an AI Fitness Coach.
Your task is to summarize the conversation history provided below.
Include any key details the user mentioned (e.g., goals, injuries, specific foods discussed, workouts planned).
If there is an existing summary, update it with the new information.
Keep it concise but comprehensive.

EXISTING SUMMARY: {existing_summary}"""),
            ("human", "NEW MESSAGES TO ADD TO SUMMARY:\n{messages_str}")
        ])

    async def run(self, state: AgentState) -> Dict[str, Any]:
        messages = state.get('messages', [])
        
        # Only prune if we exceed the limit
        if len(messages) <= self.max_messages:
            return {}
            
        logger.info(f"🧠 [Memory] Trimming conversation history (Current length: {len(messages)} messages).")
        
        # We summarize everything EXCEPT the most recent N messages
        messages_to_summarize = messages[:-self.keep_recent]
        recent_messages = messages[-self.keep_recent:]
        
        # Format messages for the LLM
        msg_str = "\n".join([f"{m.type}: {m.content}" for m in messages_to_summarize])
        existing_summary = state.get("conversation_summary", "No previous summary.")
        
        chain = self.prompt | self.llm
        res = await chain.ainvoke({
            "existing_summary": existing_summary,
            "messages_str": msg_str
        })
        
        new_summary = res.content
        logger.info("🧠 [Memory] Summarization complete.")
        
        return {
            "conversation_summary": new_summary,
            # This special dict triggers our custom reducer to OVERWRITE the list
            "messages": {"type": "replace", "messages": recent_messages}
        }
