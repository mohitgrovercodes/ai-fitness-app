from app.modules.ai.ml_models.posture_model import PostureModel
from app.modules.ai.ml_models.recommendation_model import RecommendationModel
from app.core.graph import build_graph
from langchain_core.messages import HumanMessage
import uuid

# Initialize the compiled LangGraph
fitness_graph = build_graph()

class AIService:

    @staticmethod
    async def check_posture(file):
        import asyncio
        return await asyncio.to_thread(PostureModel.process, file)

    @staticmethod
    async def recommend_workout(data):
        import asyncio
        # Legacy support for simple recommendation
        return await asyncio.to_thread(RecommendationModel.predict, data)

    @staticmethod
    async def chat(user_input: str, user_id: str, context: dict = None, image_bytes: bytes = None):
        """
        Main entry point for the Agentic AI Gym Chatbot.
        Processes user input through the LangGraph Multi-Agent system.
        """
        config = {"configurable": {"thread_id": user_id}}
        
        # Initial state for the graph
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": context or {},
            "conversation_summary": "" # This will be updated by the memory_manager
        }
        if image_bytes:
            initial_state["image_bytes"] = image_bytes
            
        # Run the graph
        final_state = await fitness_graph.ainvoke(initial_state, config=config)
        
        # Extract the last AI message
        last_msg = final_state["messages"][-1]
        
        return {
            "response": last_msg.content,
            "intents": final_state.get("intent", []),
            "summary": final_state.get("conversation_summary", "")
        }