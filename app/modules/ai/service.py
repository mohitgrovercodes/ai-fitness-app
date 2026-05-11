from app.modules.ai.ml_models.posture_model import PostureModel
from app.modules.ai.ml_models.recommendation_model import RecommendationModel
from app.core.graph import build_graph
from langchain_core.messages import HumanMessage
import uuid

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
            
        return out_data


    @staticmethod
    async def generate_workout_plan(data: dict):
        """Directly calls the TrainingAgent with structured data from a button/form."""
        from app.agents.training_agent import TrainingAgent
        
        user_id = data.get("user_id", "default")
        goal = data.get("goal", "")
        level = data.get("level", "")
        duration = data.get("duration", "")
        injuries = data.get("injuries", [])
        
        # Flexibility: if the frontend sends a specific message, use it. Otherwise, build one.
        user_input = data.get("message")
        if not user_input:
            parts = ["Create a structured workout plan"]
            if goal: parts.append(f"for {goal}")
            if level: parts.append(f"at a {level} fitness level")
            if duration: parts.append(f"for a duration of {duration}")
            user_input = " ".join(parts) + "."
            
        context = {"goal": goal, "injuries": injuries, "level": level}
        
        state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": context,
            "conversation_summary": "Direct API Generation Request"
        }
        result = await TrainingAgent().run(state)
        output = result.get("specialist_results", {}).get("training", {})
        return {
            "summary": output.get("summary", ""),
            "workout": output.get("workout", []),
            "tip": output.get("tip", ""),
            "response": output.get("answer", "Could not generate plan."),
        }

    @staticmethod
    async def generate_diet_plan(data: dict):
        """Directly calls the NutritionAgent with structured data from a button/form."""
        from app.agents.nutrition_agent import NutritionAgent
        
        user_id = data.get("user_id", "default")
        goal = data.get("goal", "")
        diet_type = data.get("diet_type", "")
        allergies = data.get("allergies", [])
        
        # Flexibility: if the frontend sends a specific message, use it. Otherwise, build one.
        user_input = data.get("message")
        if not user_input:
            parts = ["Create a structured time-based daily meal plan"]
            if goal: parts.append(f"for {goal}")
            if diet_type: parts.append(f"with a {diet_type} dietary preference")
            if allergies: parts.append(f". Avoid these allergies: {', '.join(allergies)}")
            user_input = " ".join(parts) + "."
            
        context = {"goal": goal, "injuries": allergies} # pass allergies as injuries so the agent avoids them
        
        state = {
            "messages": [HumanMessage(content=user_input)],
            "user_context": context,
            "conversation_summary": "Direct API Generation Request"
        }
        result = await NutritionAgent().run(state)
        output = result.get("specialist_results", {}).get("nutrition", {})
        return {
            "summary": output.get("summary", ""),
            "meals": output.get("meals", []),
            "daily_totals": output.get("daily_totals", {}),
            "tip": output.get("tip", ""),
            "response": output.get("answer", "Could not generate plan.")
        }