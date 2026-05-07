from app.modules.ai.service import AIService
from app.common.response import success, error

async def posture_check(file):
    try:
        result = await AIService.check_posture(file)
        return success(result)
    except Exception as e:
        return error(str(e))

async def recommend_workout(data):
    try:
        result = await AIService.recommend_workout(data)
        return success(result)
    except Exception as e:
        return error(str(e))

async def chat(data: dict):
    try:
        user_input = data.get("message")
        user_id = data.get("user_id", "default_user")
        context = data.get("context", {})
        
        if not user_input:
            return error("Message is required")
            
        result = await AIService.chat(user_input, user_id, context)
        return success(result)
    except Exception as e:
        return error(f"Chat error: {str(e)}")

async def chat_with_image(message: str, user_id: str, context: dict, file):
    try:
        image_bytes = None
        if file:
            image_bytes = await file.read()
            
        result = await AIService.chat(message, user_id, context=context, image_bytes=image_bytes)
        return success(result)
    except Exception as e:
        return error(f"Chat vision error: {str(e)}")

async def generate_workout(data: dict):
    try:
        result = await AIService.generate_workout_plan(data)
        return success(result)
    except Exception as e:
        return error(f"Workout Generation Error: {str(e)}")

async def generate_diet(data: dict):
    try:
        result = await AIService.generate_diet_plan(data)
        return success(result)
    except Exception as e:
        return error(f"Diet Generation Error: {str(e)}")