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
        result = AIService.recommend_workout(data)
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