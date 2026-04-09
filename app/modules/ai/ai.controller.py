from app.modules.ai.ai.service import AIService
from app.common.response import success, error

def posture_check(file):
    try:
        result = AIService.check_posture(file)
        return success(result)
    except Exception as e:
        return error(str(e))

def recommend_workout(data):
    try:
        result = AIService.recommend_workout(data)
        return success(result)
    except Exception as e:
        return error(str(e))