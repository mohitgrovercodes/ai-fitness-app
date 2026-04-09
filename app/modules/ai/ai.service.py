from app.modules.ai.ml_models.posture_model import PostureModel
from app.modules.ai.ml_models.recommendation_model import RecommendationModel

class AIService:

    @staticmethod
    def check_posture(file):
        return PostureModel.process(file)

    @staticmethod
    def recommend_workout(data):
        return RecommendationModel.predict(data)