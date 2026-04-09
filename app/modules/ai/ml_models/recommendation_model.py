class RecommendationModel:

    @staticmethod
    def predict(data):
        goal = data.get("goal")

        if goal == "weight_loss":
            return ["Cardio", "HIIT"]
        elif goal == "muscle_gain":
            return ["Weight Training", "Protein Diet"]
        return ["General Fitness"]