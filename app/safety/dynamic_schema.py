from typing import List, Dict
from pydantic import BaseModel, Field
from app.agents.training_agent import TrainingAnalysis, RestDay
from app.safety.constrained_output import build_workout_schema

def build_dynamic_training_analysis(safe_pool):
    WorkoutItem, _ = build_workout_schema(safe_pool)
    safe_pool_ids_str = ", ".join([f"'{ex.exercise_id}'" for ex in safe_pool])
    
    class DynamicTrainingAnalysis(BaseModel):
        is_accurate: bool = Field(description="Are the retrieved exercises relevant and safe?")
        needs_web_search: bool = Field(description="True if exercise is unknown or local DB lacks info.")
        sub_queries: List[str] = Field(default=[], description="Alternative search terms for routines.")
        final_answer: str = Field(description="Full text markdown response for chat users.")
        summary: str = Field(default="", description="Brief introduction/summary of the workout.")
        workout: List[WorkoutItem] = Field(default=[], description="List of active physical exercises. You MUST use valid exercise_ids from the schema.")
        rest_days: List[RestDay] = Field(default=[], description="List of rest days. Must be completely separate from the workout list.")
        tip: str = Field(default="", description="Closing tip for safety or cooldown.")
        exercise_gifs: Dict[str, str] = Field(default={}, description="Mapping of exercise name to GIF relative path.")
        exercise_images: Dict[str, str] = Field(default={}, description="Mapping of exercise name to Image relative path.")

    DynamicTrainingAnalysis.__doc__ = (
        "A personalized multi-day training plan.\n"
        "WARNING: Try to minimize repeating exercises, but you MAY repeat them across different days if necessary to fulfill a highly specific user focus.\n"
        f"CRITICAL: You MUST select 'exercise_id' ONLY from this exact list: [{safe_pool_ids_str}]. Never invent or shorten IDs."
    )
        
    return DynamicTrainingAnalysis
