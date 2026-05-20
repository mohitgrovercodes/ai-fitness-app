from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message to the AI coach.")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional extra context (e.g. goal, injuries).")

class WorkoutGenerationRequest(BaseModel):
    goal: str = Field(..., description="User's primary goal (e.g., muscle gain, weight loss).")
    level: str = Field(..., description="User's experience level (e.g., beginner, intermediate, advanced).")
    gender: str=Field(..., description="Gender male or feamale or not specified")
    height: int=Field(...,description="height of the user")
    weight: int=Field(..., description="user weight")
    injuries: Optional[List[str]] = Field(default_factory=list, description="Any injuries to accommodate.")

class DietGenerationRequest(BaseModel): 
    goal: str = Field(..., description="User's primary goal (e.g., weight loss, maintenance).")
    diet_type: str = Field(..., description="Dietary preference (e.g., veg, non-veg, vegan, keto).")
    gender: str=Field(..., description="Gender male or feamale or not specified")
    height: int=Field(...,description="height of the user")
    weight: int=Field(..., description="user weight")
    diet_prefrence: str=Field(...,description="User diet prefrence 9")
    allergies: Optional[List[str]] = Field(default_factory=list, description="Food allergies to avoid.")

class DomainQueryRequest(BaseModel):
    message: str = Field(..., description="A general fitness or science question.")
