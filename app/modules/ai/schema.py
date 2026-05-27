from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message to the AI coach.")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional extra context (e.g. goal, injuries).")

class WorkoutGenerationRequest(BaseModel):
    goal: Optional[str] = Field(None, description="User's primary goal (e.g., muscle gain, weight loss).")
    level: Optional[str] = Field(None, description="User's experience level (e.g., beginner, intermediate, advanced).")
    gender: Optional[str] = Field(None, description="Gender male or female or not specified")
    age: Optional[int] = Field(None, description="Age of the user")
    height: Optional[int] = Field(None, description="height of the user")
    weight: Optional[int] = Field(None, description="user weight")
    injuries: Optional[List[str]] = Field(default_factory=list, description="Any injuries to accommodate.")
    message: Optional[str] = Field(None, description="Optional custom instructions or query override.")

class DietGenerationRequest(BaseModel): 
    goal: Optional[str] = Field(None, description="User's primary goal (e.g., weight loss, maintenance).")
    diet_type: Optional[str] = Field(None, description="Dietary preference (e.g., veg, non-veg, vegan, keto).")
    gender: Optional[str] = Field(None, description="Gender male or female or not specified")
    age: Optional[int] = Field(None, description="Age of the user")
    height: Optional[int] = Field(None, description="height of the user")
    weight: Optional[int] = Field(None, description="user weight")
    allergies: Optional[List[str]] = Field(default_factory=list, description="Food allergies to avoid.")
    message: Optional[str] = Field(None, description="Optional custom instructions or query override.")

class DomainQueryRequest(BaseModel):
    message: str = Field(..., description="A general fitness or science question.")

