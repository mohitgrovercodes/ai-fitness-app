from pydantic import BaseModel, Field
from typing import List, Optional

class UserContext(BaseModel):
    """Validated schema for user profile data."""
    age: Optional[int] = None
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    goal: Optional[str] = "General Fitness"
    activity_level: Optional[str] = "Moderate"
    injuries: List[str] = []
    medical_conditions: List[str] = []

class IntentResponse(BaseModel):
    """Strict schema for Orchestrator output."""
    intent: str = Field(description="The classified intent: workout, nutrition, progress, image, general, out_of_scope")
    is_fitness_domain: bool = Field(description="Is this related to the fitness gym domain?")
    reasoning: str = Field(description="Short explanation of the classification")
    confidence: float = Field(ge=0, le=1)
