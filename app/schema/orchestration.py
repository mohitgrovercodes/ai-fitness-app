from pydantic import BaseModel, Field
from typing import List, Optional

class UserContext(BaseModel):
    """Validated schema for user profile data."""
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    goal: Optional[str] = "General Fitness"
    activity_level: Optional[str] = "Moderate"
    diet_preference: Optional[str] = None
    injuries: List[str] = []
    medical_conditions: List[str] = []

class IntentResponse(BaseModel):
    """Strict schema for Orchestrator output."""
    intents: List[str] = Field(description="List of classified intents: workout, nutrition, progress, image, general, out_of_scope")
    is_fitness_domain: bool = Field(description="Is this related to the fitness gym domain?")
    has_body_transformation_goal: bool = Field(description="True if the user's message involves a body transformation goal that requires BOTH workout AND nutrition — e.g. losing weight, gaining muscle, slimming down, bulking up, body recomposition, fat loss, or any goal that implies changing their body composition. False for single-topic queries like 'how many calories in X?' or 'show me a push-up'.")
    reasoning: str = Field(description="Short explanation of the classification")
    confidence: float = Field(ge=0, le=1)
    detected_language: str = Field(description="The detected language of the query. Must be one of: 'english', 'hindi' (if Devanagari script), or 'hinglish' (if Hindi grammar/vocabulary in Roman script).")
    english_translation: str = Field(description="The English translation of the user's message. If the user message is already in English, return the original message as-is.")

class SafetyResult(BaseModel):
    """Schema for safety guardrail results."""
    is_safe: bool
    reason: str
    suggested_response: str = ""
