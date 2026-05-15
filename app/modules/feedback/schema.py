from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class RatingEnum(str, Enum):
    up = "up"
    down = "down"


class FeedbackCreate(BaseModel):
    rating: RatingEnum = Field(description="Thumbs up or down rating.")
    session_id: Optional[str] = Field(default=None, description="The conversation thread/session ID.")
    agent_intents: Optional[str] = Field(default=None, description="Comma-separated intents used, e.g. 'nutrition,workout'.")
    user_message: Optional[str] = Field(default=None, description="The user's original query.")
    ai_response_snippet: Optional[str] = Field(default=None, description="First 500 chars of the AI response.")
    comment: Optional[str] = Field(default=None, description="Optional free-text comment from user.")


class FeedbackResponse(BaseModel):
    id: str
    user_id: str
    session_id: Optional[str]
    rating: RatingEnum
    agent_intents: Optional[str]
    user_message: Optional[str]
    ai_response_snippet: Optional[str]
    comment: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class FeedbackSummary(BaseModel):
    total: int
    thumbs_up: int
    thumbs_down: int
    satisfaction_rate: float = Field(description="Percentage of thumbs up ratings (0-100).")
    recent_comments: List[str] = Field(default=[], description="Latest 5 user comments.")
