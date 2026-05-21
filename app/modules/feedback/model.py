from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Enum as SAEnum
from app.core.sql_db import Base
import uuid
import enum
from sqlalchemy.dialects.mysql import JSON

class RatingEnum(str, enum.Enum):
    up = "up"
    down = "down"


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    session_id = Column(String(255), nullable=True, index=True)  # thread_id / conversation session
    rating = Column(SAEnum(RatingEnum), nullable=False)
    agent_intents = Column(String(255), nullable=True)           # e.g. "nutrition,workout"
    user_message = Column(Text, nullable=True)                   # The user's original query
    ai_response_snippet = Column(JSON, nullable=True)            # First 500 chars of AI response
    comment = Column(Text, nullable=True)                        # Optional free-text from user
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "rating": self.rating,
            "agent_intents": self.agent_intents,
            "user_message": self.user_message,
            "ai_response_snippet": self.ai_response_snippet,
            "comment": self.comment,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
