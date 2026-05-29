from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime
from app.core.sql_db import Base
import uuid

class User(Base):
    __tablename__ = "users"
 
    user_id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)  
    username = Column(String(100), unique=True, index=True, nullable=False)  
    password_hash = Column(String(255), nullable=False)    
    created_at = Column(DateTime, default=datetime.utcnow)    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
 
 

    def to_dict(self) -> dict:
        """Convert user to dictionary"""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "username": self.username,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active,
            "is_admin": self.is_admin
        }