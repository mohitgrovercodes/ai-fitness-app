from datetime import datetime
from typing import Optional

class User:
    """User authentication model"""
    
    def __init__(
        self,
        user_id: str,
        email: str,
        username: str,
        password_hash: str,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        is_active: bool = True
    ):
        self.user_id = user_id
        self.email = email
        self.username = username
        self.password_hash = password_hash
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        self.is_active = is_active
    
    def to_dict(self) -> dict:
        """Convert user to dictionary"""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "username": self.username,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active
        }


class Token:
    """Authentication token model"""
    
    def __init__(
        self,
        access_token: str,
        token_type: str = "bearer",
        expires_in: int = 3600
    ):
        self.access_token = access_token
        self.token_type = token_type
        self.expires_in = expires_in