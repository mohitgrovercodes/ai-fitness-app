import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Production-grade configuration management.
    Handles environment variables and global constants.
    """
    # API & Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key-for-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # OpenAI Config
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Tavily Web Search
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Cultural & Safety Restrictions
    RESTRICTED_FOODS: List[str] = ["beef"]
    
    # Agent Performance Thresholds
    VISION_CONFIDENCE_THRESHOLD: float = 0.70
    AMBIGUITY_GAP_THRESHOLD: float = 0.15
    VISION_NON_FOOD_THRESHOLD: float = 0.82
    
    # RAG Settings
    TOP_K_RECORDS: int = 5
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()