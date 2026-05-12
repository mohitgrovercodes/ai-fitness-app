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

    # Redis Persistence
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # MySQL Persistence
    MYSQL_URL: str = os.getenv("MYSQL_URL", "mysql+pymysql://root:password@localhost:3306/fitness_db")

    # Cultural & Safety Restrictions
    RESTRICTED_FOODS: List[str] = ["beef"]
    
    # Agent Performance Thresholds
    VISION_CONFIDENCE_THRESHOLD: float = 0.86
    AMBIGUITY_GAP_THRESHOLD: float = 0.01
    VISION_NON_FOOD_THRESHOLD: float = 0.82
    
    # RAG Settings
    TOP_K_RECORDS: int = 5
    NUTRITION_CANDIDATES_COUNT: int = 3
    NUTRITION_SIMILARITY_THRESHOLD: float = 0.60
    
    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()