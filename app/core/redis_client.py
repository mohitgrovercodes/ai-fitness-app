import redis
import json
from app.utils.logger import logger
from app.core.config import settings

class RedisClient:
    """
    Singleton Redis client for persistent conversation memory.
    """
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
            try:
                # Production-grade Redis with ConnectionPool and Timeouts
                redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
                pool = redis.ConnectionPool.from_url(
                    redis_url, 
                    decode_responses=True,
                    max_connections=20,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                    retry_on_timeout=True
                )
                cls._client = redis.Redis(connection_pool=pool)
                cls._client.ping() # Verify connection
                logger.info("[Redis] Connected successfully with ConnectionPool.")
            except Exception as e:
                logger.error(f"[Redis] Connection failed: {e}")
                cls._client = None
        return cls._instance

    @property
    def client(self):
        return self._client

    def is_available(self) -> bool:
        return self._client is not None

# Global instance
redis_manager = RedisClient()
