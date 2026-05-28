import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.modules.auth.routes import router as auth_router
from app.modules.ai.routes import router as ai_router
from app.modules.profile.routes import router as profile_router
from app.modules.feedback.routes import router as feedback_router

from contextlib import asynccontextmanager
from app.utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # PRE-INITIALIZATION (Production Grade for Mac/Uvicorn stability)
    logger.info("🚀 [Startup] Initializing Production Services...")
    
    # 1. Initialize ChromaDB (Main Thread)
    from app.core.database import db_manager
    import asyncio
    await asyncio.to_thread(db_manager.initialize)
    
    # 2. Pre-load CLIP (Avoids mid-request segfaults on Mac)
    from app.tools.vision_tools import _load_clip
    await asyncio.to_thread(_load_clip)
    
    # 3. Initialize SQL Database (Profiles & Auth)
    from app.core.sql_db import engine, Base
    from app.modules.profile.model import Profile   # Ensure model is registered
    from app.modules.auth.model import User           # Ensure model is registered
    from app.modules.feedback.model import Feedback   # Ensure model is registered
    Base.metadata.create_all(bind=engine)
    logger.info("✅ [Startup] SQL Database (Profiles & Auth) initialized.")

    
    logger.info("✅ [Startup] All services ready.")
    yield
    logger.info("🛑 [Shutdown] Cleaning up services...")

app = FastAPI(title="AI Fitness App", lifespan=lifespan)

# Register routes
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(ai_router, prefix="/api/ai", tags=["AI"])
app.include_router(profile_router, prefix="/api/profile", tags=["Profile"])
app.include_router(feedback_router, prefix="/api/feedback", tags=["Feedback"])

@app.get("/")
def root():
    return {"message": "AI Fitness API Running"}


@app.get("/health", tags=["Health"])
def ping():
    """
    Ultra-simple health check.
    Returns 200 OK if the FastAPI server is running.
    """
    return {"status": "ok", "message": "FastAPI Server is running successfully"}


@app.get("/health", tags=["Health"])
def health():
    """
    Liveness + readiness probe.

    Returns HTTP 200 with status="ok" when every critical dependency is
    reachable, or HTTP 503 with status="degraded" when one or more critical
    dependencies (MySQL, ChromaDB) are unreachable. Redis is reported but
    does not flip the overall status because the app can serve requests
    without it (history/summary become best-effort).

    Useful for Kubernetes / load balancer health checks, uptime monitors,
    and quick local debugging.
    """
    from fastapi.responses import JSONResponse
    from sqlalchemy import text

    components = {}
    overall_ok = True

    # ── MySQL (critical) ──────────────────────────────────────────────
    try:
        from app.core.sql_db import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        components["mysql"] = "ok"
    except Exception as e:
        components["mysql"] = f"error: {type(e).__name__}"
        overall_ok = False

    # ── ChromaDB (critical for AI features) ───────────────────────────
    try:
        from app.core.database import db_manager
        client = db_manager.initialize()  # lazily creates client on first call
        components["chromadb"] = "ok" if client is not None else "not initialized"
        if client is None:
            overall_ok = False
    except Exception as e:
        components["chromadb"] = f"error: {type(e).__name__}"
        overall_ok = False

    # ── Redis (degraded, not fatal) ───────────────────────────────────
    try:
        from app.core.redis_client import redis_manager
        if redis_manager.is_available() and redis_manager.client.ping():
            components["redis"] = "ok"
        else:
            components["redis"] = "unavailable"
    except Exception as e:
        components["redis"] = f"error: {type(e).__name__}"

    payload = {
        "status": "ok" if overall_ok else "degraded",
        "components": components,
    }
    return JSONResponse(content=payload, status_code=200 if overall_ok else 503)