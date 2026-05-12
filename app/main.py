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
    from app.modules.profile.model import Profile # Ensure model is registered
    from app.modules.auth.model import User # Ensure model is registered
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

@app.get("/")
def root():
    return {"message": "AI Fitness API Running"}