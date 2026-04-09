from fastapi import FastAPI
from app.modules.auth.auth.routes import router as auth_router
from app.modules.ai.ai.routes import router as ai_router

app = FastAPI(title="AI Fitness App")

# Register routes
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(ai_router, prefix="/api/ai", tags=["AI"])

@app.get("/")
def root():
    return {"message": "AI Fitness API Running"}