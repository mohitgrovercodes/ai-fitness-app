from fastapi import APIRouter, UploadFile, File
from app.modules.ai.controller import posture_check, recommend_workout

router = APIRouter()

@router.post("/posture-check")
async def check_posture(file: UploadFile = File(...)):
    return await posture_check(file)

@router.post("/recommend-workout")
async def recommend(data: dict):
    return await recommend_workout(data)

@router.post("/chat")
async def chat_endpoint(data: dict):
    """
    Agentic AI Chat Endpoint.
    Expects: { "message": "...", "user_id": "...", "context": { "goal": "...", "injuries": [...] } }
    """
    from app.modules.ai.controller import chat
    return await chat(data)