from fastapi import APIRouter, UploadFile, File
from app.modules.ai.controller import posture_check, recommend_workout
from fastapi import Form
from typing import Optional
import json


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


@router.post("/chat-vision")
async def chat_vision_endpoint(
    message: str = Form(...),
    user_id: str = Form("default_user"),
    context: str = Form("{}"),
    file: Optional[UploadFile] = File(None)
):
    """
    Agentic AI Chat Endpoint with optional Image Upload (Vision Agent).
    Accepts multipart/form-data.
    'context' should be a stringified JSON object (e.g. '{"goal": "weight loss"}').
    """
    from app.modules.ai.controller import chat_with_image
    
    try:
        context_dict = json.loads(context)
    except json.JSONDecodeError:
        context_dict = {}
        
    return await chat_with_image(message, user_id, context_dict, file)