from fastapi import APIRouter, UploadFile, File, Depends
from fastapi import Form
from typing import Optional
import json
from app.core.security import get_current_user


router = APIRouter()

@router.post("/chat")
async def chat_endpoint(
    data: dict, 
    user_id: str = Depends(get_current_user)
):
    """
    Agentic AI Chat Endpoint.
    Expects: { "message": "...", "context": { "goal": "...", "injuries": [...] } }
    """
    from app.modules.ai.controller import chat
    data["user_id"] = user_id
    return await chat(data)


@router.post("/chat-vision")
async def chat_vision_endpoint(
    message: str = Form(...),
    context: str = Form("{}"),
    file: Optional[UploadFile] = File(None),
    user_id: str = Depends(get_current_user)
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


@router.post("/generate-workout")
async def generate_workout_endpoint(
    data: dict,
    user_id: str = Depends(get_current_user)
):
    """
    Direct API to generate a workout plan (Bypasses Orchestrator).
    Expects: { "goal": "muscle gain", "level": "beginner", "duration": "1 month", "injuries": [] }
    """
    from app.modules.ai.controller import generate_workout
    data["user_id"] = user_id
    return await generate_workout(data)


@router.post("/generate-diet")
async def generate_diet_endpoint(
    data: dict,
    user_id: str = Depends(get_current_user)
):
    """
    Direct API to generate a structured diet plan (Bypasses Orchestrator).
    Expects: { "goal": "weight loss", "diet_type": "veg", "allergies": ["peanuts"] }
    """
    from app.modules.ai.controller import generate_diet
    data["user_id"] = user_id
    return await generate_diet(data)