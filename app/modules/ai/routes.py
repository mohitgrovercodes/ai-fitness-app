from fastapi import APIRouter, UploadFile, File, Depends
from fastapi import Form
from typing import Optional
import json
from app.core.security import get_current_user
from app.modules.ai.schema import ChatRequest, WorkoutGenerationRequest, DietGenerationRequest, DomainQueryRequest


router = APIRouter()

@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest, 
    user_id: str = Depends(get_current_user)
):
    """
    Agentic AI Chat Endpoint.
    Expects: { "message": "...", "context": { "goal": "...", "injuries": [...] } }
    """
    from app.modules.ai.controller import chat
    data = request.model_dump()
    data["user_id"] = user_id
    return await chat(data)


@router.post("/chat-vision")
async def chat_vision_endpoint(
    message: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    """
    Agentic AI Chat Endpoint with Image Upload (Vision Agent).
    Accepts multipart/form-data.
    """
    from app.modules.ai.controller import chat_with_image
        
    return await chat_with_image(message, user_id, {}, file)


@router.post("/generate-workout")
async def generate_workout_endpoint(
    request: WorkoutGenerationRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Direct API to generate a workout plan (Bypasses Orchestrator).
    Expects: { "goal": "muscle gain", "level": "beginner", "duration": "1 month", "injuries": [] }
    """
    from app.modules.ai.controller import generate_workout
    data = request.model_dump()
    data["user_id"] = user_id
    return await generate_workout(data)


@router.post("/generate-diet")
async def generate_diet_endpoint(
    request: DietGenerationRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Direct API to generate a structured diet plan (Bypasses Orchestrator).
    Expects: { "goal": "weight loss", "diet_type": "veg", "allergies": ["peanuts"] }
    """
    from app.modules.ai.controller import generate_diet
    data = request.model_dump()
    data["user_id"] = user_id
    return await generate_diet(data)


@router.post("/ask-domain")
async def ask_domain_endpoint(
    request: DomainQueryRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Direct API for general fitness/science questions (Bypasses Orchestrator).
    Expects: { "message": "What is muscle hypertrophy?" }
    """
    from app.modules.ai.controller import ask_domain
    data = request.model_dump()
    data["user_id"] = user_id
    return await ask_domain(data)