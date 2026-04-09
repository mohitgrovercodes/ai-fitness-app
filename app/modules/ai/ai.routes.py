from fastapi import APIRouter, UploadFile, File
from app.modules.ai.ai.controller import posture_check, recommend_workout

router = APIRouter()

@router.post("/posture-check")
def check_posture(file: UploadFile = File(...)):
    return posture_check(file)

@router.post("/recommend-workout")
def recommend(data: dict):
    return recommend_workout(data)