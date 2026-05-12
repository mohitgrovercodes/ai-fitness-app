from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.core.sql_db import get_db
from app.core.security import get_current_user
from app.modules.profile.schema import ProfileCreate, ProfileUpdate, ProfileResponse
from app.modules.profile.controller import ProfileController

router = APIRouter()

@router.get("/me", response_model=ProfileResponse)
def get_my_profile(
    db: Session = Depends(get_db), 
    user_id: str = Depends(get_current_user)
):
    return ProfileController.get_profile(db, user_id)

@router.post("/onboarding", response_model=ProfileResponse, status_code=status.HTTP_201_CREATED)
def onboarding(
    profile_data: ProfileCreate, 
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    return ProfileController.create_or_update_profile(db, user_id, profile_data)

@router.patch("/me", response_model=ProfileResponse)
def update_profile(
    profile_data: ProfileUpdate, 
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    return ProfileController.update_profile(db, user_id, profile_data)
