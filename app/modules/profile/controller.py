from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.modules.profile.service import ProfileService
from app.modules.profile.schema import ProfileCreate, ProfileUpdate

class ProfileController:
    @staticmethod
    def get_profile(db: Session, user_id: str):
        profile = ProfileService.get_profile(db, user_id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        return profile

    @staticmethod
    def create_or_update_profile(db: Session, user_id: str, profile_data: ProfileCreate):
        try:
            return ProfileService.create_or_update_profile(db, user_id, profile_data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

    @staticmethod
    def update_profile(db: Session, user_id: str, profile_data: ProfileUpdate):
        profile = ProfileService.update_profile(db, user_id, profile_data)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        return profile
