from sqlalchemy.orm import Session
from app.modules.profile.model import Profile
from app.modules.profile.schema import ProfileCreate, ProfileUpdate

class ProfileService:
    @staticmethod
    def get_profile(db: Session, user_id: str):
        return db.query(Profile).filter(Profile.user_id == user_id).first()

    @staticmethod
    def create_or_update_profile(db: Session, user_id: str, profile_data: ProfileCreate):
        db_profile = db.query(Profile).filter(Profile.user_id == user_id).first()
        
        if db_profile:
            # Update existing profile
            for key, value in profile_data.dict().items():
                setattr(db_profile, key, value)
        else:
            # Create new profile
            db_profile = Profile(user_id=user_id, **profile_data.dict())
            db.add(db_profile)
        
        db.commit()
        db.refresh(db_profile)
        return db_profile

    @staticmethod
    def update_profile(db: Session, user_id: str, profile_data: ProfileUpdate):
        db_profile = db.query(Profile).filter(Profile.user_id == user_id).first()
        if not db_profile:
            return None
        
        update_data = profile_data.dict(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_profile, key, value)
            
        db.commit()
        db.refresh(db_profile)
        return db_profile
