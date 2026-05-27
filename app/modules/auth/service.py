from sqlalchemy.orm import Session
from app.core.security import create_access_token, get_password_hash, verify_password
from app.modules.auth.model import User
from fastapi import HTTPException, status

class AuthService:

    @staticmethod
    def register(db: Session, payload):
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == payload.email).first() or db.query(User).filter(User.username==payload.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email or username already exists"
            )
        
        # Create new user
        new_user = User(
            email=payload.email,
            username=getattr(payload, "username", payload.email.split("@")[0]),
            password_hash=get_password_hash(payload.password)
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user

    @staticmethod
    def login(db: Session, payload):
        # Find user
        user = db.query(User).filter(User.username == payload.username).first()
        if not user or not verify_password(payload.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = create_access_token({"sub": user.user_id, "username": user.username})
        return {"access_token": token, "token_type": "bearer", "user_id": user.user_id}