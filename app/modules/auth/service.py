from sqlalchemy.orm import Session
from app.core.security import create_access_token, get_password_hash, verify_password
from app.modules.auth.model import User
from app.modules.profile.model import Profile
from app.modules.feedback.model import Feedback
from app.core.redis_client import redis_manager
from app.utils.logger import logger
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

    @staticmethod
    def delete_account(db: Session, user_id: str, password: str) -> dict:
        """
        Permanently delete the user and ALL related data:
          - Profile         (user_profiles row)
          - Feedback        (feedback rows)
          - User            (users row)
          - Redis           (chat_history, chat_summary, chat_summary_index)

        Re-authentication: the caller MUST provide their current password.
        This protects against accidental deletion if the user's JWT is
        leaked, borrowed, or left active on a shared device — the attacker
        would still need the password to wipe the account.

        SQL deletes are atomic — committed in one transaction. Redis cleanup
        is best-effort: a Redis failure after the SQL commit is logged but
        does not raise, because the user account no longer exists either way.

        Raises:
          HTTP 404 — user_id has no matching row (token from already-deleted user).
          HTTP 401 — password does not match.
        """
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Account not found",
            )

        # ── Re-authentication: verify the supplied password ──────────────
        # Identical error semantics to login() to avoid leaking whether the
        # account exists vs. password is wrong (we passed the existence
        # check above only because the JWT was already valid — without the
        # JWT this branch is unreachable).
        if not verify_password(password, user.password_hash):
            logger.warning(
                f"🛑 [Auth] Account deletion blocked — wrong password for user_id='{user_id}'"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 1. Delete profile row (if present).
        db.query(Profile).filter(Profile.user_id == user_id).delete(
            synchronize_session=False
        )

        # 2. Delete all feedback entries owned by this user.
        db.query(Feedback).filter(Feedback.user_id == user_id).delete(
            synchronize_session=False
        )

        # 3. Delete the user itself.
        db.delete(user)
        db.commit()

        logger.info(f"🗑️ [Auth] Account deleted for user_id='{user_id}'")

        # 4. Best-effort Redis cleanup (history, summary, summary cursor).
        try:
            if redis_manager.is_available():
                redis_manager.client.delete(
                    f"chat_history:{user_id}",
                    f"chat_summary:{user_id}",
                    f"chat_summary_index:{user_id}",
                )
                logger.info(f"🧹 [Auth] Redis keys cleared for '{user_id}'")
        except Exception as e:
            # SQL already committed; do not raise.
            logger.warning(
                f"[Auth] Failed to clean Redis for deleted user '{user_id}': {e}"
            )

        return {"user_id": user_id, "deleted": True}

    @staticmethod
    def logout(token: str):
        from jose import jwt, JWTError
        from app.core.config import settings
        import time
        
        if not token:
            return {"message": "No token provided"}
            
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            exp = payload.get("exp")
            if not exp:
                return {"message": "Logged out successfully"}
            
            ttl = exp - int(time.time())
            if ttl > 0:
                if redis_manager.is_available():
                    redis_manager.client.setex(f"blacklist_{token}", ttl, "blacklisted")
                    logger.info(f"🔒 [Auth] Token blacklisted for {ttl} seconds.")
                    
            return {"message": "Logged out successfully"}
        except JWTError:
            return {"message": "Logged out successfully"}