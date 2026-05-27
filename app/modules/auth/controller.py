from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.modules.auth.service import AuthService
from app.common.response import success
from app.utils.logger import logger


def register_user(db: Session, payload):
    """
    Register a new user. HTTPExceptions raised by the service layer
    (e.g. duplicate email/username) propagate untouched so FastAPI returns
    the correct status code. Only unexpected errors are caught and
    converted to a generic 500 with no internal details leaked to the client.
    """
    try:
        user = AuthService.register(db, payload)
        return success(user.to_dict(), "User registered")
    except HTTPException:
        # Let FastAPI render the proper 4xx response.
        raise
    except Exception:
        # Log full detail server-side, but expose nothing to the client.
        logger.exception("Unexpected error during registration")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again later.",
        )


def login_user(db: Session, payload):
    """
    Authenticate a user. HTTPExceptions (e.g. 401 invalid credentials)
    propagate untouched. Unexpected errors are logged and replaced with a
    generic 500 — no DB / SQLAlchemy text is returned to the caller.
    """
    try:
        token = AuthService.login(db, payload)
        return success(token, "Login success")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error during login")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed. Please try again later.",
        )
