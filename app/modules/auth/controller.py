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


def logout_user(token: str):
    """
    Logout a user by blacklisting their JWT token in Redis.
    """
    try:
        result = AuthService.logout(token)
        return success(result, "Logout success")
    except Exception:
        logger.exception("Unexpected error during logout")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed. Please try again later.",
        )


def delete_account(db: Session, user_id: str, payload):
    """
    Permanently delete the currently authenticated user's account and all
    related data. Requires the user's current password (in `payload`) as a
    re-authentication step. HTTPExceptions (404, 401) propagate untouched.
    Unexpected errors are logged and surfaced as a generic 500.
    """
    try:
        result = AuthService.delete_account(db, user_id, payload.password)
        return success(result, "Account deleted")
    except HTTPException:
        raise
    except Exception:
        logger.exception(f"Unexpected error deleting account for user '{user_id}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account. Please try again later.",
        )

def forgot_password(db: Session, payload):
    try:
        result = AuthService.forgot_password(db, payload.email)
        return success(result, "Forgot password processed")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error during forgot password")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process request. Please try again later.",
        )

def reset_password(db: Session, payload):
    try:
        result = AuthService.reset_password(db, payload.token, payload.new_password)
        return success(result, "Reset password processed")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error during reset password")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password. Please try again later.",
        )
