from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.modules.auth.controller import register_user, login_user, logout_user, delete_account
from app.modules.auth.schema import RegisterSchema, LoginSchema, DeleteAccountSchema
from app.core.sql_db import get_db
from app.core.security import get_current_user, oauth2_scheme

router = APIRouter(tags=["Auth"])


@router.post("/register")
def register(payload: RegisterSchema, db: Session = Depends(get_db)):
    return register_user(db, payload)


@router.post("/login")
def login(payload: LoginSchema, db: Session = Depends(get_db)):
    return login_user(db, payload)


@router.post("/logout")
def logout(token: str = Depends(oauth2_scheme)):
    return logout_user(token)


@router.delete(
    "/account",
    status_code=status.HTTP_200_OK,
    summary="Permanently delete the authenticated user's account and data",
)
def delete_my_account(
    payload: DeleteAccountSchema,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    """
    Deletes the caller's User row, Profile row, all Feedback entries,
    and their Redis chat history.

    Requires:
      - A valid bearer token whose user still exists and is active
        (enforced by `get_current_user`).
      - The user's current password in the request body, as a
        re-authentication step.

    Body example:
    ```json
    { "password": "my-current-password" }
    ```
    """
    return delete_account(db, user_id, payload)