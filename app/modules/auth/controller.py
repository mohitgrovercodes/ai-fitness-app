    # from fastapi import Depends
    # from fastapi.security import OAuth2PasswordRequestForm
    # from sqlalchemy.orm import Session

from app.modules.auth.service import AuthService
from app.common.response import success, error
from sqlalchemy.orm import Session


def register_user(db: Session, payload):
    try:
        user = AuthService.register(db, payload)
        return success(user.to_dict(), "User registered")

    except Exception as e:
        return error(str(e))


# def login_user(
#     db: Session,
#     form_data: OAuth2PasswordRequestForm = Depends()
# ):
#     try:
#         token = AuthService.login(db, form_data)

#         return token

#     except Exception as e:
#         return error(str(e))

def login_user(db: Session, payload):
    try:
        token = AuthService.login(db, payload)
        return success(token, "Login success")
    except Exception as e:
        return error(str(e))