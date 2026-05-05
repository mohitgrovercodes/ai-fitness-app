from fastapi import APIRouter, Depends
from app.modules.auth.controller import register_user, login_user
from app.modules.auth.schema import RegisterSchema, LoginSchema

router = APIRouter()

@router.post("/register")
def register(payload: RegisterSchema):
    return register_user(payload)

@router.post("/login")
def login(payload: LoginSchema):
    return login_user(payload)