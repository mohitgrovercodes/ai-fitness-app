from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.modules.auth.controller import register_user, login_user
from app.modules.auth.schema import RegisterSchema, LoginSchema
from app.core.sql_db import get_db

router = APIRouter(tags=["Auth"])


@router.post("/register")
def register(payload: RegisterSchema, db: Session = Depends(get_db)):
    return register_user(db, payload)

@router.post("/login")
def login(payload: LoginSchema, db: Session = Depends(get_db)):
    return login_user(db, payload)