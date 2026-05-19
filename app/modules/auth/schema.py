from pydantic import BaseModel, EmailStr
from typing import Optional

class RegisterSchema(BaseModel):
    email: EmailStr
    password: str
    username: Optional[str] = None

class LoginSchema(BaseModel):
    username: str
    password: str