from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class RegisterSchema(BaseModel):
    email: EmailStr
    username:str
    password: str

class LoginSchema(BaseModel):
    email: EmailStr
    password: str

class DeleteAccountSchema(BaseModel):
    """
    Body for DELETE /api/auth/account.
    Requires the user's current password as a re-authentication step
    before the account and all related data are permanently removed.
    """
    password: str = Field(
        ...,
        min_length=1,
        description="Current account password — required to confirm deletion.",
    )