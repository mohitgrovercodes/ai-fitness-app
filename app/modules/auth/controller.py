from app.modules.auth.service import AuthService
from app.common.response import success, error

def register_user(payload):
    try:
        user = AuthService.register(payload)
        return success(user, "User registered")
    except Exception as e:
        return error(str(e))

def login_user(payload):
    try:
        token = AuthService.login(payload)
        return success(token, "Login success")
    except Exception as e:
        return error(str(e))