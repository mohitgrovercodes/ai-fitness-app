from app.core.security import create_access_token

class AuthService:

    @staticmethod
    def register(payload):
        # Save user to DB (mock)
        return {"email": payload.email}

    @staticmethod
    def login(payload):
        # Validate user (mock)
        token = create_access_token({"sub": payload.email})
        return {"access_token": token}