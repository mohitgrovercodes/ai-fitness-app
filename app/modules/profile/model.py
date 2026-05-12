from sqlalchemy import Column, String, Float, Integer, Enum as SQLEnum, JSON
from app.core.sql_db import Base
from app.modules.profile.schema import Gender,  ActivityLevel

class Profile(Base):
    __tablename__ = "user_profiles"

    user_id = Column(String(36), primary_key=True, index=True)
    full_name = Column(String(100))
    age = Column(Integer)
    gender = Column(SQLEnum(Gender))
    height = Column(Float)
    weight = Column(Float)
    goal = Column(String(100))
    activity_level = Column(SQLEnum(ActivityLevel))
    diet_preference = Column(String(100), nullable=True)
    injuries = Column(JSON, nullable=True)
    medical_conditions = Column(JSON, nullable=True)
    allergies = Column(JSON, nullable=True)
    def to_dict(self):
        return {
            "user_id": self.user_id,
            "full_name": self.full_name,
            "age": self.age,
            "gender": self.gender.value if self.gender else None,
            "height": self.height,
            "weight": self.weight,
            "goal": self.goal,
            "activity_level": self.activity_level.value if self.activity_level else None,
            "diet_preference": self.diet_preference,
            "injuries": self.injuries,
            "medical_conditions": self.medical_conditions,
            "allergies": self.allergies
        }
