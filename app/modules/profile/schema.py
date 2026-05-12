# from pydantic import BaseModel, Field
# from typing import Optional, List
# from enum import Enum

# class Gender(str, Enum):
#     MALE = "male"
#     FEMALE = "female"
#     OTHER = "other"

# class Goal(str, Enum):
#     FAT_LOSS = "fat_loss"
#     MUSCLE_GAIN = "muscle_gain"
#     MAINTENANCE = "maintenance"
#     ATHLETIC_PERFORMANCE = "athletic_performance"

# class ActivityLevel(str, Enum):
#     SEDENTARY = "sedentary"
#     LIGHTLY_ACTIVE = "lightly_active"
#     MODERATELY_ACTIVE = "moderately_active"
#     VERY_ACTIVE = "very_active"
#     EXTRA_ACTIVE = "extra_active"

# class ProfileBase(BaseModel):
#     full_name: str = Field(..., example="John Doe")
#     age: int = Field(..., ge=1, le=120)
#     gender: Gender
#     height: float = Field(..., description="Height in cm", ge=50, le=250)
#     weight: float = Field(..., description="Weight in kg", ge=10, le=300)
#     goal: Goal
#     activity_level: ActivityLevel
#     diet_preference: Optional[str] = Field(None, example="Vegetarian")
#     injuries: Optional[str] = Field(None, example="Knee pain")
#     medical_conditions: Optional[str] = Field(None, example="Asthma")

# class ProfileCreate(ProfileBase):
#     pass

# class ProfileUpdate(BaseModel):
#     full_name: Optional[str] = None
#     age: Optional[int] = None
#     gender: Optional[Gender] = None
#     height: Optional[float] = None
#     weight: Optional[float] = None
#     goal: Optional[Goal] = None
#     activity_level: Optional[ActivityLevel] = None
#     diet_preference: Optional[str] = None
#     injuries: Optional[str] = None
#     medical_conditions: Optional[str] = None

# class ProfileResponse(ProfileBase):
#     user_id: str

#     class Config:
#         from_attributes = True


from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List
from enum import Enum


# -----------------------------------
# ENUMS
# -----------------------------------

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class ActivityLevel(str, Enum):
    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    EXTRA_ACTIVE = "extra_active"


# -----------------------------------
# DYNAMIC VALUE NORMALIZATION
# -----------------------------------

GOAL_MAPPINGS = {
    "fat loss": "fat_loss",
    "lose weight": "fat_loss",
    "weight loss": "fat_loss",
    "cutting": "fat_loss",

    "muscle gain": "muscle_gain",
    "build muscle": "muscle_gain",
    "bulk": "muscle_gain",
    "bulking": "muscle_gain",

    "maintenance": "maintenance",
    "maintain": "maintenance",

    "athletic performance": "athletic_performance",
    "sports performance": "athletic_performance",
}


DIET_MAPPINGS = {
    "veg": "vegetarian",
    "vegetarian": "vegetarian",

    "nonveg": "non_vegetarian",
    "non veg": "non_vegetarian",
    "non vegetarian": "non_vegetarian",

    "vegan": "vegan",

    "jain": "jain",

    "eggetarian": "eggetarian",

    "keto": "keto",

    "paleo": "paleo",
}


# -----------------------------------
# BASE MODEL
# -----------------------------------

class ProfileBase(BaseModel):

    model_config = ConfigDict(
        str_strip_whitespace=True
    )

    # REQUIRED FIELDS

    full_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        examples=["John Doe"]
    )

    age: int = Field(
        ...,
        ge=1,
        le=120
    )

    gender: Gender

    height: float = Field(
        ...,
        description="Height in cm",
        ge=50,
        le=250
    )

    weight: float = Field(
        ...,
        description="Weight in kg",
        ge=10,
        le=300
    )

    # Dynamic but required
    goal: str = Field(
        ...,
        examples=["fat loss", "build muscle"]
    )

    activity_level: ActivityLevel

    # Dynamic but required
    diet_preference: str = Field(
        ...,
        examples=[
            "Vegetarian",
            "Non Veg",
            "Vegan",
            "Keto"
        ]
    )

    # ONLY OPTIONAL FIELDS

    injuries: Optional[List[str]] = None

    medical_conditions: Optional[List[str]] = None

    allergies: Optional[List[str]] = None

    # -----------------------------------
    # VALIDATORS
    # -----------------------------------

    @field_validator("gender", "activity_level", mode="before")
    @classmethod
    def normalize_enums(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("goal", mode="before")
    @classmethod
    def normalize_goal(cls, value):

        if not value:
            raise ValueError("Goal is required")

        value = value.strip().lower()

        return GOAL_MAPPINGS.get(
            value,
            value.replace(" ", "_")
        )

    @field_validator("diet_preference", mode="before")
    @classmethod
    def normalize_diet(cls, value):

        if not value:
            raise ValueError("Diet preference is required")

        value = value.strip().lower()

        return DIET_MAPPINGS.get(
            value,
            value.replace(" ", "_")
        )

    @field_validator(
        "injuries",
        "medical_conditions",
        "allergies",
        mode="before"
    )
    @classmethod
    def convert_string_to_list(cls, value):

        if value is None:
            return None

        # Allow:
        # "Asthma, Diabetes"
        # OR ["Asthma", "Diabetes"]

        if isinstance(value, str):
            return [
                item.strip().lower()
                for item in value.split(",")
            ]

        if isinstance(value, list):
            return [str(item).strip().lower() for item in value]

        return value


# -----------------------------------
# CREATE
# -----------------------------------

class ProfileCreate(ProfileBase):
    pass


# -----------------------------------
# UPDATE
# -----------------------------------

class ProfileUpdate(BaseModel):

    model_config = ConfigDict(
        str_strip_whitespace=True
    )

    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[Gender] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    goal: Optional[str] = None
    activity_level: Optional[ActivityLevel] = None
    diet_preference: Optional[str] = None
    injuries: Optional[List[str]] = None
    medical_conditions: Optional[List[str]] = None
    allergies: Optional[List[str]] = None

    @field_validator("gender", "activity_level", mode="before")
    @classmethod
    def normalize_enums(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("goal", mode="before")
    @classmethod
    def normalize_goal(cls, value):
        if not value:
            return value
        value = value.strip().lower()
        return GOAL_MAPPINGS.get(value, value.replace(" ", "_"))

    @field_validator("diet_preference", mode="before")
    @classmethod
    def normalize_diet(cls, value):
        if not value:
            return value
        value = value.strip().lower()
        return DIET_MAPPINGS.get(value, value.replace(" ", "_"))

    @field_validator(
        "injuries",
        "medical_conditions",
        "allergies",
        mode="before"
    )
    @classmethod
    def convert_string_to_list(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",")]
        if isinstance(value, list):
            return [str(item).strip().lower() for item in value]
        return value


# -----------------------------------
# RESPONSE
# -----------------------------------

class ProfileResponse(ProfileBase):

    user_id: str

    model_config = ConfigDict(
        from_attributes=True
    )