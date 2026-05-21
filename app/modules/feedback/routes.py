from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.sql_db import get_db
from app.core.security import get_current_user, get_current_admin
from app.modules.feedback.schema import FeedbackCreate, FeedbackResponse, FeedbackSummary
from app.modules.feedback.service import FeedbackService

router = APIRouter()


@router.post("/submit", response_model=FeedbackResponse, summary="Submit thumbs up/down feedback")
async def submit_feedback(
    data: FeedbackCreate,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Step 12 — Feedback Loop.
    Submit a thumbs-up or thumbs-down rating for any AI response.

    Body example:
    ```json
    {
        "rating": "up",
        "session_id": "user-abc-thread-1",
        "agent_intents": "nutrition,workout",
        "user_message": "Give me a weekly diet plan",
        "ai_response_snippet": "Here is your 7-day meal plan...",
        "comment": "Very detailed and helpful!"
    }
    ```
    """
    entry = FeedbackService.submit(db, user_id, data)
    return entry


@router.get("/history", response_model=List[FeedbackResponse], summary="Get my feedback history")
async def get_my_feedback(
    limit: int = 20,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns the most recent feedback entries submitted by the authenticated user."""
    return FeedbackService.get_history(db, user_id, limit=limit)


@router.get("/summary", response_model=FeedbackSummary, summary="Get my satisfaction summary")
async def get_my_summary(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns aggregated satisfaction metrics for the current user:
    total ratings, thumbs up/down count, and satisfaction percentage.
    """
    return FeedbackService.get_summary(db, user_id)


@router.get("/admin/summary", response_model=FeedbackSummary, summary="Global feedback summary (admin)")
async def get_global_summary(
    intent: Optional[str] = None,
    admin_id: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    Admin endpoint — returns aggregated satisfaction metrics across ALL users.
    Requires the calling user to have is_admin=True in the database.
    """
    return FeedbackService.get_global_summary(db, intent=intent)


@router.get("/admin/list", response_model=List[FeedbackResponse], summary="Paginated list of all feedback (admin)")
async def get_global_list(
    page: int = 1,
    size: int = 50,
    intent: Optional[str] = None,
    rating: Optional[str] = None,
    admin_id: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    Admin endpoint — returns a paginated list of feedback entries.
    Useful for reading all negative comments in detail.
    Requires the calling user to have is_admin=True in the database.
    """
    return FeedbackService.get_global_list(db, page=page, size=size, intent=intent, rating=rating)
