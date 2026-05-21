from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from app.modules.feedback.model import Feedback, RatingEnum
from app.modules.feedback.schema import FeedbackCreate, FeedbackSummary
from app.utils.logger import logger


class FeedbackService:

    @staticmethod
    def submit(db: Session, user_id: str, data: FeedbackCreate) -> Feedback:

        snippet = data.ai_response_snippet or None

        entry = Feedback(
            user_id=user_id,
            session_id=data.session_id,
            rating=data.rating,
            agent_intents=data.agent_intents,
            user_message=(data.user_message or "")[:1000] or None,
            ai_response_snippet=snippet,
            comment=(data.comment or "")[:2000] or None,
        )

        db.add(entry)
        db.commit()
        db.refresh(entry)

        logger.info(
            f"👍 [Feedback] User '{user_id}' rated '{data.rating}' | intents: {data.agent_intents}"
        )

        return entry

    @staticmethod
    def get_history(db: Session, user_id: str, limit: int = 20) -> List[Feedback]:
        """Fetch the most recent feedback entries for a user."""
        return (
            db.query(Feedback)
            .filter(Feedback.user_id == user_id)
            .order_by(Feedback.created_at.desc())
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_summary(db: Session, user_id: str) -> FeedbackSummary:
        """Compute aggregated satisfaction metrics for a user."""
        total = db.query(func.count(Feedback.id)).filter(Feedback.user_id == user_id).scalar() or 0
        thumbs_up = (
            db.query(func.count(Feedback.id))
            .filter(Feedback.user_id == user_id, Feedback.rating == RatingEnum.up)
            .scalar() or 0
        )
        thumbs_down = total - thumbs_up
        satisfaction_rate = round((thumbs_up / total) * 100, 1) if total > 0 else 0.0

        # Pull latest comments
        recent_rows = (
            db.query(Feedback.comment)
            .filter(Feedback.user_id == user_id, Feedback.comment.isnot(None))
            .order_by(Feedback.created_at.desc())
            .limit(5)
            .all()
        )
        recent_comments = [r.comment for r in recent_rows if r.comment]

        return FeedbackSummary(
            total=total,
            thumbs_up=thumbs_up,
            thumbs_down=thumbs_down,
            satisfaction_rate=satisfaction_rate,
            recent_comments=recent_comments,
        )

    @staticmethod
    def get_global_summary(db: Session) -> FeedbackSummary:
        """Admin-level aggregated metrics across ALL users."""
        total = db.query(func.count(Feedback.id)).scalar() or 0
        thumbs_up = (
            db.query(func.count(Feedback.id))
            .filter(Feedback.rating == RatingEnum.up)
            .scalar() or 0
        )
        thumbs_down = total - thumbs_up
        satisfaction_rate = round((thumbs_up / total) * 100, 1) if total > 0 else 0.0

        recent_rows = (
            db.query(Feedback.comment)
            .filter(Feedback.comment.isnot(None))
            .order_by(Feedback.created_at.desc())
            .limit(5)
            .all()
        )
        recent_comments = [r.comment for r in recent_rows if r.comment]

        return FeedbackSummary(
            total=total,
            thumbs_up=thumbs_up,
            thumbs_down=thumbs_down,
            satisfaction_rate=satisfaction_rate,
            recent_comments=recent_comments,
        )
