from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Boolean

from .base import Base


class ModelPrediction(Base):
    """
    Stores every prediction or labeling action on a comment.
    """

    __tablename__ = "model_predictions"

    id = Column(Integer, primary_key=True, index=True)

    comment_id = Column(Integer, ForeignKey("comments.id"), nullable=False)

    model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    # NULL = GPT labeling
    # NOT NULL = model inference

    stage = Column(String, nullable=False)
    # "labeling" | "inference"

    source = Column(String, nullable=False)
    # "gpt" | "model"

    predicted_label = Column(String, nullable=True)

    confidence_score = Column(String, nullable=True)

    cost_usd = Column(String, nullable=True)

    latency_ms = Column(Integer, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    # maybe the infrence gpu used to make prediction...... 