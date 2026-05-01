from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timezone


class ModelPredictionRepository:
    """
    Stores ALL labeling events (GPT or model inference)
    """

    def __init__(self, db: Session):
        self.db = db

    def insert_predictions_batch(self, records: list[dict]):
        self.db.execute(
            text("""
                INSERT INTO model_predictions (
                    comment_id,
                    model_id,
                    stage,
                    source,
                    predicted_label,
                    confidence_score,
                    cost_usd,
                    latency_ms,
                    created_at
                )
                VALUES (
                    :comment_id,
                    :model_id,
                    :stage,
                    :source,
                    :predicted_label,
                    :confidence_score,
                    :cost_usd,
                    :latency_ms,
                    :created_at
                )
            """),
            [
                {**r, "created_at": datetime.now(timezone.utc)}
                for r in records
            ]
        )

        self.db.commit()