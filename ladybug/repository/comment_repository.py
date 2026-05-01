from sqlalchemy.orm import Session
from sqlalchemy import text


class CommentRepository:
    """
    DB access layer for training ingestion.
    """

    def __init__(self, db_session: Session):
        self.db_session = db_session


    def get_training_view(self, start_date, end_date):
        result = self.db_session.execute(
            text("""
                SELECT 
                    c.id,
                    c.comment_text,
                    c.published_at,

                    p.predicted_label,
                    p.confidence_score,
                    p.source

                FROM comments c
                LEFT JOIN model_predictions p
                ON c.id = p.comment_id

                WHERE c.published_at > :start_date
                AND c.published_at <= :end_date
                ORDER BY c.published_at ASC
            """),
            {
                "start_date": start_date,
                "end_date": end_date
            }
        )

        return result.mappings().all() or []