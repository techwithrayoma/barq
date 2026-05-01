import pandas as pd
from sqlalchemy.orm import Session
from datetime import datetime

from repository.comment_repository import CommentRepository


class IngestData:
    """
    Responsible ONLY for selecting training data.
    No labeling, no cost, no predictions.
    """

    def __init__(self, db: Session):
        self.repo = CommentRepository(db_session=db)

    def fetch_training_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        # fetch only data within time window
        raw_data = self.repo.get_training_view(start_date, end_date)

        df = pd.DataFrame(raw_data)

        return df