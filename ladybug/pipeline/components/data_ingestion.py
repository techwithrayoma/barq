import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session


class IngestData:
    def __init__(self, db: Session):
        self.db = db

    def fetch_training_data(self, limit: int | None = None) -> pd.DataFrame:
        base_query = """
            SELECT
                id,
                comment_text
            FROM comments
        """

        if limit is not None:
            base_query += " LIMIT :limit"

        query = text(base_query)

        result = self.db.execute(query, {"limit": limit} if limit else {})
        rows = result.mappings().all()

        return pd.DataFrame(rows)
