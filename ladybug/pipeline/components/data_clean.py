import pandas as pd

class DataClean:
    def __init__(self, ingested_data: pd.DataFrame):
        self.ingested_data = ingested_data

    def clean_training_data(self) -> pd.DataFrame:
        self.ingested_data = self.ingested_data.rename(columns={"comment_text": "comment"})
        self.ingested_data = self.ingested_data.drop(columns=['id'])

        return self.ingested_data