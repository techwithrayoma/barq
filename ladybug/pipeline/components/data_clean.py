import pandas as pd


class DataClean:
    def __init__(self, ingested_data: pd.DataFrame):
        self.df = ingested_data

    def clean_training_data(self) -> pd.DataFrame:
        # ─────────────────────────────────────────────
        # 1. STANDARDIZE COLUMN NAME
        # ─────────────────────────────────────────────
        self.df = self.df.rename(columns={"comment_text": "comment"})

        # ─────────────────────────────────────────────
        # 2. BASIC TEXT NORMALIZATION
        # ─────────────────────────────────────────────
        self.df["comment"] = (
            self.df["comment"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        # ─────────────────────────────────────────────
        # 3. OPTIONAL: TRACK DUPLICATES (VERY USEFUL)
        # ─────────────────────────────────────────────
        self.df["duplicate_count"] = (
            self.df.groupby("comment")["comment"]
            .transform("count")
        )

        # ─────────────────────────────────────────────
        # 4. REMOVE EXACT DUPLICATES (KEEP FIRST OCCURRENCE)
        # ─────────────────────────────────────────────
        self.df = self.df.drop_duplicates(subset=["comment"], keep="first")

        # ─────────────────────────────────────────────
        # 5. RESET INDEX (clean dataset hygiene)
        # ─────────────────────────────────────────────
        self.df = self.df.reset_index(drop=True)

        return self.df