import pandas as pd
import json
from ladybug.pipeline.templates.finetune_data import SYSTEM_FINETUNE, INSTRUCTION_FINETUNE

class DataTransformation:
    def __init__(self, labeled_data: pd.DataFrame):
        self.labeled_data = labeled_data

    def prepare_llm_finetuning_data(self) -> pd.DataFrame:
        """
        Prepares the labeled data for LLM fine-tuning and returns as a DataFrame.
        Expects 'comment' and 'intent' columns in labeled_data.
        """
        df = self.labeled_data

        # Check required columns
        if "comment" not in df.columns or "intent" not in df.columns:
            raise ValueError("DataFrame must contain 'comment' and 'intent' columns")

        system_message = SYSTEM_FINETUNE.substitute()

        records = []

        for _, row in df.iterrows():
            comment_text = str(row["comment"]).strip()
            predicted_intent = row["intent"]

            instruction = INSTRUCTION_FINETUNE.substitute(comment=comment_text)
            output = json.dumps({"predicted_intent": predicted_intent}, ensure_ascii=False)

            records.append({
                "system": system_message,
                "instruction": instruction,
                "input": "",
                "output": output,
                "history": []
            })

        return pd.DataFrame(records)
