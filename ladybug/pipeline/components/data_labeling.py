from typing import List
from ladybug.store.llm.providers.openai_provider import OpenAIProvider
from ladybug.store.llm.templates.comment_intent import SYSTEM_RULES, USER_RULES
import json 
import pandas as pd 



class DataLabeling:
    def __init__(self, clean_data: pd.DataFrame, llm: OpenAIProvider):
        self.clean_data = clean_data
        self.llm = llm

    def generate_comment_intent(self):
        predictions = []

        comments = self.clean_data["comment"].tolist()

        for comment in comments:
            messages = [
                self.llm.construct_prompt(SYSTEM_RULES.safe_substitute(), role="system"),
                self.llm.construct_prompt(USER_RULES.substitute(comment=comment), role="user")
            ]

            response = self.llm.generate_text(chat_history=messages)
            raw_text = response["text"]

            try:
                intent = (
                    json.loads(
                        raw_text.replace("```json", "").replace("```", "").strip()
                    ).get("predicted_intent")
                )
            except json.JSONDecodeError:
                intent = None

            predictions.append({
                "comment": comment,
                "intent": intent,
                "raw": raw_text,
                "messages": messages,
                "prompt_tokens": response.get("prompt_tokens"),
                "completion_tokens": response.get("completion_tokens"),
                "total_tokens": response.get("total_tokens"),
            })

        return pd.DataFrame(predictions)

