from string import Template
import time
import json
import pandas as pd

from ladybug.store.llm.providers.openai_provider import OpenAIProvider
from ladybug.store.llm.templates.comment_intent import SYSTEM_RULES, USER_RULES
from ladybug.core.logger import pipeline_logger

logger = pipeline_logger


VALID_LABELS = {
    "Question",
    "Complaint",
    "Statement",
    "Praise",
    "Suggestion"
}


class DataLabeling:
    """
    Handles hybrid labeling:
    - Rule/model-based labels (free)
    - LLM-based labels (costly)
    """

    CONFIDENCE_THRESHOLD = 0.95

    def __init__(self, clean_data: pd.DataFrame, llm: OpenAIProvider):
        self.clean_data = clean_data
        self.llm = llm

        self.cost_tracker = {
            "total_cost": 0.0,
            "llm_cost": 0.0,
            "llm_count": 0,
            "rule_model_count": 0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
        }


    def _should_use_model_label(self, row) -> bool:
        label = row.get("predicted_label")
        confidence = row.get("confidence_score")

        if label is None or confidence is None:
            return False

        return float(confidence) >= self.CONFIDENCE_THRESHOLD


    def _extract_valid_intent(self, raw_text: str, comment: str):
        try:
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(cleaned)

            intent = parsed.get("predicted_intent")

            # normalize
            if isinstance(intent, str):
                intent = intent.strip().capitalize()

            # enforce valid labels
            if intent not in VALID_LABELS:
                logger.warning(f"[DataLabeling] Invalid label: {intent} → fallback")
                return "Statement"   # ✅ fallback instead of None

            return intent

        except Exception:
            logger.warning(f"[DataLabeling] JSON parse failed: {comment[:50]}")
            return "Statement"       # ✅ fallback instead of None

    # ─────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────
    def generate_comment_intent(self) -> tuple[pd.DataFrame, list, dict]:

        predictions = []
        db_records = []

        for _, row in self.clean_data.iterrows():
            comment = row["comment"]
            comment_id = row.get("comment_id")

            # ───────── RULE / MODEL PATH (FREE) ─────────
            if self._should_use_model_label(row):

                intent = row.get("predicted_label")

                if intent not in VALID_LABELS:
                    logger.warning(f"[RuleModel] Invalid label: {intent} → fallback")
                    intent = "Statement"

                predictions.append({
                    "comment": comment,
                    "intent": intent,
                    "source": row.get("model"),
                    "confidence": row.get("confidence_score"),
                    "latency_ms": row.get("latency_ms"),
                    "cost_usd": 0.0,
                    "stage": "training",
                })

                self.cost_tracker["rule_model_count"] += 1
                continue

            # ───────── LLM PATH (COSTLY) ─────────
            start = time.time()

            messages = [
                self.llm.construct_prompt(SYSTEM_RULES.safe_substitute(), role="system"),
                self.llm.construct_prompt(USER_RULES.substitute(comment=comment), role="user"),
            ]

            response = self.llm.generate_text(chat_history=messages)

            latency_ms = (time.time() - start) * 1000

            intent = self._extract_valid_intent(response["text"], comment)

            # ── cost calculation ──
            prompt_tokens = response.get("prompt_tokens", 0)
            completion_tokens = response.get("completion_tokens", 0)
            cost_usd = self.llm.estimate_cost(prompt_tokens, completion_tokens)

            # ── update tracker ──
            self.cost_tracker["llm_cost"] += cost_usd
            self.cost_tracker["total_cost"] += cost_usd
            self.cost_tracker["llm_count"] += 1

            self.cost_tracker["total_tokens_in"] += prompt_tokens
            self.cost_tracker["total_tokens_out"] += completion_tokens

            # ── prediction record ──
            predictions.append({
                "comment": comment,
                "intent": intent,
                "source": self.llm.generation_model_id,
                "confidence": 0.95,
                "latency_ms": round(latency_ms, 2),
                "cost_usd": round(cost_usd, 6),
                "stage": "training",
            })

            # ── DB record ──
            db_records.append({
                "comment_id": comment_id,
                "model_id": None,
                "stage": "labeling",
                "source": self.llm.generation_model_id,
                "predicted_label": intent,
                "confidence_score": 0.95,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
            })

        # ── final log ──
        logger.info(
            f"[DataLabeling] Done — "
            f"rule_model={self.cost_tracker['rule_model_count']}, "
            f"llm={self.cost_tracker['llm_count']}, "
            f"cost=${self.cost_tracker['total_cost']:.4f}"
        )

        return (
            pd.DataFrame(predictions),
            db_records,
            self.cost_tracker
        )