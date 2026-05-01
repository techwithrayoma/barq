import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from ladybug.core.logger import pipeline_logger

logger = pipeline_logger


class ModelEvaluation:
    """
    Evaluates a trained model against a fixed benchmark dataset.

    Usage (from TrainPipeline.run_model_training):
        benchmark_df = self.storage.load_benchmark()
        evaluator    = ModelEvaluation(model=trainer, benchmark_df=benchmark_df)
        metrics      = evaluator.evaluate()
    """

    def __init__(self, model, benchmark_df: pd.DataFrame = None, benchmark_path: str = None):
        """
        Args:
            model          : ModelTraining instance (or any inference wrapper with .predict())
            benchmark_df   : DataFrame with columns [comment, true_label]
                             Pass this directly to avoid file I/O coupling.
            benchmark_path : (optional) path to a CSV benchmark file.
                             benchmark_df takes priority if both are supplied.
        """
        self.model          = model
        self.benchmark_df   = benchmark_df
        self.benchmark_path = benchmark_path

    # ────────────────────────────────────────────────────────────────────────
    # LOAD
    # ────────────────────────────────────────────────────────────────────────

    def load_benchmark(self) -> pd.DataFrame:
        """
        Return the benchmark DataFrame.
        Priority: in-memory df  >  file path.
        """
        if self.benchmark_df is not None:
            logger.info(f"[EVAL] Using in-memory benchmark ({len(self.benchmark_df)} rows)")
            return self.benchmark_df

        if self.benchmark_path:
            logger.info(f"[EVAL] Loading benchmark from file: {self.benchmark_path}")
            return pd.read_csv(self.benchmark_path)

        raise ValueError(
            "ModelEvaluation requires either benchmark_df or benchmark_path. "
            "Got neither."
        )

    # ────────────────────────────────────────────────────────────────────────
    # PREDICT
    # ────────────────────────────────────────────────────────────────────────

    def predict(self, comment: str) -> str:
        """
        Run inference for a single comment.
        Replace self.model.predict() with your actual LLaMA-Factory inference wrapper.
        """
        return self.model.predict(comment)

    # ────────────────────────────────────────────────────────────────────────
    # EVALUATE
    # ────────────────────────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        """
        Run full evaluation loop and return metrics dict.

        Returns:
            {
                "accuracy":  float,
                "f1_score":  float,
                "total":     int,
                "correct":   int,
            }
        """
        df = self.load_benchmark()

        required_cols = {"comment", "true_label"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Benchmark DataFrame must have columns {required_cols}. "
                f"Found: {list(df.columns)}"
            )

        y_true = []
        y_pred = []

        for _, row in df.iterrows():
            comment    = row["comment"]
            true_label = row["true_label"]

            pred_label = self.predict(comment)

            y_true.append(true_label)
            y_pred.append(pred_label)

        accuracy = accuracy_score(y_true, y_pred)
        f1       = f1_score(y_true, y_pred, average="macro", zero_division=0)
        correct  = sum(t == p for t, p in zip(y_true, y_pred))

        metrics = {
            "accuracy":  round(float(accuracy), 6),
            "f1_score":  round(float(f1), 6),
            "total":     len(y_true),
            "correct":   correct,
        }

        logger.info(
            f"[EVAL] accuracy={metrics['accuracy']:.4f} | "
            f"f1={metrics['f1_score']:.4f} | "
            f"correct={correct}/{len(y_true)}"
        )

        return metrics