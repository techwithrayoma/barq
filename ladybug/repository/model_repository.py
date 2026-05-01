from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timezone


class ModelRepository:
    """
    Stores model training metadata (1 row per training run)
    """

    def __init__(self, db: Session):
        self.db = db

    def create_model_run(
        self,
        model_name: str,
        model_version: str,
        mlflow_run_id: str,
        gpu_type: str,
        gpu_hours: float,
        cost_usd: float,
        dataset_hash: str | None = None,
    ):
        self.db.execute(
            text("""
                INSERT INTO models (
                    model_name,
                    model_version,
                    training_run_type,
                    mlflow_run_id,
                    gpu_type,
                    gpu_hours,
                    total_training_cost_usd,
                    dataset_dvc_hash,
                    created_at
                )
                VALUES (
                    :model_name,
                    :model_version,
                    'runpod',
                    :mlflow_run_id,
                    :gpu_type,
                    :gpu_hours,
                    :cost_usd,
                    :dataset_hash,
                    :created_at
                )
            """),
            {
                "model_name": model_name,
                "model_version": model_version,
                "mlflow_run_id": mlflow_run_id,
                "gpu_type": gpu_type,
                "gpu_hours": gpu_hours,
                "cost_usd": cost_usd,
                "dataset_hash": dataset_hash,
                "created_at": datetime.now(timezone.utc),
            }
        )

        self.db.commit()