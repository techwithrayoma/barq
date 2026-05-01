from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Boolean

from .base import Base


class Model(Base):
    """
    Represents a trained ML model version.
    """

    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)

    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)

    training_run_type = Column(String, nullable=False)
    # "runpod" | "local"

    dataset_dvc_hash = Column(String)

    mlflow_run_id = Column(String, nullable=True)

    accuracy = Column(String, nullable=True)
    f1_score = Column(String, nullable=True)

    total_training_cost_usd = Column(String, nullable=True)

    gpu_type = Column(String, nullable=True)
    gpu_hours = Column(String, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )