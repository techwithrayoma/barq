from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    
    # Database type and drivers
    DB_TYPE: str
    DB_SYNC_DRIVER: str

    # Database credentials
    DB_HOST: str
    DB_PORT: int
    DB_USERNAME: str
    DB_PASSWORD: str
    DB_NAME: str
    
    # LLM Configuration
    LLM_PROVIDER: str
    OPENAI_API_KEY: str
    OPENAI_API_BASE_URL: str
    OPENAI_MODEL_ID: str
    OPENAI_MAX_INPUT_CHARS: int
    OPENAI_MAX_OUTPUT_TOKENS: int
    OPENAI_TEMPERATURE: float
    OPENAI_INPUT_PRICING: float
    OPENAI_OUTPUT_PRICING: float



    # Cloud Storage Configuration
    STORAGE_BACKEND: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_ACCESS_KEY_ID: str
    AWS_REGION: str
    S3_BUCKET_NAME: str
    
    # Cloud Training Configuration
    TRAINING_BACKEND: str

    # Celery Configuration
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    CELERY_TASK_SERIALIZER: str
    CELERY_TASK_TIME_LIMIT: int
    CELERY_TASK_ACKS_LATE: bool
    CELERY_WORKER_CONCURRENCY_CPU: int
    CELERY_WORKER_CONCURRENCY_GPU: int

    @property
    def sync_database_url(self) -> str:
        """
        Returns a sync SQLAlchemy database URL.
        Example for PostgreSQL: postgresql://user:pass@host:port/db
        """
        return (
            f"{self.DB_TYPE}://{self.DB_USERNAME}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    class Config:
        # Go two levels up from current file to find .env
        env_file = str(Path(__file__).resolve().parents[1] / ".env")


def get_settings() -> Settings:
    return Settings()
