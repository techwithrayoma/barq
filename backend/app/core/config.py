from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Supported databases:
        - PostgreSQL: async driver = asyncpg, sync driver = psycopg2
        - MySQL: async driver = aiomysql, sync driver = pymysql
        - SQLite: async driver = aiosqlite, sync driver = sqlite
    """

    # Database type and drivers
    DB_TYPE: str
    DB_ASYNC_DRIVER: str
    DB_SYNC_DRIVER: str

    # Database credentials
    DB_HOST: str
    DB_PORT: int
    DB_USERNAME: str
    DB_PASSWORD: str
    DB_NAME: str

    # Other keys
    YOUTUBE_API_KEY: str

    @property
    def async_database_url(self) -> str:
        """
        Returns an async SQLAlchemy database URL.
        Example for PostgreSQL: postgresql+asyncpg://user:pass@host:port/db
        """
        return (
            f"{self.DB_TYPE}+{self.DB_ASYNC_DRIVER}://"
            f"{self.DB_USERNAME}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

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
        env_file = str(Path(__file__).resolve().parents[2] / ".env")


def get_settings() -> Settings:
    return Settings()
