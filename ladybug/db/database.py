from typing import Annotated, Generator
from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ladybug.core.config import get_settings

# Load configuration
settings = get_settings()
DATABASE_URL = settings.sync_database_url

# Create sync engine and session factory
engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)
