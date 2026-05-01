from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
from app.core.config import get_settings

from app.entities.base import Base
from app.entities.comment import Comment
from app.entities.video import Video
from app.entities.model import Model
from app.entities.model_prediction import ModelPrediction

# Load application settings (e.g., DB URL)
settings = get_settings()

# Alembic configuration object, represents the .ini file
config = context.config

# If the Alembic .ini file has logging config, load it
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# SQLAlchemy metadata (used for autogenerate migrations)
target_metadata = Base.metadata

# Synchronous database URL (used for Alembic offline and online migrations)
DB_URL = settings.sync_database_url
config.set_main_option("sqlalchemy.url", DB_URL)


# Offline migration mode
def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Offline mode generates SQL scripts without needing a DB connection.
    Useful when you just want to see the SQL or apply it manually.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,  # Render values directly into SQL
        dialect_opts={"paramstyle": "named"},
    )

    # Start a transaction context and run migrations
    with context.begin_transaction():
        context.run_migrations()


# Online migration mode
def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Online mode connects to the database using SQLAlchemy Engine
    and applies migrations directly.
    """
    # Create a SQLAlchemy Engine using the config
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),  # Pull config from .ini
        prefix="sqlalchemy.",  # Only use keys starting with 'sqlalchemy.'
        poolclass=pool.NullPool,  # Disable connection pooling (safe for migrations)
    )

    # Connect and associate the connection with Alembic context
    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        # Run migrations in a transaction
        with context.begin_transaction():
            context.run_migrations()


# Run migrations based on mode
if context.is_offline_mode():
    # If Alembic is invoked in offline mode, use offline function
    run_migrations_offline()
else:
    # Otherwise, run online migrations
    run_migrations_online()
