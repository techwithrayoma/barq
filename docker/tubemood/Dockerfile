FROM ghcr.io/astral-sh/uv:0.6.14-python3.11-bookworm

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libavif-dev pkg-config \
    libjpeg-dev \
    gcc unzip zip \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for caching)
COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole backend folder into /app
COPY backend/ . 

# Copy env file into /app
COPY docker/env/.env.fastapi /app/.env

# Make entrypoint executable
COPY docker/tubemood/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Start FastAPI with reload for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

