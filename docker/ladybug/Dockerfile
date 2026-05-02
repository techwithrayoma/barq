# Now inside the container:
# /app/
#  └── ladybug/
#       ├── app/celery_app/
#       ├── core/
#       ├── db/
#       ├── pipeline/
#       └── requirements.txt
#  └── requirements.txt

FROM ghcr.io/astral-sh/uv:0.6.14-python3.11-bookworm

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
COPY ladybug/requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# 🔹 Copy the ladybug folder itself
COPY ladybug /app/ladybug

# Copy env file
COPY docker/env/.env.ladybug /app/.env

# Make entrypoint executable
COPY docker/ladybug/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Start FastAPI
CMD ["uvicorn", "ladybug.app.main:app", "--host", "0.0.0.0", "--port", "8001"]


