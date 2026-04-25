# Start from the official Airflow image
FROM apache/airflow:2.7.1

# Set working directory
WORKDIR /opt/airflow

# Copy requirements.txt from the build context (docker/airflow/) into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
