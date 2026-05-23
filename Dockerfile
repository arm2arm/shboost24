FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files into the image
COPY . /app
WORKDIR /app/src

# Entrypoint script to start MLflow server and run the training script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# MLflow server port (we start it on 5123 by default to match existing scripts)
EXPOSE 5123

CMD ["/app/entrypoint.sh"]

