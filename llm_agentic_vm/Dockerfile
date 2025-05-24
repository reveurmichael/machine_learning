FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Copy the example config if real config doesn't exist
RUN if [ ! -f config.yml ]; then cp config.example.yml config.yml; fi

# Default command
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

# Expose port
EXPOSE 8501 