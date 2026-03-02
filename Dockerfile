# Use official Python lightweight image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies (postgres client for healthchecks if needed)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Set Python path to ensure module imports work
ENV PYTHONPATH=.

# Run the FastAPI server
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
