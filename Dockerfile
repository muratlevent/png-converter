# Use a slim Python base image
FROM python:3.11-slim

# Install system dependencies, including Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py converter.py ./
COPY templates/ ./templates/

# Run the web server using Gunicorn
# Using JSON array form with sh -c to allow graceful SIGTERM signal handling while resolving $PORT and setting a 120s timeout
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --timeout 120 app:app"]
