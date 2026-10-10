FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5001

# Install essential OS dependencies for headless OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files and trained models
COPY . .

# Ensure runtime directories exist
RUN mkdir -p static/uploads static/generated

EXPOSE 5001

# Run with Gunicorn production WSGI server
CMD ["sh", "-c", "gunicorn wsgi:app --bind 0.0.0.0:${PORT:-5001} --workers 2 --timeout 120"]
