
# Builder Stage

FROM python:3.12-slim AS builder

WORKDIR /app

# Install build deps (only needed here)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY backend/requirements.txt .

# Install dependencies into a custom folder
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# Runtime Stage

FROM python:3.12-slim

WORKDIR /app

# Runtime libs required by OpenCV/MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY backend /app/backend

# Env optimizations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run app
CMD ["bash", "-c", "cd backend && PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000"]
