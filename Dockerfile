# ── Stage 1: Build React frontend ───────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/react-app/package*.json ./
RUN npm ci --silent

COPY frontend/react-app/ ./
RUN npm run build
# Output: /app/frontend/dist/


# ── Stage 2: Python backend + embedded frontend ──────────────────────────────
FROM python:3.11-slim

# System deps needed by PyMuPDF + healthcheck curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# Install Python dependencies
COPY requirements.txt ../requirements.txt
RUN pip install --no-cache-dir -r ../requirements.txt

# Copy backend source
COPY backend/ ./

# Copy React build into backend/static/dist/ so FastAPI serves it
COPY --from=frontend-builder /app/frontend/dist/ ./static/dist/

# ChromaDB data persisted via a named volume (see docker-compose.yml)
VOLUME ["/app/chroma_db"]

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
