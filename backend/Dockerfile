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
COPY . ./

# ChromaDB data persisted via a named volume (see docker-compose.yml)
VOLUME ["/app/chroma_db"]

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
