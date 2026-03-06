FROM python:3.11-slim-bookworm

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    ghostscript \
    default-jre-headless \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    build-essential \
    curl \
    pkg-config \
    libpango1.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY etl/ ./etl/
COPY agents/ ./agents/
COPY tools/ ./tools/
COPY storage/ ./storage/
COPY api/ ./api/
COPY run_etl.py .
COPY tests/ ./tests/

RUN mkdir -p /app/data /app/logs /app/chroma_store

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]