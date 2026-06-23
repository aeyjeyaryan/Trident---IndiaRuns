FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy sample data and pre-built index first (caching layer)
COPY sample_candidates.jsonl /app/candidates.jsonl
COPY data/ /app/data/

# Copy code
COPY trident/ /app/trident/

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app/trident
ENV TRIDENT_DATA_PATH=/app/candidates.jsonl
ENV TRIDENT_INDEX_PATH=/app/data/faiss_index.bin
ENV TRIDENT_METADATA_PATH=/app/data/metadata.json

CMD ["sleep", "infinity"]
