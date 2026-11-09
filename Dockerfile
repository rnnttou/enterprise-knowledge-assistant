FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-ocr.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Scanned PDFs and images (Step 11). Off by default because PyTorch makes the
# image much bigger; enable with: docker compose build --build-arg WITH_OCR=true
# The CPU-only PyTorch wheels keep it around 1 GB instead of several.
ARG WITH_OCR=false
RUN if [ "$WITH_OCR" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/* \
        && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
           -r requirements-ocr.txt; \
    fi

COPY app/ app/

# data/ (PDFs) and chroma_db/ (the index) are mounted as volumes in
# docker-compose.yml, not baked into the image.

EXPOSE 8000 8501

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
