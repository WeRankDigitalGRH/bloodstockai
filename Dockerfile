FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app.py ./app.py
COPY frontend/ ./frontend/

RUN mkdir -p /app/uploads /app/results /app/data

ENV BAI_FRONTEND_DIR=/app/frontend
ENV BAI_UPLOAD_DIR=/app/uploads
ENV BAI_RESULTS_DIR=/app/results
ENV BAI_DB_PATH=/app/data/bloodstockai.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
