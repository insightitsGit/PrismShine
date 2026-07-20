# Build context = bench/shims/oss-llmguard
FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/models

COPY requirements.txt /tmp/req.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/req.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app.py /app/serve/app.py
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
