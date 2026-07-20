# Build context = lean staging dir (see azure stack deploy) or repo root with matching COPY paths.
FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
ENV PRISMGUARD_USE_ONNX=1
ENV PRISMGUARD_APP_PROFILE=law_pilot
ENV HF_HOME=/models

COPY requirements.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt \
 && prismguard-model download

COPY pyproject.toml README.md /app/
COPY prismshine /app/prismshine
RUN pip install --no-deps /app && pip install pydantic numpy pyyaml

COPY serve/app.py /app/serve/app.py
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
