# Build from repository root:
#   docker build -f bench/shims/insight-stack/Dockerfile -t insight-stack:v4 .
FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
ENV PRISMGUARD_USE_ONNX=1
ENV PRISMGUARD_APP_PROFILE=law_pilot
ENV PRISMGUARD_DOMAIN=law
ENV PRISMGUARD_SEED_PROFILE=authored
ENV PRISMGUARD_FEEDBACK_PERSIST=1
ENV PRISMGUARD_STORAGE_BACKEND=memory
ENV CHORUSGRAPH_ALLOW_HASH_EMBEDDER=1
ENV HF_HOME=/models

COPY bench/shims/insight-stack/requirements.txt /tmp/req.txt
# Install stack deps; force prismlib-plus if anything pulled bare prismlib.
RUN pip install -r /tmp/req.txt \
 && pip uninstall -y prismlib || true \
 && pip install "prismlib-plus>=0.8.0" \
 && prismguard-model download

COPY pyproject.toml README.md /app/
COPY prismshine /app/prismshine
RUN pip install --no-deps /app && pip install pydantic numpy pyyaml

COPY bench/shims/insight-stack/app.py /app/serve/app.py
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
