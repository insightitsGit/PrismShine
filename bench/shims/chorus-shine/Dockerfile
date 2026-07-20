# Build from repository root:
#   docker build -f bench/shims/chorus-shine/Dockerfile -t chorus-shine:v1 .
FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
ENV CHORUSGRAPH_ALLOW_HASH_EMBEDDER=1
ENV HF_HOME=/models

COPY bench/shims/chorus-shine/requirements.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt \
 && pip uninstall -y prismlib || true \
 && pip install "prismlib-plus>=0.8.0"

COPY pyproject.toml README.md /app/
COPY prismshine /app/prismshine
RUN pip install --no-deps /app && pip install pydantic numpy pyyaml

COPY bench/shims/chorus-shine/app.py /app/serve/app.py
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
