# Build context = repository root.
FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1

COPY bench/shims/insight-stack/requirements.txt /tmp/req.txt
RUN pip install -r /tmp/req.txt

COPY pyproject.toml README.md /app/
COPY prismshine /app/prismshine
RUN pip install --no-deps /app

COPY bench/shims/insight-stack/app.py /app/serve/app.py
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
