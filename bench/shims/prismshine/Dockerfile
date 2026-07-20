# Build context = repo root:  docker build -f bench/shims/prismshine/Dockerfile .
# For ONNX bake: ensure bench/shims/prismshine/baked/{model.onnx,model.onnx.data,tokenizer.json}
# exist locally; ACR pack uses a temporary gitignore that allows *.onnx (see deploy notes).
FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 HF_HOME=/models TRANSFORMERS_OFFLINE=0

COPY bench/shims/prismshine/requirements.txt /tmp/req.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/req.txt

COPY pyproject.toml README.md /app/
COPY prismshine /app/prismshine
RUN pip install --no-deps /app && pip install pydantic numpy pyyaml

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY bench/shims/prismshine/baked/ /app/span/
ENV PRISMSHINE_SPAN_ONNX=/app/span/model.onnx
ENV PRISMSHINE_SPAN_TOKENIZER=/app/span/tokenizer.json

COPY benchmarks/calibration/halueval_minilm.json /app/calibration/halueval_minilm.json
ENV PRISMSHINE_CALIBRATION=/app/calibration/halueval_minilm.json

COPY bench/shims/prismshine/app.py /app/serve/app.py
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
