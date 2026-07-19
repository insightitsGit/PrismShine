# Ollama with the judge model baked in (reproducible, no startup pull).
# Build context = bench/shims/ollama
FROM ollama/ollama:0.5.4

ENV OLLAMA_HOST=0.0.0.0:11434
# Pull during build: start server, pull, stop.
RUN ollama serve & \
    sleep 5 && \
    ollama pull llama3.2:3b-instruct-q4_K_M && \
    pkill ollama || true

EXPOSE 11434
