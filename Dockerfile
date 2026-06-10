FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY watcher.py index.py ./

ENV PYTHONPATH=/app/src

EXPOSE 8000

# Default entrypoint to web-ui, but can be overridden in docker-compose
CMD ["python", "-m", "repo_knowledge.web_ui.server"]
