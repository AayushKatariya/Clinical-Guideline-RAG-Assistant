FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CLINRAG_HOME=/app

WORKDIR /app

# CPU-only torch keeps the image small; swap the index URL for CUDA builds.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[models]"

COPY prompts ./prompts
COPY sql ./sql

EXPOSE 8000
CMD ["uvicorn", "clinrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
