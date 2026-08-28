# Multi-stage build for the ResearchOps Agent API.
# Validated in CI (we don't run Docker locally — see docs/ROADMAP.md).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- deps layer (cached unless pyproject changes) ---
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# --- runtime ---
FROM python:3.12-slim
WORKDIR /app
COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin /usr/local/bin
COPY src ./src

EXPOSE 8000
CMD ["uvicorn", "researchops.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
