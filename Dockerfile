# syntax=docker/dockerfile:1
# ── Stage 1: builder ──────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build-time system deps needed by some Python packages (e.g. nibabel)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="wired87" \
      description="BrainMaster File MCP Server — processes research SPECT brain output files"

# Non-root user for production security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy only the application source
COPY brain_processor.py server.py ./

USER appuser

# Default: HTTP/SSE transport so the container acts as a network-accessible
# MCP server (override with `python server.py` for stdio transport).
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

ENTRYPOINT ["python", "server.py"]
CMD ["--http", "--host", "0.0.0.0", "--port", "8000"]
