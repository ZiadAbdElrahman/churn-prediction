# Dockerfile — uv + pyproject.toml
FROM ghcr.io/astral-sh/uv:python3.11-bookworm

# Prevent .pyc files & ensure unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only dependency metadata first (better caching)
COPY pyproject.toml ./
# If you have a lock file, include it too (optional but recommended)
# COPY uv.lock ./

# Install deps into a project venv (created at .venv)
# --frozen avoids network if uv.lock is present
RUN uv sync --no-dev

# Add app code
COPY src ./src
COPY config ./config
# Optional: baked-in artifacts (you can also mount them at runtime)
COPY artifacts ./artifacts

# Make src importable
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]