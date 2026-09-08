# OrbitSight final competition Docker image (CPU only)
FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ORBITSIGHT_TEAM_NAME=OrbitSight \
    ORBITSIGHT_DATASET=/OrbitSight_dataset \
    ORBITSIGHT_WORK=/work \
    ORBITSIGHT_MODEL_DIR=/models/final

WORKDIR /app

# System deps kept minimal (CPU scientific stack wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-submission.txt /app/requirements-submission.txt
RUN pip install --no-cache-dir -r /app/requirements-submission.txt

COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY src /app/src
COPY models/final /models/final

RUN pip install --no-cache-dir --no-deps /app

# Non-root optional; keep root for organizer flexibility
RUN mkdir -p /work

# Automatic competition entrypoint — no interactive prompts
CMD ["python", "-m", "orbitsight.submission"]
