FROM python:3.12-slim

# Fail fast and log straight through - no buffered stdout, so the JSON
# log lines reach the platform's log stream as they happen.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render (and most PaaS) inject the port to bind as $PORT and route
# traffic there; 8000 is the fallback for plain `docker run` and
# docker-compose, which don't set it.
ENV PORT=8000
EXPOSE 8000

# Migrations and seeding run from render.yaml's preDeployCommand in
# production, so they happen once per deploy rather than once per
# replica. This default keeps `docker compose up` a single step.
CMD alembic upgrade head && python -m app.domain.seed && exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
