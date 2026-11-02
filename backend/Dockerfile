# syntax=docker/dockerfile:1
FROM python:3.11-slim AS build
WORKDIR /w
COPY requirements.txt .
# build tooling is removed from the runtime venv: not needed, and its vendored copies (wheel, jaraco.context) carried HIGH CVEs
RUN python -m venv /venv && /venv/bin/pip install --no-cache-dir -r requirements.txt \
 && /venv/bin/pip uninstall -y setuptools wheel pip

FROM python:3.11-slim
# libgomp1: Prophet's Stan binaries need OpenMP. No compiler, no pip, no shell tools beyond the base image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --create-home app \
 && python -m pip uninstall -y setuptools wheel pip
COPY --from=build /venv /venv
WORKDIR /srv
COPY app ./app
COPY alembic ./alembic
COPY ui ./ui
ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/home/app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
# one worker: detection is CPU-bound and synchronous; scale by replicas (state is all external), not by workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
