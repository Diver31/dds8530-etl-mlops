FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DDS8530_MODEL_DIR=/app/model_release
WORKDIR /app
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/deployment/requirements.txt \
    && useradd --system --create-home --uid 10001 service
COPY api /app/api
COPY model_release /app/model_release
RUN chown -R service:service /app
USER service
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "60", "api.release_server:app"]
