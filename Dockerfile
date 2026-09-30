FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Kolkata

RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps fyers-apiv3==3.1.18

COPY core/ ./core/
COPY config/ ./config/
COPY scripts/ ./scripts/

ENTRYPOINT ["python", "scripts/cloud_job.py"]

CMD ["--roots", "intraday_archive_dir", "fyers_db_dir", "--", "python", "scripts/update_data.py", "--fyers"]
