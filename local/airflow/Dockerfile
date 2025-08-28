FROM apache/airflow:2.9.3-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Python deps used by quality_scan
RUN pip install --no-cache-dir polars pyarrow soundfile tqdm numpy