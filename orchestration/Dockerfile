FROM apache/airflow:3.1.5-python3.13

USER airflow

COPY requirements-worker.txt /tmp/requirements-worker.txt

RUN python -m venv /home/airflow/dds8530-venv \
    && /home/airflow/dds8530-venv/bin/python -m pip install \
       --no-cache-dir -r /tmp/requirements-worker.txt
