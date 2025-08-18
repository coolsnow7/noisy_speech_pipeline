import csv
import pathlib
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_DIR = pathlib.Path("/opt/airflow/data")
RAW_ROOT = DATA_DIR / "raw" / "voices" / "VOiCES_devkit"


def check_raw():
    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"{RAW_ROOT} not found. Mount correct path in docker-compose."
        )
    # sanity: find at least one wav
    if next(RAW_ROOT.rglob("*.wav"), None) is None:
        raise FileNotFoundError(f"No .wav files under {RAW_ROOT}")


def build_manifest_csv():
    out = DATA_DIR / "voices_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for wav in RAW_ROOT.rglob("*.wav"):
        rel = wav.relative_to(DATA_DIR).as_posix()
        rows.append(rel)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path"])
        for p in rows:
            w.writerow([p])
    print(f"Wrote {len(rows)} rows to {out}")


with DAG(
    dag_id="voices_ingest",
    start_date=datetime(2025, 8, 1),
    schedule=None,
    catchup=False,
    tags=["voices", "ingest", "skeleton"],
) as dag:
    t1 = PythonOperator(task_id="check_raw", python_callable=check_raw)
    t2 = PythonOperator(
        task_id="build_manifest_csv", python_callable=build_manifest_csv
    )
    t1 >> t2
