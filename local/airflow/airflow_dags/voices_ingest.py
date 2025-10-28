import csv
import pathlib
import random
from datetime import datetime

import numpy as np
import polars as pl
import soundfile as sf
from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_DIR = pathlib.Path("/opt/airflow/data")
RAW_ROOT = DATA_DIR / "raw" / "voices" / "VOiCES_devkit"


def quality_scan(sample_n: int = 300, seed: int = 7):
    wavs = list(RAW_ROOT.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files under {RAW_ROOT}")

    rand = random.Random(seed)
    rand.shuffle(wavs)
    subset = wavs[:sample_n] if sample_n else wavs

    rows = []
    for wav in subset:
        rel = wav.relative_to(DATA_DIR).as_posix()
        try:
            y, sr = sf.read(wav, dtype="float32", always_2d=True)
            # compute over all samples/channels
            rms = float(np.sqrt(np.mean(y**2)))
            clip_ratio = float(np.mean(np.abs(y) > 0.99))
        except Exception as e:
            # if a file is unreadable, record nulls but keep the row
            rms, clip_ratio = None, None
            print(f"[warn] failed {rel}: {e}")

        rows.append({"path": rel, "rms": rms, "clip_ratio": clip_ratio})

    out = DATA_DIR / "triage.parquet"
    pl.DataFrame(rows).write_parquet(out)  # default compression is fine
    print(f"Wrote {len(rows)} rows to {out}")


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


def triage_merge():
    triage = pl.read_parquet(DATA_DIR / "triage.parquet")
    manifest = pl.read_csv(DATA_DIR / "voices_manifest.csv")
    merged = manifest.join(triage, on="path", how="left")
    out = DATA_DIR / "voices_manifest_enriched.csv"
    merged.write_csv(out)
    print(f"Wrote {len(merged)} rows to {out}")


with DAG(
    dag_id="voices_ingest",
    start_date=datetime(2025, 8, 1),
    schedule=None,
    catchup=False,
    tags=["voices", "ingest", "skeleton"],
) as dag:
    t_check_raw = PythonOperator(task_id="check_raw", python_callable=check_raw)

    # NEW task between check_raw and manifest
    t_quality_scan = PythonOperator(
        task_id="quality_scan",
        python_callable=quality_scan,
        op_kwargs={"sample_n": 300, "seed": 7},
    )

    t_build_manifest_csv = PythonOperator(
        task_id="build_manifest_csv", python_callable=build_manifest_csv
    )

    t_triage_merge = PythonOperator(
        task_id="triage_merge", python_callable=triage_merge
    )

    t_check_raw >> [t_quality_scan, t_build_manifest_csv]
    [t_quality_scan, t_build_manifest_csv] >> t_triage_merge
