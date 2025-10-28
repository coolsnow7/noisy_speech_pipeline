from __future__ import annotations

import os
import time
from typing import Dict, Optional

import polars as pl


def _rel_key_expr(col: str) -> pl.Expr:
    # Normalize to relative key under data/: raw/.../clip.wav
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.replace_all(r"\\", "/")  # windows → posix
        .str.replace(r".*?/data/", "", literal=False)  # drop any .../data/ prefix
        .str.replace(r"^/+", "", literal=False)  # remove leading slashes
    )


def _atomic_replace(
    path_tmp: str, path_final: str, retries: int, sleep_ms: int
) -> None:
    last_err: Optional[BaseException] = None
    for _ in range(retries):
        try:
            os.replace(path_tmp, path_final)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(sleep_ms / 1000.0)
    # final attempt (raise if it fails)
    if last_err:
        raise last_err


def enrich_manifest_with_triage(
    source_manifest_path: str = "data/voices_manifest.parquet",
    decisions_csv_path: str = "data/triage_decisions.csv",
    output_path: str = "data/voices_manifest_enriched.parquet",
    derive_key_from: Optional[str] = None,  # e.g., "path" if manifest lacks "key"
    max_replace_retries: int = 5,
    replace_sleep_ms: int = 120,
) -> Dict[str, int]:
    """
    Join latest triage decisions (CSV) onto the manifest (Parquet) and write
    data/voices_manifest_enriched.parquet. Idempotent; latest wins on timestamp.

    Output columns added (nullable): triage_decision, triage_reviewed_at, triage_notes
    """
    # ---- manifest (Lazy) ----
    manifest = pl.scan_parquet(source_manifest_path)
    manifest_cols = set(manifest.columns)

    join_key_col = "key"
    if join_key_col not in manifest_cols:
        src_col = derive_key_from or ("path" if "path" in manifest_cols else None)
        if not src_col:
            raise ValueError(
                "Manifest is missing 'key' and no suitable derive_key_from column "
                "found (e.g., 'path'). Pass derive_key_from='path' (or another col)."
            )
        manifest = manifest.with_columns(_rel_key_expr(src_col).alias("key"))

    # ---- decisions (latest per key) ----
    try:
        decisions_eager = pl.read_csv(decisions_csv_path, try_parse_dates=False)
    except FileNotFoundError:
        decisions_eager = pl.DataFrame(
            {
                "key": pl.Series([], pl.Utf8),
                "decision": pl.Series([], pl.Utf8),
                "timestamp": pl.Series([], pl.Utf8),
                "reviewer": pl.Series([], pl.Utf8),
                "notes": pl.Series([], pl.Utf8),
            }
        )

    if decisions_eager.height == 0:
        latest_lazy = decisions_eager.lazy()
    else:
        # Parse timestamp as UTC with a safe try/fallback
        try:
            # newer Polars has .str.to_datetime
            decisions_eager = decisions_eager.with_columns(
                pl.col("timestamp")
                .str.to_datetime(strict=False)  # no utc kw
                .dt.replace_time_zone("UTC")  # make tz-aware
            )
        except AttributeError:
            # older Polars → use strptime
            decisions_eager = decisions_eager.with_columns(
                pl.col("timestamp")
                .str.strptime(pl.Datetime, strict=False)
                .dt.replace_time_zone("UTC")
            )

        latest_lazy = (
            decisions_eager.sort(["key", "timestamp"]).group_by("key").tail(1).lazy()
        )

    # ---- left join (Lazy on both sides) ----
    enriched_lazy = manifest.join(
        latest_lazy.select(
            "key",
            pl.col("decision").alias("triage_decision"),
            pl.col("timestamp").alias("triage_reviewed_at"),
            pl.col("notes").alias("triage_notes"),
        ),
        on="key",
        how="left",
    )

    # ---- materialize once ----
    out_df = enriched_lazy.collect()

    # Ensure triage columns exist & are nullable even if CSV was empty
    if "triage_decision" not in out_df.columns:
        out_df = out_df.with_columns(
            pl.lit(None).cast(pl.Utf8).alias("triage_decision")
        )
    if "triage_reviewed_at" not in out_df.columns:
        out_df = out_df.with_columns(
            pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("triage_reviewed_at")
        )
    if "triage_notes" not in out_df.columns:
        out_df = out_df.with_columns(pl.lit(None).cast(pl.Utf8).alias("triage_notes"))

    # ---- atomic write (Windows-safe) ----
    tmp_path = f"{output_path}.tmp"
    out_df.write_parquet(tmp_path)
    _atomic_replace(tmp_path, output_path, max_replace_retries, replace_sleep_ms)

    # ---- basic metrics ----
    manifest_rows = out_df.height
    decision_keys = (
        decisions_eager.select(pl.len()).item() if decisions_eager.height else 0
    )
    merged_non_null = int(out_df["triage_decision"].is_not_null().sum())

    return {
        "manifest_rows": manifest_rows,
        "decision_rows": int(decision_keys),
        "triage_applied": merged_non_null,
    }
