from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict

import pandas as pd
import polars as pl
import streamlit as st

TRIAGE_PARQUET = Path("data/triage.parquet")
AUDIO_CANDIDATE_COLS = ["audio_path", "path", "filepath", "audio_file"]
WRITE_PATH = Path("data/triage_review.csv")  # output manifest
DATA_DIR = Path("data")
DECISIONS = DATA_DIR / "triage_decisions.csv"


@st.cache_data(show_spinner=False)
def load_triage_df(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


# This section is all about locked CSV read/write
@contextmanager
def _csv_flock(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # open the real CSV file handle (create if missing)
    fd = os.open(csv_path, os.O_RDWR | os.O_CREAT)
    try:
        with os.fdopen(fd, "r+b", buffering=0) as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # block until exclusive
            yield f  # you don't have to use f; lock is held until exit
    finally:
        # fd is closed by the context manager; flock releases automatically
        pass


def read_decisions_locked() -> pl.DataFrame:
    # Reads don't strictly need a lock, but doing so avoids read-while-write glitches.
    with _csv_flock(DECISIONS):
        if not DECISIONS.exists() or DECISIONS.stat().st_size == 0:
            return pl.DataFrame(
                {
                    "key": [],
                    "decision": [],
                    "timestamp": [],
                    "reviewer": [],
                    "notes": [],
                }
            )
        return pl.read_csv(str(DECISIONS))


def write_decisions_locked(df: pl.DataFrame) -> None:
    tmp = DECISIONS.with_suffix(DECISIONS.suffix + ".tmp")
    with _csv_flock(DECISIONS):
        df.write_csv(str(tmp))
        os.replace(tmp, DECISIONS)  # atomic


# This whole section deals with paths for the audio files #


def _strip_prefix(s: str, prefix: str) -> str:
    return s[len(prefix) :] if s.startswith(prefix) else s


def relative_key_from(val: str | Path) -> str:
    """
    Normalize an audio path into a repo-portable key under data/.
    Examples (all -> 'raw/voices/clip.wav'):
      '/opt/airflow/data/raw/voices/clip.wav'
      '/app/data/raw/voices/clip.wav'
      'C:\\...\\noisy_speech_pipeline\\data\\raw\\voices\\clip.wav'
      'raw/voices/clip.wav'
      'data/raw/voices/clip.wav'
    """
    s = str(val).strip().replace("\\", "/")
    if not s:
        return ""

    # Drop Windows drive like 'C:/'
    if len(s) >= 3 and s[1] == ":" and s[2] == "/":
        s = s[3:]

    # If the repo name appears, cut everything before it
    anchor = "/noisy_speech_pipeline/"
    if anchor in s:
        s = s.split(anchor, 1)[1]

    # Drop common container roots
    for p in ("/opt/airflow/", "/app/"):
        s = _strip_prefix(s, p)

    # Keep only the part after data/ if present
    if "/data/" in s:
        s = s.split("/data/", 1)[1]
    s = _strip_prefix(s, "data/")

    # Final tidy: no leading slash; forward slashes only
    return s.lstrip("/")


def key_to_path(column_name):
    raw_val = row[audio_col]

    s = str(raw_val).replace("\\", "/")  # normalize slashes from Windows
    p_rel = Path(s)
    cand1 = Path.cwd() / p_rel  # /app/raw/voices/...
    cand2 = Path.cwd() / "data" / p_rel  # /app/data/raw/voices/...
    audio_path = cand1 if cand1.exists() else cand2

    return audio_path, raw_val


def normalize_key(key: str, drop_prefix: str = "data/") -> str:
    if key is None:
        return ""
    k = str(key).strip().replace("\\", "/")  # Windows → POSIX
    if k.startswith(drop_prefix):
        k = k[len(drop_prefix) :]
    # keep case by default; uncomment next line if you want case-insensitive keys
    # k = k.lower()
    # collapse any accidental '//' after replacements
    k = str(PurePosixPath(k))
    return k


CSV_PATH = Path("data/triage_decisions.csv")


def _atomic_replace(src: Path, dst: Path, retries: int = 5, delay: float = 0.15):
    last_err = None
    for _ in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(delay)
    raise PermissionError(f"Could not replace {dst} with {src}: {last_err}")


def remove_decision(key: str, csv_path: Path = CSV_PATH) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if "key" not in df.columns:
        return
    new_df = df[df["key"] != key]
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    new_df.to_csv(tmp, index=False)
    _atomic_replace(tmp, csv_path)


# OK done with paths, now let's build some stuff

st.set_page_config(page_title="Triage Review", layout="wide")
st.title("Triage Review — One-Clip Preview")

if not TRIAGE_PARQUET.exists():
    st.error(f"Missing {TRIAGE_PARQUET}. Run your DAG until quality_scan emits it.")
    st.stop()

# load triage.parquet
df = load_triage_df(TRIAGE_PARQUET)
if df.height == 0:
    st.warning("triage.parquet is empty.")
    st.stop()

# find an audio path column, reformat to string
audio_col = next((c for c in AUDIO_CANDIDATE_COLS if c in df.columns), None)
if audio_col is None:
    st.error(
        f"triage.parquet is missing an audio path column. "
        f"Expected one of: {AUDIO_CANDIDATE_COLS}"
    )
    st.stop()

df = df.with_columns(pl.col(audio_col).cast(pl.Utf8).alias(audio_col))
# check that audio paths look like audio files
missing = (
    df.select(pl.col(audio_col).is_null() | (pl.col(audio_col) == "")).to_series().sum()
)
if missing:
    st.warning(f"{missing} rows have empty {audio_col} values; they’ll be unplayable.")

if "idx" not in st.session_state:
    st.session_state.idx = 0

N = df.height
progress_box = st.empty()
latest = pd.read_csv(Path("data/triage_decisions.csv"))

# Progress
pct = (st.session_state.idx + 1) / N
st.caption(f"Record Number {st.session_state.idx + 1} / {N}  ·  {pct:.1%}")

# Jump to index
jump = st.number_input("Jump to index", 0, N - 1, st.session_state.idx, 1)
if jump != st.session_state.idx:
    st.session_state.idx = int(jump)

on = st.toggle("Skip decided")
if on:
    st.session_state.skip = True
else:
    st.session_state.skip = False

row = df.row(st.session_state.idx, named=True)
audio_path, raw_val = key_to_path(audio_col)
row_key = relative_key_from(raw_val)

if "decisions" not in st.session_state:
    st.session_state.decisions = {}

existing_item = st.session_state.decisions.get(row_key)
existing_notes = ""
if isinstance(existing_item, dict):
    existing_notes = existing_item.get("notes", "")
elif isinstance(existing_item, str):
    existing_notes = ""  # older entries were strings only

notes = st.text_input("Notes (optional)", value=existing_notes, key=f"notes|{row_key}")

key_path = Path.cwd() / "data" / row_key


# Buttons that skip files with a decision already made

decided = {
    k
    for k, v in st.session_state.get("decisions", {}).items()
    if v and v != "— choose —"
}


def key_at(i: int) -> str:
    raw = df.row(i, named=True)[audio_col]
    return relative_key_from(raw)


def hop(idx: int, step: int) -> int:
    """step = +1 for Next, -1 for Prev"""
    j = idx + step
    if not on:
        return max(0, min(j, N - 1))
    # skip decided rows
    while 0 <= j < N and key_at(j) in decided:
        j += step
    return max(0, min(j, N - 1))


col_prev, col_next = st.columns(2)
if col_prev.button("◀ Prev", key="prev_btn"):
    st.session_state.idx = hop(st.session_state.idx, -1)

if col_next.button("Next ▶", key="next_btn"):
    st.session_state.idx = hop(st.session_state.idx, +1)

# More key path stuff
assert key_path.exists(), f"Key didn't map to a file: {key_path}"

if audio_path.exists():
    with audio_path.open("rb") as f:
        data = f.read()
    st.audio(data)
else:
    st.warning(f"Missing audio file: {audio_path}")

# show path and metadata
c1, c2, c3 = st.columns(3)
if "rms" in df.columns:
    c1.metric("RMS", f"{row.get('rms'):.3f}")
if "clip_ratio" in df.columns:
    c2.metric("Clip ratio", f"{row.get('clip_ratio')*100:.1f}%")
if "maybe_pii" in df.columns:
    c3.metric("PII flag", "Yes" if row.get("maybe_pii") else "No")


if "decisions" not in st.session_state:
    st.session_state.decisions = {}
existing = st.session_state.decisions.get(row_key)  # e.g., "keep" / "drop" / "review"
widget_key = f"decision:{row_key}"
if existing is not None and widget_key not in st.session_state:
    st.session_state[widget_key] = existing


decisions_df = pd.DataFrame(
    {
        "key": key_path.as_posix(),
        "decision": [None],
        "timestamp": [None],
        "reviewer": [None],
        "notes": [None],
    }
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


st.caption(f"CSV path: {(Path.cwd() / 'data/triage_decisions.csv').resolve()}")


@contextmanager
def _csv_lock(csv_path: Path, timeout_s: float = 3.0, poll_s: float = 0.05):
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    start = time.time()
    # acquire
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Could not acquire lock: {lock_path}")
            time.sleep(poll_s)
    try:
        yield
    finally:
        # release
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def save_decisions(
    decisions: Dict[str, Any],
    csv_path: Path = Path("data/triage_decisions.csv"),
    reviewer: str = "DR",
) -> int:
    """
    Merge in-memory decisions into CSV and write atomically.
    decisions can be either:
      {key: "keep"/"drop"/"review"}  OR
      {key: {"decision": "...", "timestamp": "...", "reviewer": "...", "notes": "..."}}
    Returns number of unique keys saved.
    """
    rows = []
    for key, val in decisions.items():
        if not key:
            continue
        key_n = normalize_key(key)  # boundary normalization
        if isinstance(val, dict):
            decision = val.get("decision")
            ts = val.get("timestamp") or _now_iso()
            rev = val.get("reviewer") or reviewer
            notes = val.get("notes") or ""
        else:
            decision = str(val) if val is not None else None
            ts, rev, notes = _now_iso(), reviewer, ""
        if decision in (None, "", "— choose —"):
            continue
        rows.append(
            {
                "key": key_n,
                "decision": decision,
                "timestamp": ts,
                "reviewer": rev,
                "notes": notes,
            }
        )

    new_df = pd.DataFrame(
        rows, columns=["key", "decision", "timestamp", "reviewer", "notes"]
    )

    # --- flocked read/merge/atomic write ---
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")

    with _csv_flock(csv_path):
        if csv_path.exists() and csv_path.stat().st_size > 0:
            old_df = pd.read_csv(csv_path)
            if not old_df.empty and "key" in old_df.columns:
                old_df["key"] = old_df["key"].astype(str).map(normalize_key)
            all_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            all_df = new_df

        if not all_df.empty:
            # ISO timestamps sort lexicographically; keep last per key
            all_df = all_df.sort_values(["key", "timestamp"]).drop_duplicates(
                subset="key", keep="last"
            )

        all_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, csv_path)  # atomic on Linux/NTFS inside Docker

    return int(all_df["key"].nunique()) if not all_df.empty else 0


def _clear_current(rk: str):
    # remove from in-memory store
    st.session_state.get("decisions", {}).pop(rk, None)
    # drop widget states so they re-initialize next run
    st.session_state.pop(f"decision|{rk}", None)
    st.session_state.pop(f"notes|{rk}", None)
    # remove from CSV on disk
    remove_decision(rk)
    # mark + rerun to re-create widgets with empty defaults
    st.session_state["_did_clear"] = True


st.button(
    "Clear decision for this clip",
    key=f"clear|{row_key}",
    on_click=_clear_current,
    args=(row_key,),
)

if st.session_state.pop("_did_clear", False):
    st.rerun()

decision = st.radio(
    "Decision",
    ["Accept", "Reject", "Review"],
    key=widget_key,
    index=None,
    horizontal=True,
)

# --- Normalize to dict & persist (autosave) ---
if decision:
    st.session_state.decisions[row_key] = {
        "decision": decision,
        "notes": notes or "",
    }
    save_decisions(st.session_state.decisions, reviewer="DR")

st.session_state.decisions[row_key] = decision
part_done = save_decisions(st.session_state.decisions, reviewer="DR")
st.write(st.session_state.decisions)

st.write(f"Progress: {part_done + 1} / {N}  ·  {pct:.1%} Reviewed")

# meta_keys = [k for k in ["rms", "clip_ratio", "maybe_pii"] if k in df.columns]
# meta_df = pd.DataFrame(
#    {"metric": meta_keys, "value": [row.get(k) for k in meta_keys]}
# )
# show metadata table
# st.dataframe(meta_df, hide_index=True, use_container_width=True)

# --- Progress bar ---

MANIFEST = DATA_DIR / "voices_manifest_enriched.parquet"
DECIDED_VALUES = ["Accept", "Reject", "Review"]

if DECISIONS.exists():
    latest = read_decisions_locked()
    if {"key", "decision"}.issubset(set(latest.columns)):
        done = (
            latest.select(
                pl.col("key").cast(pl.Utf8),
                pl.col("decision").cast(pl.Utf8),
            )
            .filter(pl.col("decision").is_in(DECIDED_VALUES))
            .select("key")
            .unique()
            .height
        )
    else:
        done = 0
else:
    done = 0


def load_manifest_keys_simple() -> pl.Series:
    if not MANIFEST.exists():
        st.warning(f"Missing manifest: {MANIFEST}")
        return pl.Series([], dtype=pl.Utf8)
    # unique keys in scope
    return (
        pl.scan_parquet(str(MANIFEST))
        .select(pl.col("key").cast(pl.Utf8))
        .unique()
        .collect()["key"]
    )


total = load_manifest_keys_simple().n_unique()
pct = round((done / total), 2) if total > 0 else 0.0

progress_text = "Samples Reviewed From Entire Dataset"
my_bar = st.progress(pct, text=f"{progress_text} {pct}%   ·  {done}/{total}")
