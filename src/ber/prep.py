"""Parallel normalisation of every source file into parquet parts.

23M records x ~0.2 ms each is ~80 CPU-minutes, so chunks are normalised in worker
processes and written as parquet parts. The output folder is keyed by a hash of the
normalisation code, so editing normalize.py / dictionaries.py / translit.py re-runs it.
"""
import csv
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from . import normalize

SRC_DIR = Path(__file__).parent
CODE_FILES = ["normalize.py", "dictionaries.py", "translit.py", "prep.py"]
KEEP = ["entity_id", "src", "bc", "business_name", "business_address", "name_core", "legal",
        "dba_core", "name_skel", "name_nums", "name_acr", "name_indic", "addr_core", "postal",
        "house_nums", "units", "landmark"]


def code_hash() -> str:
    return hashlib.sha1(b"".join((SRC_DIR / f).read_bytes() for f in CODE_FILES)).hexdigest()[:10]


def _norm_chunk(df: pd.DataFrame, src: int) -> pd.DataFrame:
    for c in df.columns:
        df[c] = df[c].str.strip()
    df = df.assign(src=src)
    return normalize.normalize_records(df)[KEEP]


def norm_dir(cfg: dict, split: str) -> Path:
    return Path(cfg["paths"].get("work_dir", "work")) / f"norm_{code_hash()}" / split


def prep_split(cfg: dict, split: str, workers: int = None, chunk: int = 100_000) -> Path:
    """Normalise <split>_source{1,2,3}.tsv into norm_<hash>/<split>/s<k>_<i>.parquet (cached)."""
    out = norm_dir(cfg, split)
    if (out / "_DONE").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or cfg.get("workers", 7)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for s in (1, 2, 3):
            path = Path(cfg["paths"]["data_dir"]) / split / f"{split}_source{s}.tsv"
            reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                 quoting=csv.QUOTE_NONE, chunksize=chunk, encoding="utf-8")
            pending, n = {}, 0
            for i, df in enumerate(reader):
                if len(pending) >= workers * 2:  # bounded queue keeps memory flat
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result().to_parquet(pending.pop(f), index=False)
                pending[ex.submit(_norm_chunk, df, s)] = out / f"s{s}_{i:05d}.parquet"
                n += len(df)
            for f in pending:
                f.result().to_parquet(pending[f], index=False)
            print(f"[prep] {split} source{s}: {n} rows ({time.time() - t0:.0f}s)")
    (out / "_DONE").write_text("ok")
    return out


def load(cfg: dict, split: str, srcs, columns=None, bc=None) -> pd.DataFrame:
    """Load normalised records for the given sources (and optionally one country)."""
    d = norm_dir(cfg, split)
    files = sorted(str(p) for s in srcs for p in d.glob(f"s{s}_*.parquet"))
    dataset = ds.dataset(files, format="parquet")
    flt = (ds.field("bc") == bc) if bc is not None else None
    return dataset.to_table(columns=columns, filter=flt).to_pandas()
