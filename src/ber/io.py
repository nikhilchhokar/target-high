"""Safe TSV loading/writing for the challenge file formats."""
import csv
from pathlib import Path

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path) -> pd.DataFrame:
    """Read a challenge TSV as all-string columns.

    keep_default_na=False stops names like "NA"/"None" becoming NaN;
    QUOTE_NONE stops a stray double quote from swallowing following rows.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def load_split(data_dir, split: str) -> pd.DataFrame:
    """Load <split>_source{1,2,3}.tsv and stack them with a `src` column (1/2/3)."""
    d = Path(data_dir) / split
    frames = []
    for s in (1, 2, 3):
        df = read_tsv(d / f"{split}_source{s}.tsv")
        missing = set(SOURCE_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"{split}_source{s}.tsv missing columns {missing}")
        df = df[SOURCE_COLS].copy()
        df["src"] = s
        frames.append(df)
    recs = pd.concat(frames, ignore_index=True)
    if recs.entity_id.duplicated().any():
        raise ValueError(f"duplicate entity_id in {split}")
    return recs


def load_ground_truth(path, s1_ids) -> dict:
    """Return {s1_id: set(matched ids)} for every S1 id (missing rows -> empty set)."""
    gt = read_tsv(path)
    truth = {}
    for sid, ids in zip(gt.source1_entity_id, gt.matched_entity_ids):
        truth[sid] = {x.strip() for x in ids.split(",") if x.strip()}
    n_missing = sum(1 for s in s1_ids if s not in truth)
    if n_missing:
        print(f"[warn] {n_missing} S1 ids absent from ground truth; treated as singletons")
    return {s: truth.get(s, set()) for s in s1_ids}


def write_id_lists(mapping: dict, s1_ids, path, col: str) -> None:
    """Write one row per S1 id; ids comma-joined with no spaces, empty for none."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for sid in s1_ids:
            ids = list(dict.fromkeys(mapping.get(sid, [])))  # dedupe, keep order
            f.write(f"{sid}\t{','.join(ids)}\n")
