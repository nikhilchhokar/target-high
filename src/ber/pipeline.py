"""Build normalised records -> candidates -> features for one split, with caching.

Cache keys include the relevant config sections, the input files' size/mtime AND a hash
of the source code of the stages, so editing normalize.py invalidates stale caches.
"""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import io, normalize, blocking, features

SRC_DIR = Path(__file__).parent
STAGE_FILES = ["normalize.py", "dictionaries.py", "blocking.py", "features.py"]


def load_config(path) -> dict:
    """Load YAML; `inherit: other.yaml` deep-merges this file on top of its parent."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_config(path.parent / parent)
        cfg = _deep_merge(base, cfg)
    return cfg


def _deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def stable_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _input_signature(data_dir, split):
    d = Path(data_dir) / split
    return [(p.name, p.stat().st_size, int(p.stat().st_mtime)) for p in sorted(d.glob("*.tsv"))]


def _code_signature():
    return hashlib.sha1(b"".join((SRC_DIR / f).read_bytes() for f in STAGE_FILES)).hexdigest()[:12]


def build_blocking(cfg: dict, split: str) -> dict:
    """Normalise + block only (no features, no cache) - for fast blocking experiments."""
    t0 = time.time()
    recs = normalize.normalize_records(io.load_split(cfg["paths"]["data_dir"], split))
    ti = blocking.TextIndex(recs)
    union, cands = blocking.run_blocking(recs, ti, cfg["blocking"])
    ids = recs.entity_id.values
    cands = cands.assign(s1=ids[cands.qi.values], cand=ids[cands.pj.values])
    return {"recs": recs, "union": union, "cands": cands, "seconds": time.time() - t0}


def build_split(cfg: dict, split: str, verbose=True) -> dict:
    """Return {'recs', 'union', 'feats', 'timings'} for split 'train' or 'test'."""
    data_dir = cfg["paths"]["data_dir"]
    key = stable_hash({"split": split, "inputs": _input_signature(data_dir, split),
                       "code": _code_signature(), "normalize": cfg.get("normalize"),
                       "blocking": cfg["blocking"], "features": cfg.get("features")})
    cdir = Path(cfg["paths"].get("cache_dir", "cache")) / f"{split}_{key}"
    if (cdir / "feats.parquet").exists():
        if verbose:
            print(f"[cache] {split}: {cdir}")
        return {"recs": pd.read_parquet(cdir / "recs.parquet"),
                "union": pd.read_parquet(cdir / "union.parquet"),
                "feats": pd.read_parquet(cdir / "feats.parquet"),
                "timings": json.loads((cdir / "timings.json").read_text())}
    t = {}
    t0 = time.time()
    recs = normalize.normalize_records(io.load_split(data_dir, split))
    t["normalize_s"] = time.time() - t0
    t0 = time.time()
    ti = blocking.TextIndex(recs)
    union, cands = blocking.run_blocking(recs, ti, cfg["blocking"])
    t["blocking_s"] = time.time() - t0
    t0 = time.time()
    feats = features.build_features(cands, recs, ti, cfg.get("features", {}))
    ids = recs.entity_id.values
    feats.insert(0, "s1", ids[feats.qi.values])
    feats.insert(1, "cand", ids[feats.pj.values])
    t["features_s"] = time.time() - t0
    if verbose:
        print(f"[build] {split}: {len(recs)} recs, union {len(union)}, cands {len(feats)}, "
              + ", ".join(f"{k}={v:.1f}" for k, v in t.items()))
    cdir.mkdir(parents=True, exist_ok=True)
    recs.to_parquet(cdir / "recs.parquet")
    union.to_parquet(cdir / "union.parquet")
    feats.to_parquet(cdir / "feats.parquet")
    (cdir / "timings.json").write_text(json.dumps(t))
    return {"recs": recs, "union": union, "feats": feats, "timings": t}
