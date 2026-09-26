"""Config loading and the scalable train / test drivers.

train: normalise (parallel, cached) -> sample S1 entities -> per country: fit index on the
       FULL pool, retrieve for the sampled S1 -> features -> cached parquet.
test : same per country for ALL S1, yielded in chunks so memory stays flat.
The pool is never sampled, so candidate density matches the real test setting.
"""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

from . import prep, retrieve, featurize

SRC_DIR = Path(__file__).parent
STAGE_FILES = ["retrieve.py", "featurize.py", "pipeline.py"]
Q_COLS = ["entity_id", "bc", "src", "business_address"] + featurize.VIEW_COLS


def load_config(path) -> dict:
    """Load YAML; `inherit: other.yaml` deep-merges this file on top of its parent."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    return cfg


def _deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def stable_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _code_hash():
    return hashlib.sha1(b"".join((SRC_DIR / f).read_bytes() for f in STAGE_FILES)).hexdigest()[:10]


def _input_sig(cfg, split):
    d = Path(cfg["paths"]["data_dir"]) / split
    return [(p.name, p.stat().st_size) for p in sorted(d.glob("*.tsv"))]


def _country_runs(cfg, split, s1_all, q_filter=None):
    """Yield (bc, query_chunk, union, feats) for each country and query chunk."""
    bcfg = cfg["blocking"]
    workers, threads = cfg.get("workers", 7), cfg.get("threads", 7)
    qchunk = bcfg.get("query_chunk", 200_000)
    for bc in sorted(s1_all.bc.unique()):
        t0 = time.time()
        pool = prep.load(cfg, split, [2, 3], Q_COLS, bc=bc)
        s1c = s1_all[s1_all.bc == bc]
        q = s1c if q_filter is None else s1c[s1c.entity_id.isin(q_filter)]
        if len(pool) == 0 or len(q) == 0:
            yield bc, q.reset_index(drop=True), None, None
            continue
        idx = retrieve.CountryIndex(pool, s1c, bcfg, threads)
        print(f"[index] {split}/{bc}: pool {len(pool)}, S1 {len(s1c)}, queries {len(q)} ({time.time() - t0:.0f}s)")
        # chunk by region so competing S1 (same region) are scored together -> reverse features
        q = q.assign(region=retrieve.region_of(q)).sort_values("region", kind="stable").reset_index(drop=True)
        bounds, start = [], 0
        reg_end = q.groupby("region", sort=False).size().cumsum().values
        for e in reg_end:
            if e - start >= qchunk:
                bounds.append((start, e))
                start = e
        if start < len(q):
            bounds.append((start, len(q)))
        for s, e in bounds:
            t1 = time.time()
            qq = q.iloc[s:e].reset_index(drop=True)
            union, cands = idx.query(qq)
            f = featurize.compute(cands, qq, idx, workers)
            ids_q, ids_p = qq.entity_id.values, idx.pool.entity_id.values
            f.insert(0, "s1", ids_q[f.qi.values])
            f.insert(1, "cand", ids_p[f.pj.values])
            union = union.assign(s1=ids_q[union.qi.values], cand=ids_p[union.pj.values])
            print(f"   {bc} queries {s}-{e}: union {len(union)}, cands {len(f)} "
                  f"({time.time() - t1:.0f}s)")
            yield bc, qq, union, f.drop(columns=["qi", "pj"])
        del idx, pool


def _pruner_sig(cfg):
    p = cfg["blocking"].get("prune", {}).get("model")
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()[:10] if p and Path(p).exists() else None


def train_sample_ids(cfg: dict, s1_all: pd.DataFrame) -> set:
    """Training S1 sample. mode 'region' takes WHOLE regions (US/Indian states), so every sampled
    entity's local competitors are sampled too - as at test time, where all S1 are scored."""
    scfg = cfg.get("sample", {})
    n = scfg.get("n_s1")
    if not n:
        return set(s1_all.entity_id)
    if scfg.get("mode", "random") != "region":
        return set(s1_all.entity_id.sample(n=min(n, len(s1_all)), random_state=scfg.get("seed", 42)))
    reg = retrieve.region_of(s1_all).values
    bc = s1_all.bc.values
    known = reg != ""
    frac_target = n / len(s1_all)
    take = np.zeros(len(s1_all), bool)
    rng = np.random.default_rng(scfg.get("seed", 42))
    for c in np.unique(bc):  # per country, so every country is represented in proportion
        m = (bc == c) & known
        sizes = pd.Series(reg[m]).value_counts()
        target = frac_target * m.sum()
        order = sizes.sample(frac=1.0, random_state=scfg.get("seed", 42))
        # prefer regions no bigger than ~2x the target so a single giant state cannot dominate
        order = pd.concat([order[order <= max(target / 2, 1)], order[order > max(target / 2, 1)]])
        chosen, total = [], 0
        for k, cnt in order.items():
            if total >= target:
                break
            if total + cnt > 1.5 * target and total > 0.5 * target:
                continue
            chosen.append(k)
            total += cnt
        take |= m & np.isin(reg, chosen)
    take |= (~known) & (rng.random(len(s1_all)) < frac_target)  # no region: random share
    return set(s1_all.entity_id.values[take])


def build_train(cfg: dict) -> dict:
    """Features for a sample of train S1 entities (cached)."""
    key = stable_hash({"inputs": _input_sig(cfg, "train"), "norm": prep.code_hash(), "pruner": _pruner_sig(cfg),
                       "code": _code_hash(), "blocking": cfg["blocking"],
                       "features": cfg.get("features"), "sample": cfg.get("sample")})
    cdir = Path(cfg["paths"].get("cache_dir", "cache")) / f"train_{key}"
    if (cdir / "feats.parquet").exists():
        print(f"[cache] train: {cdir}")
        return {"feats": pd.read_parquet(cdir / "feats.parquet"),
                "union": pd.read_parquet(cdir / "union.parquet"),
                "s1": pd.read_parquet(cdir / "s1.parquet"),
                **json.loads((cdir / "meta.json").read_text())}
    t0 = time.time()
    prep.prep_split(cfg, "train")
    t_prep = time.time() - t0
    s1_all = prep.load(cfg, "train", [1], Q_COLS)
    sample_set = train_sample_ids(cfg, s1_all)
    feats, unions, n_pool = [], [], 0
    for bc, qq, union, f in _country_runs(cfg, "train", s1_all, sample_set):
        if f is not None:
            feats.append(f)
            unions.append(union[["s1", "cand", "prune_rank"]])
    n_pool = int(ds.dataset([str(p) for s in (2, 3) for p in prep.norm_dir(cfg, "train").glob(f"s{s}_*.parquet")],
                            format="parquet").count_rows())
    s1 = s1_all.loc[s1_all.entity_id.isin(sample_set), ["entity_id", "bc"]].reset_index(drop=True)
    out = {"feats": pd.concat(feats, ignore_index=True), "union": pd.concat(unions, ignore_index=True),
           "s1": s1, "n_pool": n_pool, "timings": {"prep_s": t_prep, "build_s": time.time() - t0}}
    cdir.mkdir(parents=True, exist_ok=True)
    out["feats"].to_parquet(cdir / "feats.parquet")
    out["union"].to_parquet(cdir / "union.parquet")
    s1.to_parquet(cdir / "s1.parquet")
    (cdir / "meta.json").write_text(json.dumps({"n_pool": n_pool, "timings": out["timings"]}))
    return out


def iter_test(cfg: dict):
    """Yield (bc, query_chunk, feats) over ALL test S1 entities."""
    prep.prep_split(cfg, "test")
    s1_all = prep.load(cfg, "test", [1], Q_COLS)
    for bc, qq, union, f in _country_runs(cfg, "test", s1_all):
        yield bc, qq, f


def load_raw(cfg: dict, split: str, ids) -> pd.DataFrame:
    """Raw name/address/country for a set of ids (error analysis)."""
    d = prep.norm_dir(cfg, split)
    dset = ds.dataset([str(p) for p in d.glob("s*_*.parquet")], format="parquet")
    t = dset.to_table(columns=["entity_id", "business_name", "business_address", "bc"],
                      filter=ds.field("entity_id").isin(list(ids)))
    return t.to_pandas().rename(columns={"bc": "country"})
