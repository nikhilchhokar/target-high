"""Per-S1 training weights that match the TEST set's ambiguity profile (decoy density).

    python tools/density_weights.py --config configs/exp/e011_hop.yaml --out <cache>/density_w_e011.parquet

Needs the source experiment's out-of-fold predictions (run_experiment) and its saved test chunk
scores (make_submission); paths are resolved from the config (or given with --oof/--s1/--test-chunks).

The test pool has ~1.1 more unmatched records per business than train, so test businesses more
often have several borderline candidates. Bucket = number of candidates with 0.1 < p < 0.9
(capped at 4); weight = P_test(bucket | country) / P_train(bucket | country), clipped to [0.5, 4].
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def bucket(df):
    return ((df.p > 0.1) & (df.p < 0.9)).groupby(df.s1).sum().clip(upper=4)


def resolve(config):
    """(oof, train s1, test chunk dir) of an experiment, using the pipeline's own cache keys."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ber import pipeline, prep
    cfg = pipeline.load_config(config)
    cache = Path(cfg["paths"].get("cache_dir", "cache"))
    key = pipeline.stable_hash({"inputs": pipeline._input_sig(cfg, "train"), "norm": prep.code_hash(),
                                "pruner": pipeline._pruner_sig(cfg), "code": pipeline._code_hash(),
                                "blocking": cfg["blocking"], "features": cfg.get("features"),
                                "sample": cfg.get("sample")})
    return (str(Path(cfg["paths"].get("results_dir", "results")) / cfg["exp_id"] / "oof.parquet"),
            str(cache / f"train_{key}" / "s1.parquet"),
            str(cache / f"test_chunks_{cfg['exp_id']}_{pipeline.stable_hash(cfg)}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="source experiment config; fills in the three paths below")
    ap.add_argument("--oof")
    ap.add_argument("--s1")
    ap.add_argument("--test-chunks")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.config:
        a.oof, a.s1, a.test_chunks = [x or y for x, y in zip((a.oof, a.s1, a.test_chunks), resolve(a.config))]
    oof = pd.read_parquet(a.oof, columns=["s1", "p"])
    s1 = pd.read_parquet(a.s1)
    ctry = dict(zip(s1.entity_id, s1.bc))
    bo = bucket(oof)
    rows = []
    for c in ("in", "us"):
        te = pd.concat([pd.read_parquet(f, columns=["s1", "p"]) for f in glob.glob(f"{a.test_chunks}/{c}_*.parquet")])
        pt = bucket(te).value_counts(normalize=True)
        ids = [s for s in s1.entity_id if ctry[s] == c]
        b = bo.reindex(ids).fillna(0)
        po = b.value_counts(normalize=True)
        w = np.clip([pt.get(x, 0) / po.get(x, 1e-9) for x in b.values], 0.5, 4.0)
        rows.append(pd.DataFrame({"s1": ids, "w": w}))
        print(c, "train", po.round(3).to_dict(), "\n   test", pt.round(3).to_dict())
    out = pd.concat(rows, ignore_index=True)
    out["w"] = out.w / out.w.mean()
    out.to_parquet(a.out)
    print(f"wrote {a.out}: {len(out)} S1, weight range {out.w.min():.2f}-{out.w.max():.2f}")


if __name__ == "__main__":
    main()
