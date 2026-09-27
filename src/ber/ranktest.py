"""Does per-country rank normalisation of features improve transfer to an unseen country?

    python -m ber.ranktest <cached feats.parquet> <results/<exp>/metrics.json> <s1.parquet>

Trains on US, scores India (France proxy) on a 40% S1 subsample, three variants:
none / scale-dependent features ranked within country / all features ranked within country.
"""
import json
import sys

import numpy as np
import pandas as pd

from . import decide, io, model, pipeline

SCALE_PREFIX = ("chain_", "sim_", "prune_score", "n_max_idf", "n_idf_overlap", "rev_n_s1", "ctx_n_",
                "ctx_gap_sim", "ctx_gap_prune", "rev_gap_prune", "rev_gap_sim", "hop_best")


def rank_within(F: pd.DataFrame, country: np.ndarray, cols) -> pd.DataFrame:
    F = F.copy()
    for c in cols:
        F[c] = F[c].groupby(country).rank(pct=True).astype(np.float32)
    return F


def main():
    feats_path, metrics_path, s1_path = sys.argv[1:4]
    m = json.load(open(metrics_path))
    fcols = m["features"]
    s1 = pd.read_parquet(s1_path)
    keep = set(s1.entity_id.sample(frac=0.4, random_state=0))
    f = pd.read_parquet(feats_path, columns=["s1", "cand", "cand_src"] + [c for c in fcols if c != "cand_src"])
    f = f[f.s1.isin(keep)].reset_index(drop=True)
    cfg = pipeline.load_config("configs/base.yaml")
    truth = io.load_ground_truth(cfg["paths"]["data_dir"] + "/train/train_ground_truth.tsv", sorted(keep))
    country = f.s1.map(dict(zip(s1.entity_id, s1.bc))).values
    y = np.fromiter((c in truth[s] for s, c in zip(f.s1, f.cand)), bool, len(f)).astype(int)
    tr, te = country == "us", country == "in"
    truth_in = {k: v for k, v in truth.items() if k in set(f.s1[te])}
    grid = {"t_first": [0.3, 0.8, 0.1], "t_other": [0.5, 0.9, 0.1], "t_extra": [0.6, 0.95, 0.05]}
    mcfg = {"params": {"learning_rate": 0.1, "num_leaves": 127, "min_child_samples": 100, "max_bin": 127,
                       "feature_fraction": 0.8, "bagging_fraction": 0.7, "bagging_freq": 1, "n_jobs": 6},
            "seeds": [42]}
    scale_cols = [c for c in fcols if c.startswith(SCALE_PREFIX)]
    for name, cols in (("none", []), ("scale-dependent", scale_cols), ("all", fcols)):
        X = rank_within(f[fcols], country, cols) if cols else f[fcols]
        ms = model.train_full(X[tr], y[tr], mcfg, 500)
        d = decide.prepare(f.loc[te, ["s1", "cand", "cand_src"]].assign(p=model.predict(ms, X[te])), True)
        sc = decide.Scorer(d, truth_in)
        best = decide.grid_search(d, sc, grid)[0]
        print(f"[rank] {name:16s} ({len(cols):3d} cols ranked): US->IN {best['score_at_choice']:.4f}", flush=True)


if __name__ == "__main__":
    main()
