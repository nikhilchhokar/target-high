"""Unseen-country diagnostics (France is test-only): leave-one-country-out + self-training.

    python -m ber.loco configs/exp/e008_india.yaml

For each training country C: train stage 1 on the other country only, score C and report
macro F0.5 with (a) the global thresholds, (b) C's own best thresholds (calibration loss), and
(c) after self-training: add C's confident predictions (p>hi -> 1, p<lo -> 0) as pseudo-labels,
retrain, re-score. (c) is what we can do for France using its unlabeled test records.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import decide, io, model, pipeline


def _score(df, p, truth, grid, thr):
    d = decide.prepare(df.assign(p=p), True)
    sc = decide.Scorer(d, truth)
    fixed = sc(decide.select(d, *thr))
    best = decide.grid_search(d, sc, grid)[0]
    return fixed, best["score_at_choice"], (best["t_first"], best["t_other"], best["t_extra"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--hi", type=float, default=0.98)
    ap.add_argument("--lo", type=float, default=0.02)
    a = ap.parse_args()
    cfg = pipeline.load_config(a.config)
    built = pipeline.build_train(cfg)
    feats, s1 = built["feats"], built["s1"]
    m = json.loads((Path(cfg["paths"].get("results_dir", "results")) / cfg["exp_id"] / "metrics.json").read_text())
    thr = (m["t_first"], m["t_other"], m["t_extra"])
    fcols = m["features"]
    truth = io.load_ground_truth(Path(cfg["paths"]["data_dir"]) / "train" / "train_ground_truth.tsv",
                                 s1.entity_id.tolist())
    country = dict(zip(s1.entity_id, s1.bc))
    fc = feats.s1.map(country).values
    y = np.fromiter((c in truth[s] for s, c in zip(feats.s1, feats.cand)), bool, len(feats)).astype(int)
    grid = {"t_first": [0.3, 0.8, 0.05], "t_other": [0.5, 0.95, 0.05], "t_extra": [0.6, 0.95, 0.05]}
    n_iter = int(m["best_iter_median"])
    mcfg = {**cfg["model"], "seeds": [42]}
    for c in sorted(set(country.values())):
        tr, te = fc != c, fc == c
        truth_c = {k: v for k, v in truth.items() if country[k] == c}
        base = feats.loc[te, ["s1", "cand", "cand_src"]]
        ms = model.train_full(feats.loc[tr, fcols], y[tr], mcfg, n_iter)
        p = model.predict(ms, feats.loc[te, fcols])
        f_fix, f_best, t_best = _score(base, p, truth_c, grid, thr)
        print(f"[loco] {c}: transfer fixed-thr {f_fix:.4f} | own best thr {f_best:.4f} at {t_best}", flush=True)
        conf = (p >= a.hi) | (p <= a.lo)
        Xp = pd.concat([feats.loc[tr, fcols], feats.loc[te, fcols][conf]], ignore_index=True)
        yp = np.r_[y[tr], (p[conf] >= a.hi).astype(int)]
        ms2 = model.train_full(Xp, yp, mcfg, n_iter)
        p2 = model.predict(ms2, feats.loc[te, fcols])
        g_fix, g_best, g_t = _score(base, p2, truth_c, grid, thr)
        print(f"[loco] {c}: self-trained fixed-thr {g_fix:.4f} | own best thr {g_best:.4f} at {g_t} "
              f"(pseudo-labelled {conf.mean():.3f} of pairs, {int((p[conf] >= a.hi).sum())} positives)", flush=True)
        del ms, ms2, Xp


if __name__ == "__main__":
    main()
