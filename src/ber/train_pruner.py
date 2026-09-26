"""Train the learned pruner that ranks the retrieval union before cutting to max_k.

    python -m ber.train_pruner configs/exp/e002_pruner.yaml

Uses a separate S1 sample (pruner.seed, excluding the main training sample) so the main
matcher never trains on candidates chosen by a pruner that saw the same entities.
Reports union recall@k for the learned score vs the mean-of-cosines baseline on a
held-out 30% of those entities, then saves the booster to blocking.prune.model.
"""
import argparse
import copy
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import io, pipeline, prep, retrieve


def _recall_at(union, score, truth_n, ks=(5, 10, 15, 20, 30)):
    u = union.assign(sc=score).sort_values(["qi_g", "sc"], ascending=[True, False], kind="stable")
    r = u.groupby("qi_g").cumcount() + 1
    return {k: float(u.y.values[(r <= k).values].sum() / truth_n) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = pipeline.load_config(ap.parse_args().config)
    pcfg = cfg.get("pruner", {"n_s1": 60000, "seed": 7})
    bcfg = copy.deepcopy(cfg["blocking"])
    out_path = Path(bcfg["prune"]["model"])
    bcfg["prune"] = {"score": "mean", "max_k": 10 ** 6, "min_score": 0.0}  # keep the whole union
    t0 = time.time()
    prep.prep_split(cfg, "train")
    s1_all = prep.load(cfg, "train", [1], pipeline.Q_COLS)
    main_ids = pipeline.train_sample_ids(cfg, s1_all)
    rest = s1_all.entity_id[~s1_all.entity_id.isin(main_ids)]
    ids = set(rest.sample(n=min(pcfg["n_s1"], len(rest)), random_state=pcfg["seed"]))
    truth = io.load_ground_truth(Path(cfg["paths"]["data_dir"]) / "train" / "train_ground_truth.tsv", list(ids))
    feats, ys, groups = [], [], []
    offset = 0
    for bc in sorted(s1_all.bc.unique()):
        pool = prep.load(cfg, "train", [2, 3], pipeline.Q_COLS, bc=bc)
        s1c = s1_all[s1_all.bc == bc]
        q = s1c[s1c.entity_id.isin(ids)].reset_index(drop=True)
        idx = retrieve.CountryIndex(pool, s1c, bcfg, cfg.get("threads", 7))
        union, _ = idx.query(q)
        F = retrieve.prune_features(union, list(idx.vecs), bcfg["retrievers"],
                                    idx.pool.src.values[union.pj.values])
        s1_ids, cand_ids = q.entity_id.values[union.qi.values], idx.pool.entity_id.values[union.pj.values]
        feats.append(F)
        ys.append(np.fromiter((c in truth[s] for s, c in zip(s1_ids, cand_ids)), bool, len(union)))
        groups.append(union.qi.values + offset)
        offset += len(q)
        print(f"[pruner] {bc}: {len(q)} S1, union {len(union)} ({time.time() - t0:.0f}s)", flush=True)
        del idx, pool
    n_true = sum(len(v) for v in truth.values())
    F = pd.concat(feats, ignore_index=True)
    y = np.concatenate(ys).astype(int)
    g = np.concatenate(groups)
    rng = np.random.default_rng(0)
    hold = rng.random(g.max() + 1) < 0.3
    va, tr = hold[g], ~hold[g]
    params = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=200,
                  feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1, verbose=-1, n_jobs=8)
    dtr = lgb.Dataset(F[tr].to_numpy(), y[tr], feature_name=list(F.columns))
    dva = lgb.Dataset(F[va].to_numpy(), y[va], reference=dtr)
    booster = lgb.train(params, dtr, 400, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
    held = pd.DataFrame({"qi_g": g[va], "y": y[va]})
    base = _recall_at(held, F[va][[c for c in F.columns if c.startswith("sim_")]].mean(axis=1).values, y[va].sum())
    learned = _recall_at(held, booster.predict(F[va].to_numpy()), y[va].sum())
    union_recall = y.sum() / n_true
    print(f"[pruner] union recall (all retrieved) = {union_recall:.4f}; below: share of union positives kept")
    for k in base:
        print(f"   @{k:>2}: mean-cosine {base[k]:.4f}  learned {learned[k]:.4f}   "
              f"-> pair recall approx {learned[k] * union_recall:.4f}")
    final = lgb.train(params, lgb.Dataset(F.to_numpy(), y, feature_name=list(F.columns)),
                      booster.best_iteration or 200)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(out_path))
    print(f"[pruner] saved {out_path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
