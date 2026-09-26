"""Run one experiment on the TRAIN split and log everything.

    python -m ber.run_experiment configs/exp/e001_baseline.yaml

Writes results/<exp_id>/ (config, metrics, OOF scores, per-entity scores, threshold
curve, feature importance, error buckets) and appends a row to results/leaderboard.csv.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import io, pipeline, model, decide, metrics, errors


def git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "configs"],
                                             text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:
        sha, dirty = "nogit", True
    return sha, dirty


def _thr(best):
    return best["t_first"], best["t_other"], best["t_extra"]


def run(cfg: dict, write: bool = True) -> dict:
    t_start = time.time()
    exp_id = cfg["exp_id"]
    built = pipeline.build_train(cfg)
    union, feats, s1 = built["union"], built["feats"], built["s1"]
    truth = io.load_ground_truth(Path(cfg["paths"]["data_dir"]) / "train" / "train_ground_truth.tsv",
                                 s1.entity_id.tolist())
    s1_country = dict(zip(s1.entity_id, s1.bc))
    cand_map = feats.groupby("s1").cand.apply(list).to_dict()

    # ---- blocking (measured independently of the model)
    bm = metrics.blocking_metrics(cand_map, truth, built["n_pool"], union)

    # ---- model (out-of-fold, grouped by S1)
    t0 = time.time()
    y = np.fromiter((c in truth[s] for s, c in zip(feats.s1, feats.cand)), bool, len(feats)).astype(int)
    fcols = model.feature_cols(feats, cfg["model"].get("drop_features", []))
    oof, iters, imp = model.train_oof(feats[fcols], y, feats.s1.values, cfg["model"])
    feats = feats.assign(p=oof, y=y)
    t_train = time.time() - t0
    rel = metrics.reliability(oof, y)

    # ---- decision layer tuned on the exact macro metric
    dcfg = cfg["decision"]
    one_to_one = dcfg.get("one_to_one", True)
    d = decide.prepare(feats, one_to_one)
    scorer = decide.Scorer(d, truth, dcfg.get("target_singleton_rate"))
    print(f"[decide] grid search over thresholds ({len(d)} pairs)", flush=True)
    best, curve = decide.grid_search(d, scorer, dcfg["grid"])
    mask = decide.select(d, *_thr(best))
    pred_map = decide.to_map(d, mask)
    bd = metrics.breakdown(pred_map, truth, s1_country)
    ent = metrics.entity_scores(pred_map, truth)

    # ---- leave-one-country-out: proxy for how we'll do on unseen France
    loco = {}
    countries = sorted(set(s1_country.values()))
    if cfg.get("loco", True) and len(countries) > 1:
        n_iter = int(np.median(iters))
        fc = feats.s1.map(s1_country).values
        for c in countries:
            tr, te = fc != c, fc == c
            ms = model.train_full(feats.loc[tr, fcols], y[tr], {**cfg["model"], "seeds": [42]}, n_iter)
            dc = decide.prepare(feats.loc[te].assign(p=model.predict(ms, feats.loc[te, fcols])), one_to_one)
            truth_c = {k: v for k, v in truth.items() if s1_country[k] == c}
            sc = decide.Scorer(dc, truth_c)
            loco[f"F05_loco_{c}"] = sc(decide.select(dc, *_thr(best)))
            loco[f"F05_loco_{c}_best_thr"] = decide.grid_search(dc, sc, dcfg["grid"])[0]["raw_best"]["score"]

    # ---- errors
    eb = errors.entity_buckets(d, mask, truth, cand_map)
    bsum = errors.bucket_summary(eb)

    sha, dirty = git_info()
    res = {
        "exp_id": exp_id, "parent": cfg.get("parent"), "note": cfg.get("note", ""),
        "git_sha": sha, "dirty": dirty, "cfg_hash": pipeline.stable_hash(cfg),
        "t_first": best["t_first"], "t_other": best["t_other"], "t_extra": best["t_extra"],
        "raw_best": best["raw_best"], **bm, **bd, **loco,
        "ece": rel["ece"], "n_pairs": int(len(feats)), "pos_rate": float(y.mean()),
        "best_iter_median": int(np.median(iters)), "n_features": len(fcols),
        "n_s1_eval": len(s1), "runtime_train_s": t_train,
        **{f"runtime_{k}": v for k, v in built["timings"].items()},
        "runtime_total_s": time.time() - t_start,
    }
    _print_summary(res, bsum)
    if write:
        out = Path(cfg["paths"].get("results_dir", "results")) / exp_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (out / "metrics.json").write_text(json.dumps({**res, "features": fcols,
                                                      "reliability": rel["bins"]}, indent=2))
        feats[["s1", "cand", "cand_src", "p", "y"]].to_parquet(out / "oof.parquet")
        ent.rename("f05").to_csv(out / "entity_scores.csv", index_label="s1")
        curve.to_csv(out / "threshold_curve.csv", index=False)
        imp.rename("gain_share").to_csv(out / "feature_importance.csv", index_label="feature")
        eb.to_csv(out / "errors_entities.csv", index=False)
        bsum.to_csv(out / "error_buckets.csv")
        errors.pair_dump(d, mask, truth, lambda ids: pipeline.load_raw(cfg, "train", ids),
                         feats).to_csv(out / "errors_pairs.csv", index=False)
        row = {k: v for k, v in res.items() if not isinstance(v, (dict, list))}
        lb = Path(cfg["paths"].get("results_dir", "results")) / "leaderboard.csv"
        pd.concat([pd.read_csv(lb), pd.DataFrame([row])] if lb.exists() else [pd.DataFrame([row])],
                  ignore_index=True).to_csv(lb, index=False)
        print(f"[saved] {out}")
    return {**res, "fcols": fcols, "iters": iters}


def run_blocking_only(cfg: dict) -> dict:
    """Blocking recall / ceiling / size on the train sample, without features or a model."""
    from . import prep, retrieve
    t0 = time.time()
    prep.prep_split(cfg, "train")
    s1_all = prep.load(cfg, "train", [1], pipeline.Q_COLS)
    n = cfg.get("sample", {}).get("n_s1")
    sample = set(s1_all.entity_id.sample(n=min(n, len(s1_all)), random_state=cfg["sample"].get("seed", 42))
                 if n else s1_all.entity_id)
    unions, n_pool = [], 0
    only = cfg["blocking"].get("only_countries")
    for bc in sorted(s1_all.bc.unique()):
        if only and bc not in only:
            continue
        pool = prep.load(cfg, "train", [2, 3], pipeline.Q_COLS, bc=bc)
        n_pool += len(pool)
        s1c = s1_all[s1_all.bc == bc]
        q = s1c[s1c.entity_id.isin(sample)].reset_index(drop=True)
        t_i = time.time()
        idx = retrieve.CountryIndex(pool, s1c, cfg["blocking"], cfg.get("threads", 7))
        print(f"[index] {bc}: pool {len(pool)} ({time.time() - t_i:.0f}s), nnz per view "
              + str({v: int(idx.B[v].nnz) for v in idx.B}), flush=True)
        for s in range(0, len(q), cfg["blocking"].get("query_chunk", 200_000)):
            qq = q.iloc[s:s + cfg["blocking"].get("query_chunk", 200_000)].reset_index(drop=True)
            t_q = time.time()
            u, _ = idx.query(qq)
            print(f"   {bc} query {len(qq)}: union {len(u)} ({time.time() - t_q:.0f}s)", flush=True)
            u = u.assign(s1=qq.entity_id.values[u.qi.values], cand=idx.pool.entity_id.values[u.pj.values])
            unions.append(u.drop(columns=["qi", "pj"]))
        del idx, pool
    union = pd.concat(unions, ignore_index=True)
    prune = cfg["blocking"].get("prune", {})
    keep = (union.prune_rank <= prune.get("max_k", 15)) & (union.prune_score >= prune.get("min_score", 0.0))
    s1 = s1_all[s1_all.entity_id.isin(sample) & (s1_all.bc.isin(only) if only else True)]
    truth = io.load_ground_truth(Path(cfg["paths"]["data_dir"]) / "train" / "train_ground_truth.tsv",
                                 s1.entity_id.tolist())
    cand_map = union[keep].groupby("s1").cand.apply(list).to_dict()
    bm = metrics.blocking_metrics(cand_map, truth, n_pool, union)
    union["is_pos"] = [c in truth[s] for s, c in zip(union.s1, union.cand)]
    for r in cfg["blocking"]["retrievers"]:
        n_ = r["name"]
        found = union[f"rank_{n_}"] <= r["k"]
        others = np.zeros(len(union), bool)
        for r2 in cfg["blocking"]["retrievers"]:
            if r2["name"] != n_:
                others |= (union[f"rank_{r2['name']}"] <= r2["k"]).values
        bm[f"unique_pos_{n_}"] = int((union.is_pos & found & ~others).sum())
    bm["seconds"] = time.time() - t0
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in bm.items()}, indent=1))
    out = Path(cfg["paths"].get("results_dir", "results")) / cfg["exp_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "blocking.json").write_text(json.dumps(bm, indent=2, default=float))
    return bm


def _print_summary(r, bsum):
    keys = ["macroF05", "singleton_acc", "matched_F", "F_multi", "pair_P", "pair_R",
            "block_pair_recall", "entity_ceiling", "cand_mean", "cand_p95", "t_first", "t_other",
            "t_extra", "ece"]
    print("\n=== " + r["exp_id"] + " ===")
    print("  ".join(f"{k}={r[k]:.4f}" if isinstance(r.get(k), float) else f"{k}={r.get(k)}" for k in keys))
    extra = {k: v for k, v in r.items() if k.startswith("F05_")}
    print("  ".join(f"{k}={v:.4f}" for k, v in extra.items()))
    print(bsum.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--blocking-only", action="store_true", help="only measure blocking recall/size")
    args = ap.parse_args()
    cfg = pipeline.load_config(args.config)
    run_blocking_only(cfg) if args.blocking_only else run(cfg)


if __name__ == "__main__":
    main()
