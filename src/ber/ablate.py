"""Which feature groups hurt transfer to an unseen country? (proxy for France)

    python -m ber.ablate configs/exp/e009_legal.yaml [--train us --test in]

Trains a fast LightGBM on one training country with each feature group removed, scores the
other country with its own best thresholds, and prints the transfer score per ablation.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import decide, io, model, pipeline

GROUPS = {
    "chain": lambda c: c.startswith("chain_"),
    "ctx": lambda c: c.startswith("ctx_"),
    "rev": lambda c: c.startswith("rev_"),
    "sib": lambda c: c.startswith("sib"),
    "retrieval": lambda c: c.startswith(("sim_", "rank_")) or c in ("n_retr", "prune_score", "prune_rank"),
    "idf": lambda c: c in ("n_max_idf_unshared", "n_max_idf_shared", "n_idf_overlap"),
    "landmark": lambda c: c.startswith("landmark"),
    "numbers": lambda c: c.startswith(("num_", "unit_")) or c == "b_pmb_box",
    "script": lambda c: c in ("a_name_indic", "b_name_indic"),
    "legal": lambda c: c.startswith("legal") or c == "n_full_tsort",
    "lengths": lambda c: c in ("n_len_min", "n_len_diff", "n_tok_diff", "a_len_min", "a_missing"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--train", default="us")
    ap.add_argument("--test", default="in")
    a = ap.parse_args()
    cfg = pipeline.load_config(a.config)
    built = pipeline.build_train(cfg)
    feats, s1 = built["feats"], built["s1"]
    truth = io.load_ground_truth(Path(cfg["paths"]["data_dir"]) / "train" / "train_ground_truth.tsv",
                                 s1.entity_id.tolist())
    country = dict(zip(s1.entity_id, s1.bc))
    fc = feats.s1.map(country).values
    y = np.fromiter((c in truth[s] for s, c in zip(feats.s1, feats.cand)), bool, len(feats)).astype(int)
    all_cols = model.feature_cols(feats, ["p", "p1"])
    tr, te = fc == a.train, fc == a.test
    truth_t = {k: v for k, v in truth.items() if country[k] == a.test}
    base = feats.loc[te, ["s1", "cand", "cand_src"]]
    grid = {"t_first": [0.3, 0.8, 0.1], "t_other": [0.5, 0.9, 0.1], "t_extra": [0.6, 0.95, 0.05]}
    mcfg = {"params": {"learning_rate": 0.15, "num_leaves": 63, "min_child_samples": 100, "max_bin": 63,
                       "feature_fraction": 0.8, "bagging_fraction": 0.7, "bagging_freq": 1, "n_jobs": 8},
            "seeds": [42]}
    results = {}
    for name in ["none"] + list(GROUPS):
        t0 = time.time()
        cols = [c for c in all_cols if name == "none" or not GROUPS[name](c)]
        ms = model.train_full(feats.loc[tr, cols], y[tr], mcfg, 300)
        d = decide.prepare(base.assign(p=model.predict(ms, feats.loc[te, cols])), True)
        best = decide.grid_search(d, decide.Scorer(d, truth_t), grid)[0]["score_at_choice"]
        results[name] = best
        print(f"[ablate] drop {name:10s} ({len(all_cols) - len(cols):3d} cols): {a.train}->{a.test} "
              f"{best:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/ablate_{a.train}_to_{a.test}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
