"""Produce output/matching_results.tsv + output/candidate_pairs.tsv for the TEST split.

    python -m ber.make_submission configs/exp/e003.yaml            # full pipeline
    python -m ber.make_submission configs/base.yaml --empty        # all-empty probe

Steps: run the experiment on train (thresholds + iteration count from OOF), fit final
models on all training pairs, build test candidates/features, predict, decide, write,
self-check, run the official validator if present, and register the submission.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import io, pipeline, model, decide, run_experiment


def self_check(match_map, cand_map, s1_ids, pool_ids):
    """Mirror of the official rules so a bad file never costs a submission."""
    issues = []
    pool = set(pool_ids)
    for s in s1_ids:
        m, c = match_map.get(s, []), cand_map.get(s, [])
        if len(m) != len(set(m)) or len(c) != len(set(c)):
            issues.append(f"duplicate ids for {s}")
        if not set(m) <= pool or not set(c) <= pool:
            issues.append(f"id outside test S2/S3 for {s}")
        if not set(m) <= set(c):
            issues.append(f"match not in candidates for {s}")
    extra = set(match_map) - set(s1_ids)
    if extra:
        issues.append(f"{len(extra)} unknown S1 ids")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--empty", action="store_true", help="all-empty probe (measures public singleton rate)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    cfg = pipeline.load_config(args.config)
    paths = cfg["paths"]
    t0 = time.time()

    test = pipeline.build_split(cfg, "test")
    recs, feats = test["recs"], test["feats"]
    s1_ids = recs.entity_id[recs.src == 1].tolist()
    pool_ids = recs.entity_id[recs.src != 1].tolist()
    cand_map = feats.groupby("s1", sort=False).cand.apply(list).to_dict()

    if args.empty:
        sub_id, res, match_map = "probe_empty", {"exp_id": "empty"}, {}
    else:
        res = run_experiment.run(cfg, write=True)
        tr = pipeline.build_split(cfg, "train", verbose=False)["feats"]
        truth = io.load_ground_truth(Path(paths["data_dir"]) / "train" / "train_ground_truth.tsv",
                                     tr.s1.unique().tolist())
        y = np.fromiter((c in truth.get(s, ()) for s, c in zip(tr.s1, tr.cand)), bool, len(tr)).astype(int)
        n_iter = int(np.median(res["iters"]) * cfg["model"].get("full_iter_mult", 1.1))
        models = model.train_full(tr[res["fcols"]], y, cfg["model"], n_iter)
        feats = feats.assign(p=model.predict(models, feats[res["fcols"]]))
        d = decide.prepare(feats, cfg["decision"].get("one_to_one", True))
        mask = decide.select(d, res["t_first"], res["t_other"], res["t_extra"])
        match_map = decide.to_map(d, mask)
        # keep candidate order = model score order for readability
        cand_map = d.groupby("s1", sort=False).cand.apply(list).to_dict()
        sub_id = res["exp_id"]

    issues = self_check(match_map, cand_map, s1_ids, pool_ids)
    if issues:
        print("[FAIL] self-check:\n  " + "\n  ".join(issues[:20]))
        sys.exit(1)

    stamp = time.strftime("%Y%m%d_%H%M")
    out = Path(paths.get("submissions_dir", "submissions")) / f"sub_{stamp}_{sub_id}"
    out.mkdir(parents=True, exist_ok=True)
    io.write_id_lists(match_map, s1_ids, out / "matching_results.tsv", "matched_entity_ids")
    io.write_id_lists(cand_map, s1_ids, out / "candidate_pairs.tsv", "candidate_entity_ids")
    shutil.copy(Path(args.config), out / "config.yaml")

    validator = Path(paths.get("validator", ""))
    status = "validator_not_found"
    if validator.is_file():
        r = subprocess.run([sys.executable, str(validator), "--matching", str(out / "matching_results.tsv"),
                            "--candidate", str(out / "candidate_pairs.tsv"),
                            "--test-dir", str(Path(paths["data_dir"]) / "test")],
                           capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-2000:])
        status = "PASS" if r.returncode == 0 else "FAIL"

    # latest passing submission is mirrored to output/ (the zip layout)
    if status != "FAIL":
        Path("output").mkdir(exist_ok=True)
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            shutil.copy(out / f, Path("output") / f)

    sizes = np.array([len(cand_map.get(s, ())) for s in s1_ids])
    n_pred = np.array([len(match_map.get(s, ())) for s in s1_ids])
    bc = recs[recs.src == 1].bc.values
    pred_single = {f"pred_singleton_rate_{c}": float((n_pred[bc == c] == 0).mean()) for c in np.unique(bc)}
    sha, dirty = run_experiment.git_info()
    row = {"sub_dir": out.name, "exp_id": res.get("exp_id"), "git_sha": sha, "dirty": dirty,
           "validator": status, "cv_macroF05": res.get("macroF05"), "t_first": res.get("t_first"),
           "t_other": res.get("t_other"), "t_extra": res.get("t_extra"),
           "test_cand_mean": float(sizes.mean()), "test_cand_p95": float(np.percentile(sizes, 95)),
           "test_pred_matches": int(n_pred.sum()), "test_pred_singleton_rate": float((n_pred == 0).mean()),
           **pred_single, "note": args.note, "uploaded": "", "lb_score": ""}
    (out / "submission_info.json").write_text(json.dumps(row, indent=2, default=str))
    reg = Path(paths.get("submissions_dir", "submissions")) / "registry.csv"
    pd.concat([pd.read_csv(reg), pd.DataFrame([row])] if reg.exists() else [pd.DataFrame([row])],
              ignore_index=True).to_csv(reg, index=False)
    print(json.dumps(row, indent=2, default=str))
    print(f"[done] {out}  ({time.time() - t0:.0f}s). Validator: {status}")
    if status == "validator_not_found":
        print("[warn] official utils/validate_submission.py not found - set paths.validator before uploading")


if __name__ == "__main__":
    main()
