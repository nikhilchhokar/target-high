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

from . import io, pipeline, prep, model, decide, run_experiment, stage2


def self_check(match_map, cand_map, s1_ids):
    """Mirror of the official rules so a bad file never costs a submission."""
    issues = []
    s1_set = set(s1_ids)
    for s, c in cand_map.items():
        m = match_map.get(s, [])
        if len(m) != len(set(m)) or len(c) != len(set(c)):
            issues.append(f"duplicate ids for {s}")
        if any(not x.startswith(("S2-", "S3-")) for x in c):
            issues.append(f"non S2/S3 id for {s}")
        if not set(m) <= set(c):
            issues.append(f"match not in candidates for {s}")
    extra = (set(match_map) | set(cand_map)) - s1_set
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

    if args.empty:
        prep.prep_split(cfg, "test")
        s1_ids = prep.load(cfg, "test", [1], ["entity_id"]).entity_id.tolist()
        sub_id, res, match_map, cand_map = "probe_empty", {"exp_id": "empty"}, {}, {}
    else:
        # reuse the train experiment's thresholds / iteration count if it ran with this exact config
        saved = Path(paths.get("results_dir", "results")) / cfg["exp_id"] / "metrics.json"
        m = json.loads(saved.read_text()) if saved.exists() else {}
        if m.get("cfg_hash") == pipeline.stable_hash(cfg):
            print(f"[reuse] thresholds and iterations from {saved}")
            res = {**m, "fcols": m["features"], "iters": [m["best_iter_median"]],
                   "iters2": [m["best_iter_median_s2"]] if m.get("best_iter_median_s2") else [],
                   "s2cols": m.get("features_s2", [])}
        else:
            res = run_experiment.run(cfg, write=True)
        tr = pipeline.build_train(cfg)
        feats_tr, s1_tr = tr["feats"], tr["s1"]
        truth = io.load_ground_truth(Path(paths["data_dir"]) / "train" / "train_ground_truth.tsv",
                                     s1_tr.entity_id.tolist())
        y = np.fromiter((c in truth[s] for s, c in zip(feats_tr.s1, feats_tr.cand)), bool, len(feats_tr)).astype(int)
        n_iter = int(np.median(res["iters"]) * cfg["model"].get("full_iter_mult", 1.1))
        models = model.train_full(feats_tr[res["fcols"]], y, cfg["model"], n_iter)
        models2 = None
        if cfg["model"].get("stage2") and res.get("iters2"):
            # stage 2 is trained on OUT-OF-FOLD stage-1 probabilities (as in validation)
            oof = pd.read_parquet(Path(paths.get("results_dir", "results")) / cfg["exp_id"] / "oof.parquet")
            assert (oof.s1.values == feats_tr.s1.values).all()
            F2 = stage2.features(feats_tr, oof.p1.values)
            n2 = int(np.median(res["iters2"]) * cfg["model"].get("full_iter_mult", 1.1))
            models2 = model.train_full(F2[res["s2cols"]], y, {**cfg["model"], **cfg["model"]["stage2"]}, n2)
            del F2, oof
        del feats_tr, tr
        scored = []
        for bc, qq, f in pipeline.iter_test(cfg):
            if f is None or len(f) == 0:
                continue
            p = model.predict(models, f[res["fcols"]]).astype(np.float32)
            if models2 is not None:
                p = model.predict(models2, stage2.features(f, p)[res["s2cols"]]).astype(np.float32)
            scored.append(f[["s1", "cand", "cand_src"]].assign(p=p))
        scored = pd.concat(scored, ignore_index=True)
        cache = Path(paths.get("cache_dir", "cache"))
        cache.mkdir(exist_ok=True)
        scored.to_parquet(cache / f"test_scores_{res['exp_id']}.parquet")  # re-decide later without recompute
        d = decide.prepare(scored, cfg["decision"].get("one_to_one", True))
        # ---- per-country threshold tuning on OOF (e009+)
        # If the oof.parquet has country info, find the best per-country thresholds and
        # fall back to the global threshold for countries without enough samples.
        country_thr = {}
        try:
            oof = pd.read_parquet(Path(paths.get("results_dir", "results")) / cfg["exp_id"] / "oof.parquet")
            truth_oof = io.load_ground_truth(Path(paths["data_dir"]) / "train" / "train_ground_truth.tsv",
                                              oof.s1.unique().tolist())
            s1_train = prep.load(cfg, "train", [1], ["entity_id", "bc"])
            bc_map = dict(zip(s1_train.entity_id, s1_train.bc))
            oof_country = oof.s1.map(bc_map).fillna("UNK").values
            d_oof = decide.prepare(oof.assign(cand_src=oof.cand_src), cfg["decision"].get("one_to_one", True))
            scorer_oof = decide.Scorer(d_oof, truth_oof)
            grid = cfg["decision"]["grid"]
            for c in sorted(set(oof_country)):
                msk = oof_country == c
                if msk.sum() < 200:
                    continue
                dc = d_oof[msk].copy().reset_index(drop=True)
                sc = decide.Scorer(dc, {k: v for k, v in truth_oof.items()
                                        if bc_map.get(k, "UNK") == c})
                best_c, _ = decide.grid_search(dc, sc, grid)
                country_thr[c] = (best_c["t_first"], best_c["t_other"], best_c["t_extra"])
                print(f"[per-country] {c}: t_first={country_thr[c][0]} t_other={country_thr[c][1]} "
                      f"t_extra={country_thr[c][2]} (pairs={msk.sum()})", flush=True)
        except Exception as e:
            print(f"[per-country] tuning failed, falling back to global thresholds: {e}")

        # Apply per-country thresholds on test
        s1_test = prep.load(cfg, "test", [1], ["entity_id", "bc"])
        test_bc_map = dict(zip(s1_test.entity_id, s1_test.bc))
        d_country = d.s1.map(test_bc_map).fillna("UNK").values
        mask = np.zeros(len(d), bool)
        for c in np.unique(d_country):
            m = d_country == c
            t1, t2, t3 = country_thr.get(c, (res["t_first"], res["t_other"], res["t_extra"]))
            mask[m] = decide.select(d[m].reset_index(drop=True), t1, t2, t3)
        match_map = decide.to_map(d, mask)
        cand_map = d.groupby("s1", sort=False).cand.apply(list).to_dict()
        sub_id = res["exp_id"]
        s1_ids = prep.load(cfg, "test", [1], ["entity_id"]).entity_id.tolist()

    issues = self_check(match_map, cand_map, s1_ids)
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
    bc = prep.load(cfg, "test", [1], ["entity_id", "bc"]).set_index("entity_id").bc.reindex(s1_ids).values
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
