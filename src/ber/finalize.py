"""Build the submission files from saved test chunk scores (no models / training data in memory).

    python -m ber.finalize configs/exp/e011_hop.yaml

Loads cache/test_chunks_<exp>_<cfghash>/*.parquet (written by make_submission), applies the
experiment's one-to-one resolution and thresholds (results/<exp>/metrics.json), writes both TSVs
to submissions/sub_<time>_<exp>_final/, runs the official validator and appends to the registry.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import decide, io, pipeline, prep


def _lists(s1: np.ndarray, cand: np.ndarray) -> dict:
    """Fast {s1: [cand, ...]} for rows already sorted by s1 (keeps the given order)."""
    if len(s1) == 0:
        return {}
    cut = np.flatnonzero(s1[1:] != s1[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(s1)]
    return {s1[a]: list(cand[a:b]) for a, b in zip(starts, ends)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    a = ap.parse_args()
    t0 = time.time()
    cfg = pipeline.load_config(a.config)
    paths = cfg["paths"]
    ck = Path(paths.get("cache_dir", "cache")) / f"test_chunks_{cfg['exp_id']}_{pipeline.stable_hash(cfg)}"
    files = sorted(ck.glob("*.parquet"))
    scored = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"[finalize] {len(files)} chunks, {len(scored)} scored pairs ({time.time() - t0:.0f}s)", flush=True)
    m = json.loads((Path(paths.get("results_dir", "results")) / cfg["exp_id"] / "metrics.json").read_text())
    d = decide.prepare(scored, cfg["decision"].get("one_to_one", True))
    del scored
    mask = decide.select(d, m["t_first"], m["t_other"], m["t_extra"])
    s1v, cv = d.s1.values, d.cand.values
    match_map = _lists(s1v[mask], cv[mask])
    cand_map = _lists(s1v, cv)
    s1_ids = prep.load(cfg, "test", [1], ["entity_id"]).entity_id.tolist()
    out = Path(paths.get("submissions_dir", "submissions")) / f"sub_{time.strftime('%Y%m%d_%H%M')}_{cfg['exp_id']}_final"
    io.write_id_lists(match_map, s1_ids, out / "matching_results.tsv", "matched_entity_ids")
    io.write_id_lists(cand_map, s1_ids, out / "candidate_pairs.tsv", "candidate_entity_ids")
    print(f"[finalize] wrote {out} ({time.time() - t0:.0f}s)", flush=True)
    status = "validator_not_found"
    validator = Path(paths.get("validator", ""))
    if validator.is_file():
        r = subprocess.run([sys.executable, str(validator), "--matching", str(out / "matching_results.tsv"),
                            "--candidate", str(out / "candidate_pairs.tsv"),
                            "--test-dir", str(Path(paths["data_dir"]) / "test")], capture_output=True, text=True)
        print(r.stdout[-600:])
        status = "PASS" if r.returncode == 0 else "FAIL"
    n_pred = np.array([len(match_map.get(s, ())) for s in s1_ids])
    row = {"sub_dir": out.name, "exp_id": cfg["exp_id"], "validator": status, "cv_macroF05": m.get("macroF05"),
           "t_first": m["t_first"], "t_other": m["t_other"], "t_extra": m["t_extra"],
           "test_cand_mean": float(np.mean([len(cand_map.get(s, ())) for s in s1_ids])),
           "test_pred_matches": int(n_pred.sum()), "test_pred_singleton_rate": float((n_pred == 0).mean()),
           "note": "finalize from saved chunks", "uploaded": "", "lb_score": ""}
    (out / "submission_info.json").write_text(json.dumps(row, indent=2))
    reg = Path(paths.get("submissions_dir", "submissions")) / "registry.csv"
    pd.concat([pd.read_csv(reg), pd.DataFrame([row])] if reg.exists() else [pd.DataFrame([row])],
              ignore_index=True).to_csv(reg, index=False)
    print(json.dumps(row, indent=2))
    print(f"[finalize] done ({time.time() - t0:.0f}s). Validator: {status}")


if __name__ == "__main__":
    main()
