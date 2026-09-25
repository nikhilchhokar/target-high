"""Paired comparison of two experiments on the same S1 entities.

    python -m ber.compare e004 e003
A change is only "real" if the paired delta is clearly larger than ~2 standard errors.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .metrics import paired_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("old")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = Path(a.results)
    sa = pd.read_csv(R / a.new / "entity_scores.csv", index_col=0).f05
    sb = pd.read_csv(R / a.old / "entity_scores.csv", index_col=0).f05
    delta, se = paired_delta(sa, sb)
    print(f"{a.new} - {a.old}: delta={delta:+.4f}  SE={se:.4f}  z={delta / max(se, 1e-9):+.1f}")
    ma = json.loads((R / a.new / "metrics.json").read_text())
    mb = json.loads((R / a.old / "metrics.json").read_text())
    keys = [k for k in ma if isinstance(ma[k], (int, float)) and not isinstance(ma[k], bool) and k in mb]
    rows = [(k, mb[k], ma[k], ma[k] - mb[k]) for k in keys if ma[k] != mb[k]]
    print(pd.DataFrame(rows, columns=["metric", a.old, a.new, "diff"]).to_string(index=False))
    ea = pd.read_csv(R / a.new / "error_buckets.csv", index_col=0)
    eb = pd.read_csv(R / a.old / "error_buckets.csv", index_col=0)
    print("\nerror-bucket loss (old -> new):")
    print(pd.concat([eb.loss.rename(a.old), ea.loss.rename(a.new)], axis=1).fillna(0).round(1).to_string())


if __name__ == "__main__":
    main()
