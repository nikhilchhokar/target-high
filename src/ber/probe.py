"""Build a leaderboard probe: a copy of a submission with one country's predictions emptied.

    python -m ber.probe <submission dir> fr

France's score = (LB(full) - LB(probe)) / share_fr + singleton_rate_fr  (share ~0.15).
"""
import csv
import shutil
import sys
from pathlib import Path

import pandas as pd

from . import pipeline, prep


def main():
    src, country = Path(sys.argv[1]), sys.argv[2]
    cfg = pipeline.load_config("configs/base.yaml")
    ids = set(prep.load(cfg, "test", [1], ["entity_id", "bc"]).query("bc == @country").entity_id)
    m = pd.read_csv(src / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    blank = m.source1_entity_id.isin(ids)
    m.loc[blank, "matched_entity_ids"] = ""
    out = Path("submissions") / f"probe_{src.name}_{country}_empty"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for a, b in zip(m.source1_entity_id, m.matched_entity_ids):
            f.write(f"{a}\t{b}\n")
    shutil.copy(src / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
    print(f"{out}: blanked {int(blank.sum())} of {len(m)} rows ({blank.mean():.3f})")


if __name__ == "__main__":
    main()
