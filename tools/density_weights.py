"""Per-S1 training weights that match the TEST set's ambiguity profile (decoy density).

    python tools/density_weights.py --oof results/e011/oof.parquet --s1 <train cache>/s1.parquet
        --test-chunks <cache>/test_chunks_e011_<hash> --out <cache>/density_w.parquet

The test pool has ~1.1 more unmatched records per business than train, so test businesses more
often have several borderline candidates. Bucket = number of candidates with 0.1 < p < 0.9
(capped at 4); weight = P_test(bucket | country) / P_train(bucket | country), clipped to [0.5, 4].
"""
import argparse
import glob

import numpy as np
import pandas as pd


def bucket(df):
    return ((df.p > 0.1) & (df.p < 0.9)).groupby(df.s1).sum().clip(upper=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--s1", required=True)
    ap.add_argument("--test-chunks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    oof = pd.read_parquet(a.oof, columns=["s1", "p"])
    s1 = pd.read_parquet(a.s1)
    ctry = dict(zip(s1.entity_id, s1.bc))
    bo = bucket(oof)
    rows = []
    for c in ("in", "us"):
        te = pd.concat([pd.read_parquet(f, columns=["s1", "p"]) for f in glob.glob(f"{a.test_chunks}/{c}_*.parquet")])
        pt = bucket(te).value_counts(normalize=True)
        ids = [s for s in s1.entity_id if ctry[s] == c]
        b = bo.reindex(ids).fillna(0)
        po = b.value_counts(normalize=True)
        w = np.clip([pt.get(x, 0) / po.get(x, 1e-9) for x in b.values], 0.5, 4.0)
        rows.append(pd.DataFrame({"s1": ids, "w": w}))
        print(c, "train", po.round(3).to_dict(), "\n   test", pt.round(3).to_dict())
    out = pd.concat(rows, ignore_index=True)
    out["w"] = out.w / out.w.mean()
    out.to_parquet(a.out)
    print(f"wrote {a.out}: {len(out)} S1, weight range {out.w.min():.2f}-{out.w.max():.2f}")


if __name__ == "__main__":
    main()
