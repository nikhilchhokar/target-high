"""Hour-1 data audit: the facts that decide blocking, the decision layer and France handling.

    python -m ber.audit configs/base.yaml
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import io, normalize, pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = pipeline.load_config(ap.parse_args().config)
    data_dir = Path(cfg["paths"]["data_dir"])
    rep = {}
    for split in ("train", "test"):
        recs = normalize.normalize_records(io.load_split(data_dir, split))
        r = {"rows_by_src": recs.src.value_counts().sort_index().to_dict(),
             "country_labels_raw": recs.country.value_counts().to_dict(),
             "block_country_by_src": {f"src{s}": g.bc.value_counts().to_dict() for s, g in recs.groupby("src")},
             "empty_name_rate": float((recs.business_name == "").mean()),
             "empty_addr_rate": float((recs.business_address == "").mean()),
             "postal_found_rate_by_country": recs.groupby("bc").postal.apply(lambda s: float((s != "").mean())).to_dict(),
             "landmark_rate_by_country": recs.groupby("bc").landmark.apply(lambda s: float((s != "").mean())).to_dict(),
             "non_ascii_name_rate": float(recs.business_name.map(lambda x: any(ord(c) > 127 for c in x)).mean()),
             "pool_per_s1": float((recs.src != 1).sum() / max((recs.src == 1).sum(), 1))}
        s1 = recs[recs.src == 1]
        r["s1_dup_core_name_rate"] = float(s1.duplicated(["bc", "name_core"], keep=False).mean())
        r["top_s1_core_names"] = s1.name_core.value_counts().head(15).to_dict()
        if split == "train":
            truth = io.load_ground_truth(data_dir / "train" / "train_ground_truth.tsv", s1.entity_id.tolist())
            sizes = np.array([len(v) for v in truth.values()])
            r["singleton_rate"] = float((sizes == 0).mean())
            r["n_truth_distribution"] = {int(k): int(v) for k, v in Counter(sizes).items()}
            shape = Counter((sum(x.startswith("S2-") for x in v), sum(x.startswith("S3-") for x in v))
                            for v in truth.values())
            r["match_shape_(nS2,nS3)"] = {str(k): v for k, v in shape.most_common(12)}
            owner = Counter(x for v in truth.values() for x in v)
            r["pool_ids_in_multiple_gt_lists"] = int(sum(c > 1 for c in owner.values()))
            r["one_to_one_holds"] = r["pool_ids_in_multiple_gt_lists"] == 0
            bc = dict(zip(recs.entity_id, recs.bc))
            same = [bc.get(s) == bc.get(c) for s, v in truth.items() for c in v]
            r["gt_pairs_same_country_rate"] = float(np.mean(same)) if same else None
            post = dict(zip(recs.entity_id, recs.postal))
            both = [(post[s], post[c]) for s, v in truth.items() for c in v if post[s] and post[c]]
            r["gt_pairs_postal_equal_rate_when_both"] = float(np.mean([a == b for a, b in both])) if both else None
            r["unmatched_pool_rate"] = float(1 - len(owner) / max((recs.src != 1).sum(), 1))
        rep[split] = r
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "audit.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n[saved] reports/audit.json")


if __name__ == "__main__":
    main()
