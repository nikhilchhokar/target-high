"""Entity-level error buckets + side-by-side pair dumps for human review.

Buckets are ranked by metric loss (sum of 1 - F), so the team always fixes the
most expensive failure mode first.
"""
import pandas as pd

from .metrics import f05


def entity_buckets(d, mask, truth_map, cand_map) -> pd.DataFrame:
    sel = d[mask]
    pred_map = sel.groupby("s1").cand.apply(set).to_dict()
    top = d[d.is_top].set_index("s1")
    rows = []
    for s1, truth in truth_map.items():
        pred, cands = pred_map.get(s1, set()), set(cand_map.get(s1, ()))
        tp, fp = pred & truth, pred - truth
        top_c = top.cand.get(s1, "")
        if not truth:
            b = "ok_singleton" if not pred else "E1_singleton_merged"
        elif not truth & cands:
            b = "E6_all_true_blocked"
        elif not pred:
            b = "E3_abstained_top_correct" if top_c in truth else "E3b_abstained_top_wrong"
        elif not tp:
            b = "E2_wrong_match"
        elif fp:
            b = "E4_extra_wrong"
        elif tp != truth:
            b = "E6b_partly_blocked" if truth - cands else "E5_missed_extra"
        else:
            b = "ok"
        rows.append({"s1": s1, "bucket": b, "f05": f05(pred, truth), "n_truth": len(truth),
                     "n_pred": len(pred), "n_cands": len(cands), "top_cand": top_c,
                     "top_p": float(top.pa.get(s1, 0.0)), "truth": ",".join(sorted(truth)),
                     "pred": ",".join(sorted(pred))})
    return pd.DataFrame(rows)


def bucket_summary(eb: pd.DataFrame) -> pd.DataFrame:
    s = eb.assign(loss=1 - eb.f05).groupby("bucket").agg(n=("s1", "size"), loss=("loss", "sum"))
    s["loss_share"] = s.loss / max(s.loss.sum(), 1e-9)
    return s.sort_values("loss", ascending=False)


def pair_dump(d, mask, truth_map, raw_loader, feats: pd.DataFrame) -> pd.DataFrame:
    """FP pairs, FN pairs that were candidates, and FN pairs lost at blocking - side by side.

    raw_loader(ids) -> DataFrame[entity_id, business_name, business_address, country]
    """
    y = [c in truth_map[s] for s, c in zip(d.s1, d.cand)]
    dd = d.assign(y=y, selected=mask)
    fp = dd[dd.selected & ~dd.y].assign(kind="FP")
    fn = dd[~dd.selected & dd.y].assign(kind="FN_scored")
    cand_pairs = set(zip(d.s1, d.cand))
    blocked = [(s, c) for s, v in truth_map.items() for c in v if (s, c) not in cand_pairs]
    fnb = pd.DataFrame(blocked, columns=["s1", "cand"]).assign(kind="FN_blocked", pa=float("nan"))
    out = pd.concat([fp, fn, fnb], ignore_index=True)[["kind", "s1", "cand", "pa"]]
    info = raw_loader(set(out.s1) | set(out.cand)).drop_duplicates("entity_id").set_index("entity_id")
    key_feats = [c for c in ["sim_name", "sim_addr", "n_tset", "a_tset", "num_near", "postal_state",
                             "chain_s1", "ctx_rank_prune_score"] if c in feats.columns]
    out = out.merge(feats[["s1", "cand"] + key_feats], on=["s1", "cand"], how="left")
    for side, col in (("s1", "s1"), ("c", "cand")):
        j = info.reindex(out[col].values).reset_index(drop=True)
        out[f"{side}_name"], out[f"{side}_addr"] = j.business_name.values, j.business_address.values
    out["country"] = info.reindex(out.s1.values).country.values
    return out.sort_values(["kind", "pa"], ascending=[True, False])
