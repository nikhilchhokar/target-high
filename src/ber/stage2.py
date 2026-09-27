"""Second-stage (stacked) matcher: re-score each pair using the stage-1 probabilities of its
whole neighbourhood - the other candidates of the same S1 and the other S1 competing for the
same candidate. Stage-1 inputs are out-of-fold on train and full-model predictions on test.
"""
import numpy as np
import pandas as pd

# raw pair features carried into stage 2 alongside the group features
CARRY = ["sim_name_addr", "sim_name", "sim_addr", "n_tset", "a_tset", "num_signed_diff",
         "num_state", "a_missing", "chain_s1", "sib1_name_addr", "sib1_addr", "sib1_name",
         "sib2_name_addr", "cand_src", "prune_rank", "n_nospace_partial", "legal_canon_state",
         "n_full_tsort", "b_num1", "a_num1", "is_hop", "hop_best"]


def features(df: pd.DataFrame, p1: np.ndarray) -> pd.DataFrame:
    """df needs s1, cand (+ CARRY columns when present); p1 aligned with df rows."""
    F = pd.DataFrame(index=df.index)
    p = pd.Series(np.asarray(p1, np.float32), index=df.index)
    F["p1"] = p
    g = p.groupby(df.s1.values)
    mx = g.transform("max")
    F["s_rank"] = g.rank(ascending=False, method="min")
    F["s_gap_top"] = mx - p
    F["s_top"] = mx
    F["s_sum"] = g.transform("sum")
    F["s_n_hi"] = (p > 0.5).groupby(df.s1.values).transform("sum")
    F["s_n"] = g.transform("size")
    sec = p.where(p < mx).groupby(df.s1.values).transform("max").fillna(0)
    F["s_second"] = sec
    F["s_other_best"] = np.where(p >= mx, sec, mx)     # best OTHER candidate of this S1
    h = p.groupby(df.cand.values)
    cmx = h.transform("max")
    csec = p.where(p < cmx).groupby(df.cand.values).transform("max").fillna(0)
    F["c_rank"] = h.rank(ascending=False, method="min")
    F["c_other_best"] = np.where(p >= cmx, csec, cmx)  # best OTHER S1 claiming this candidate
    F["c_margin"] = p - F["c_other_best"]
    F["c_n"] = h.transform("size")
    # source-wise: rank within same S1 and same source
    if "cand_src" in df:
        F["ss_rank"] = p.groupby([df.s1.values, df.cand_src.values]).rank(ascending=False, method="min")
    # source cardinality: nearly every matched S1 has >=1 S3 copy (and usually >=1 S2), so a
    # business with strong S2 copies but no strong S3 copy is probably missing one
    if "cand_src" in df:
        src = df.cand_src.values
        strong5 = (p.values >= 0.5)
        n2 = pd.Series(strong5 & (src == 2)).groupby(df.s1.values).transform("sum").values
        n3 = pd.Series(strong5 & (src == 3)).groupby(df.s1.values).transform("sum").values
        F["s_n_strong_s2"], F["s_n_strong_s3"] = n2, n3
        F["own_src_strong"] = np.where(src == 2, n2, n3)
        F["other_src_strong"] = np.where(src == 2, n3, n2)
        best_in_src = p.groupby([df.s1.values, src]).rank(ascending=False, method="first").values == 1
        F["best_in_empty_src"] = (best_in_src & (F["own_src_strong"].values == 0)).astype(np.float32)
        F["p_best_in_src"] = p.groupby([df.s1.values, src]).transform("max").values
    # name rival: the best name similarity between this candidate and any OTHER S1 retrieving it
    # (multi-tenant buildings: the candidate may carry another S1's name at the same address)
    if "n_tset" in df:
        nt = pd.Series(df.n_tset.values.astype(np.float32), index=df.index)
        hn = nt.groupby(df.cand.values)
        nmx = hn.transform("max")
        nsec = nt.where(nt < nmx).groupby(df.cand.values).transform("max").fillna(0)
        n_top_cnt = (nt == nmx).groupby(df.cand.values).transform("sum")
        other = np.where((nt >= nmx) & (n_top_cnt <= 1), nsec, nmx)
        F["name_rival_margin"] = nt.values - other
    # sibling support weighted by the top candidate's probability
    if "sib1_name_addr" in df:
        F["sib_top_support"] = df.sib1_name_addr.fillna(0).values * mx.values
    # cluster consistency: does this candidate agree with the consensus of the S1's strong candidates?
    # (decoys copy a real record but shift the house number / change the legal form)
    strong = p.values >= 0.9
    for col, name in (("b_num1", "num"), ("b_legal_code", "legal")):
        if col not in df:
            continue
        v = df[col].values
        sel = strong & ~np.isnan(v)
        cons = (pd.DataFrame({"s1": df.s1.values[sel], "v": v[sel]})
                .groupby(["s1", "v"]).size().rename("n").reset_index()
                .sort_values(["s1", "n"], ascending=[True, False]).drop_duplicates("s1").set_index("s1"))
        mode = pd.Series(df.s1.values).map(cons.v).values.astype(np.float64)
        n_mode = pd.Series(df.s1.values).map(cons.n).fillna(0).values
        F[f"cons_{name}_eq"] = np.where(np.isnan(v) | np.isnan(mode), -1, (v == mode).astype(float))
        F[f"cons_{name}_n"] = n_mode
        if name == "num":
            F["cons_num_diff"] = np.clip(v - mode, -1e6, 1e6)
            a = df["a_num1"].values if "a_num1" in df else np.full(len(df), np.nan)
            F["cons_num_vs_s1"] = np.where(np.isnan(a) | np.isnan(mode), -1, (a == mode).astype(float))
    for c in CARRY:
        if c in df:
            F[f"x_{c}"] = df[c].values
    return F.astype(np.float32)
