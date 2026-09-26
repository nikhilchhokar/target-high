"""Second-stage (stacked) matcher: re-score each pair using the stage-1 probabilities of its
whole neighbourhood - the other candidates of the same S1 and the other S1 competing for the
same candidate. Stage-1 inputs are out-of-fold on train and full-model predictions on test.
"""
import numpy as np
import pandas as pd

# raw pair features carried into stage 2 alongside the group features
CARRY = ["sim_name_addr", "sim_name", "sim_addr", "n_tset", "a_tset", "num_signed_diff",
         "num_state", "a_missing", "chain_s1", "sib1_name_addr", "sib1_addr", "sib1_name",
         "sib2_name_addr", "cand_src", "prune_rank", "n_nospace_partial"]


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
    # sibling support weighted by the top candidate's probability
    if "sib1_name_addr" in df:
        F["sib_top_support"] = df.sib1_name_addr.fillna(0).values * mx.values
    for c in CARRY:
        if c in df:
            F[f"x_{c}"] = df[c].values
    return F.astype(np.float32)
