"""Second-stage (stacked) matcher: re-score each pair using the stage-1 probabilities of its
whole neighbourhood - the other candidates of the same S1 and the other S1 competing for the
same candidate. Stage-1 inputs are out-of-fold on train and full-model predictions on test.
"""
import numpy as np
import pandas as pd

# raw pair features carried into stage 2 alongside the group features
# EXPANDED (e009): include the discriminative fuzzy / IDF / numeric features that
# drive top gain in stage 1 but were missing from the original stage-2 CARRY.
CARRY = ["sim_name_addr", "sim_name", "sim_addr", "n_tset", "a_tset", "num_signed_diff",
         "num_state", "a_missing", "chain_s1", "sib1_name_addr", "sib1_addr", "sib1_name",
         "sib2_name_addr", "cand_src", "prune_rank", "n_nospace_partial",
         # New: top stage-1 gain features (e009)
         "n_partial", "n_ratio", "n_nospace_ratio", "n_skel_ratio", "n_max_idf_unshared",
         "n_idf_overlap", "n_jacc", "n_jw", "n_lev", "n_first_tok_eq", "n_contain",
         "legal_state", "name_num_state", "num_jacc", "num_near", "num_unmatched_b",
         "num_min_absdiff", "a_partial", "a_jacc", "postal_state", "postal_pref3",
         "b_pmb_box", "unit_state", "unit_signed_diff", "landmark_both", "landmark_tset",
         "n_tok_diff", "n_len_min", "n_len_diff", "a_len_min",
         # Char n-gram features (e010) — handle transliteration + typos
         "n_cng3_jacc", "n_cng3_intersect", "n_cng4_jacc",
         "n_init_match", "n_first_in_second", "a_cng3_jacc",
         # Competition / context from add_context
         "ctx_rank_prune_score", "ctx_rank_n_tset", "ctx_rank_a_tset",
         "ctx_gap_prune_score", "ctx_gap_n_tset", "ctx_gap_a_tset",
         "ctx_n_cands", "ctx_n_hi_name", "ctx_src_count",
         "rev_rank_prune_score", "rev_rank_sim_name_addr",
         "rev_gap_prune_score", "rev_gap_sim_name_addr", "rev_n_s1"]


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
