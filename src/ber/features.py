"""Pairwise + context features for (S1, candidate) pairs.

Every feature is country-agnostic by construction (no country one-hot): France is
unseen in training, so anything that only works for US/India would not transfer.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .blocking import TextIndex

COS_VIEWS = ["name_char", "name_word", "skel_char", "addr_word", "addr_char", "name_addr_word"]


def _set_state(a: str, b: str) -> int:
    """0 both missing, 1 one missing, 2 equal, 3 overlap/subset, 4 disjoint (conflict)."""
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 0
    if not sa or not sb:
        return 1
    if sa == sb:
        return 2
    return 3 if sa & sb else 4


def _jacc(sa, sb):
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def _name_token_feats(na, nb, idf, max_idf):
    ta, tb = set(na.split()), set(nb.split())
    w = lambda t: idf.get(t, max_idf)
    union = ta | tb
    shared = ta & tb
    diff = ta ^ tb
    wu = sum(w(t) for t in union)
    return (
        _jacc(ta, tb),
        sum(w(t) for t in shared) / wu if wu else 0.0,     # IDF-weighted overlap
        max((w(t) for t in diff), default=0.0),              # rarest non-shared token
        max((w(t) for t in shared), default=0.0),            # rarest shared token
        float(bool(ta) and bool(tb) and na.split()[0] == nb.split()[0]),
        float(bool(na) and bool(nb) and (na in nb or nb in na)),
    )


def pair_features(cands: pd.DataFrame, recs: pd.DataFrame, ti: TextIndex) -> pd.DataFrame:
    i, j = cands.qi.values, cands.pj.values
    A = recs.iloc[i].reset_index(drop=True)
    B = recs.iloc[j].reset_index(drop=True)
    F = {}
    for v in COS_VIEWS:
        F[f"cos_{v}"] = ti.pair_cos(v, i, j)

    na, nb = A.name_core.tolist(), B.name_core.tolist()
    F["n_ratio"] = [fuzz.ratio(a, b) for a, b in zip(na, nb)]
    F["n_partial"] = [fuzz.partial_ratio(a, b) for a, b in zip(na, nb)]
    F["n_tsort"] = [fuzz.token_sort_ratio(a, b) for a, b in zip(na, nb)]
    F["n_tset"] = [fuzz.token_set_ratio(a, b) for a, b in zip(na, nb)]
    F["n_jw"] = [JaroWinkler.similarity(a, b) for a, b in zip(na, nb)]
    F["n_lev"] = [Levenshtein.normalized_similarity(a, b) for a, b in zip(na, nb)]
    F["n_clean_tset"] = [fuzz.token_set_ratio(a, b) for a, b in zip(A.name_clean, B.name_clean)]
    F["n_skel_ratio"] = [fuzz.ratio(a, b) for a, b in zip(A.name_skel, B.name_skel)]

    idf = ti.idfs["name_word"]
    max_idf = max(idf.values()) if idf else 1.0
    tok = np.array([_name_token_feats(a, b, idf, max_idf) for a, b in zip(na, nb)], np.float32)
    for k, name in enumerate(["n_jacc", "n_idf_overlap", "n_max_idf_unshared",
                              "n_max_idf_shared", "n_first_tok_eq", "n_contain"]):
        F[name] = tok[:, k]
    F["n_acronym"] = [float(bool(x) and (x == b.replace(" ", "") or y == a.replace(" ", "")))
                      for a, b, x, y in zip(na, nb, A.name_acr, B.name_acr)]
    F["legal_state"] = [_set_state(a, b) for a, b in zip(A.legal, B.legal)]
    # best similarity using DBA/trade-name parts on either side
    F["n_dba_best"] = [max(fuzz.token_set_ratio(a, db) if db else 0,
                           fuzz.token_set_ratio(da, b) if da else 0)
                       for a, b, da, db in zip(na, nb, A.dba_core, B.dba_core)]
    F["name_num_state"] = [_set_state(a, b) for a, b in zip(A.name_nums, B.name_nums)]
    la, lb = np.array([len(x) for x in na]), np.array([len(x) for x in nb])
    F["n_len_min"] = np.minimum(la, lb)
    F["n_len_diff"] = np.abs(la - lb) / np.maximum(np.maximum(la, lb), 1)
    F["n_tok_diff"] = np.abs(np.array([len(x.split()) for x in na]) - np.array([len(x.split()) for x in nb]))

    aa, ab = A.addr_core.tolist(), B.addr_core.tolist()
    F["a_tset"] = [fuzz.token_set_ratio(a, b) for a, b in zip(aa, ab)]
    F["a_tsort"] = [fuzz.token_sort_ratio(a, b) for a, b in zip(aa, ab)]
    F["a_ratio"] = [fuzz.ratio(a, b) for a, b in zip(aa, ab)]
    F["a_clean_partial"] = [fuzz.partial_ratio(a, b) for a, b in zip(A.addr_clean, B.addr_clean)]
    F["a_jacc"] = [_jacc(set(a.split()), set(b.split())) for a, b in zip(aa, ab)]
    F["postal_state"] = [_set_state(a, b) for a, b in zip(A.postal, B.postal)]
    F["postal_pref3"] = [float(bool(a) and bool(b) and a[:3] == b[:3]) for a, b in zip(A.postal, B.postal)]
    F["num_state"] = [_set_state(a, b) for a, b in zip(A.house_nums, B.house_nums)]
    F["num_jacc"] = [_jacc(set(a.split()), set(b.split())) for a, b in zip(A.house_nums, B.house_nums)]
    F["unit_state"] = [_set_state(a, b) for a, b in zip(A.units, B.units)]
    F["landmark_both"] = [float(bool(a) and bool(b)) for a, b in zip(A.landmark, B.landmark)]
    F["landmark_any"] = [float(bool(a) or bool(b)) for a, b in zip(A.landmark, B.landmark)]
    F["landmark_tset"] = [fuzz.token_set_ratio(a, b) if a and b else -1 for a, b in zip(A.landmark, B.landmark)]
    F["a_missing"] = [int(not a) + int(not b) for a, b in zip(aa, ab)]
    F["a_len_min"] = np.minimum([len(x) for x in aa], [len(x) for x in ab])

    F["cand_src"] = B.src.values
    F["same_country"] = (A.bc.values == B.bc.values).astype(np.int8)
    # chain-ness: how many S1 / pool records share this core name in this country
    key = recs.bc + "|" + recs.name_core
    s1_counts = key[recs.src == 1].value_counts()
    pool_counts = key[recs.src != 1].value_counts()
    ka = (A.bc + "|" + A.name_core)
    F["chain_s1"] = ka.map(s1_counts).fillna(0).values
    F["chain_pool"] = ka.map(pool_counts).fillna(0).values
    return pd.DataFrame(F)


def context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Competition features: how a candidate compares with its rivals (both directions)."""
    out = {}
    for col in ["cos_name_char", "prune_score", "cos_name_addr_word"]:
        g = df.groupby("qi")[col]
        out[f"ctx_rank_{col}"] = g.rank(ascending=False, method="min").values
        out[f"ctx_gap_{col}"] = (g.transform("max") - df[col]).values
        r = df.groupby("pj")[col]
        out[f"rev_rank_{col}"] = r.rank(ascending=False, method="min").values
        out[f"rev_gap_{col}"] = (r.transform("max") - df[col]).values
    out["ctx_n_cands"] = df.groupby("qi").qi.transform("size").values
    out["ctx_n_hi_name"] = (df.cos_name_char > 0.8).groupby(df.qi).transform("sum").values
    out["rev_n_s1"] = df.groupby("pj").pj.transform("size").values
    return pd.DataFrame(out, index=df.index)


def build_features(cands: pd.DataFrame, recs: pd.DataFrame, ti: TextIndex, cfg: dict) -> pd.DataFrame:
    pf = pair_features(cands, recs, ti)
    df = pd.concat([cands.reset_index(drop=True), pf], axis=1)
    if cfg.get("context", True):
        df = pd.concat([df, context_features(df)], axis=1)
    return df
