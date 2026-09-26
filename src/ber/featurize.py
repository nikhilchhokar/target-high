"""Pair + context features, computed in parallel chunks.

a_* columns = the S1 record, b_* = the candidate. Every feature is country-agnostic
(France is unseen in training). Fuzzy string features use rapidfuzz (MIT).
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

VIEW_COLS = ["name_core", "legal", "dba_core", "name_skel", "name_nums", "name_acr", "name_indic",
             "addr_core", "postal", "house_nums", "units", "landmark"]


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


def _drop_one(x: str, y: str) -> bool:
    """True if y is x with one digit dropped (2777 -> 277), a common noise pattern."""
    return len(x) == len(y) + 1 and any(x[:k] + x[k + 1:] == y for k in range(len(x)))


def _num_near(a: str, b: str) -> int:
    """2 exact number shared, 1 near (one digit dropped), 0 none, -1 a side has no number."""
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return -1
    if sa & sb:
        return 2
    return int(any(_drop_one(x, y) or _drop_one(y, x) for x in sa for y in sb))


def _tok(na, nb, idf, max_idf):
    ta, tb = set(na.split()), set(nb.split())
    w = lambda t: idf.get(t, max_idf)
    wu = sum(w(t) for t in ta | tb)
    return (_jacc(ta, tb),
            sum(w(t) for t in ta & tb) / wu if wu else 0.0,
            max((w(t) for t in ta ^ tb), default=0.0),
            max((w(t) for t in ta & tb), default=0.0),
            float(bool(ta) and bool(tb) and na.split()[0] == nb.split()[0]),
            float(bool(na) and bool(nb) and (na in nb or nb in na)))


def features_chunk(P: pd.DataFrame, idf: dict) -> pd.DataFrame:
    """Pure function over a pair frame (runs in a worker process)."""
    F = {}
    na, nb = P.a_name_core.tolist(), P.b_name_core.tolist()
    F["n_ratio"] = [fuzz.ratio(a, b) for a, b in zip(na, nb)]
    F["n_partial"] = [fuzz.partial_ratio(a, b) for a, b in zip(na, nb)]
    F["n_tsort"] = [fuzz.token_sort_ratio(a, b) for a, b in zip(na, nb)]
    F["n_tset"] = [fuzz.token_set_ratio(a, b) for a, b in zip(na, nb)]
    F["n_jw"] = [JaroWinkler.similarity(a, b) for a, b in zip(na, nb)]
    F["n_lev"] = [Levenshtein.normalized_similarity(a, b) for a, b in zip(na, nb)]
    # names written without spaces: '@emmylove', 'maguiresprairiecafe.com'
    ja = [a.replace(" ", "") for a in na]
    jb = [b.replace(" ", "").replace("com", "") for b in nb]
    F["n_nospace_ratio"] = [fuzz.ratio(a, b) for a, b in zip(ja, jb)]
    F["n_nospace_partial"] = [fuzz.partial_ratio(a, b) if min(len(a), len(b)) >= 5 else 0
                              for a, b in zip(ja, jb)]
    sa, sb = P.a_name_skel.tolist(), P.b_name_skel.tolist()
    F["n_skel_ratio"] = [fuzz.ratio(a, b) for a, b in zip(sa, sb)]
    F["n_skel_tset"] = [fuzz.token_set_ratio(a, b) for a, b in zip(sa, sb)]
    max_idf = max(idf.values()) if idf else 1.0
    tok = np.array([_tok(a, b, idf, max_idf) for a, b in zip(na, nb)], np.float32).reshape(-1, 6)
    for k, name in enumerate(["n_jacc", "n_idf_overlap", "n_max_idf_unshared", "n_max_idf_shared",
                              "n_first_tok_eq", "n_contain"]):
        F[name] = tok[:, k]
    F["n_acronym"] = [float(bool(x) and (x == b.replace(" ", "") or y == a.replace(" ", "")))
                      for a, b, x, y in zip(na, nb, P.a_name_acr, P.b_name_acr)]
    F["legal_state"] = [_set_state(a, b) for a, b in zip(P.a_legal, P.b_legal)]
    F["n_dba_best"] = [max(fuzz.token_set_ratio(a, db) if db else 0, fuzz.token_set_ratio(da, b) if da else 0)
                       for a, b, da, db in zip(na, nb, P.a_dba_core, P.b_dba_core)]
    F["name_num_state"] = [_set_state(a, b) for a, b in zip(P.a_name_nums, P.b_name_nums)]
    la, lb = np.array([len(x) for x in na]), np.array([len(x) for x in nb])
    F["n_len_min"] = np.minimum(la, lb)
    F["n_len_diff"] = np.abs(la - lb) / np.maximum(np.maximum(la, lb), 1)
    F["n_tok_diff"] = np.abs(np.array([len(x.split()) for x in na]) - np.array([len(x.split()) for x in nb]))
    F["b_name_indic"] = P.b_name_indic.astype(np.int8).values
    F["a_name_indic"] = P.a_name_indic.astype(np.int8).values

    aa, ab = P.a_addr_core.tolist(), P.b_addr_core.tolist()
    F["a_tset"] = [fuzz.token_set_ratio(a, b) for a, b in zip(aa, ab)]
    F["a_tsort"] = [fuzz.token_sort_ratio(a, b) for a, b in zip(aa, ab)]
    F["a_ratio"] = [fuzz.ratio(a, b) for a, b in zip(aa, ab)]
    F["a_partial"] = [fuzz.partial_ratio(a, b) for a, b in zip(aa, ab)]
    F["a_jacc"] = [_jacc(set(a.split()), set(b.split())) for a, b in zip(aa, ab)]
    F["postal_state"] = [_set_state(a, b) for a, b in zip(P.a_postal, P.b_postal)]
    F["postal_pref3"] = [float(bool(a) and bool(b) and a[:3] == b[:3]) for a, b in zip(P.a_postal, P.b_postal)]
    F["num_state"] = [_set_state(a, b) for a, b in zip(P.a_house_nums, P.b_house_nums)]
    F["num_jacc"] = [_jacc(set(a.split()), set(b.split())) for a, b in zip(P.a_house_nums, P.b_house_nums)]
    F["num_near"] = [_num_near(a, b) for a, b in zip(P.a_house_nums, P.b_house_nums)]
    F["unit_state"] = [_set_state(a, b) for a, b in zip(P.a_units, P.b_units)]
    F["landmark_both"] = [float(bool(a) and bool(b)) for a, b in zip(P.a_landmark, P.b_landmark)]
    F["landmark_tset"] = [fuzz.token_set_ratio(a, b) if a and b else -1 for a, b in zip(P.a_landmark, P.b_landmark)]
    F["a_missing"] = [int(not a) + int(not b) for a, b in zip(aa, ab)]
    F["a_len_min"] = np.minimum([len(x) for x in aa], [len(x) for x in ab])
    out = pd.DataFrame(F, index=P.index)
    return out.astype(np.float32)


def _pairs(cands, q, pool):
    A = q[VIEW_COLS].iloc[cands.qi.values].reset_index(drop=True).add_prefix("a_")
    B = pool[VIEW_COLS].iloc[cands.pj.values].reset_index(drop=True).add_prefix("b_")
    return pd.concat([A, B], axis=1)


def compute(cands: pd.DataFrame, q: pd.DataFrame, index, workers: int = 7, chunk: int = 50_000) -> pd.DataFrame:
    """All features for one query chunk: retrieval columns + pair features + context features."""
    q = q.reset_index(drop=True)
    pool = index.pool
    base = cands.reset_index(drop=True)
    P = _pairs(base, q, pool)
    chunks = []
    for s in range(0, len(P), chunk):
        sub = P.iloc[s:s + chunk]
        toks = set(" ".join(sub.a_name_core).split()) | set(" ".join(sub.b_name_core).split())
        chunks.append((sub, {t: index.idf[t] for t in toks if t in index.idf}))
    if workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(workers) as ex:
            feats = list(ex.map(features_chunk, *zip(*chunks)))
    else:
        feats = [features_chunk(p, i) for p, i in chunks]
    F = pd.concat(feats) if feats else pd.DataFrame()
    df = pd.concat([base, F.reset_index(drop=True)], axis=1)
    df["cand_src"] = pool.src.values[base.pj.values].astype(np.int8)
    df["chain_s1"] = P.a_name_core.map(index.chain_s1).fillna(0).values.astype(np.float32)
    df["chain_pool"] = P.a_name_core.map(index.chain_pool).fillna(0).values.astype(np.float32)
    return add_context(df)


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Competition features within each S1 entity's candidate list."""
    g = df.groupby("qi")
    for col in ["sim_name", "sim_addr", "prune_score", "n_tset", "a_tset"]:
        if col in df:
            df[f"ctx_rank_{col}"] = g[col].rank(ascending=False, method="min").astype(np.float32)
            df[f"ctx_gap_{col}"] = (g[col].transform("max") - df[col]).astype(np.float32)
    df["ctx_n_cands"] = g.qi.transform("size").astype(np.float32)
    df["ctx_n_hi_name"] = (df.n_tset >= 90).groupby(df.qi).transform("sum").astype(np.float32)
    df["ctx_src_count"] = df.groupby(["qi", "cand_src"]).qi.transform("size").astype(np.float32)
    # reverse competition: how this S1 ranks among ALL S1 that retrieved the same candidate
    r = df.groupby("pj")
    for col in ["prune_score", "sim_name_addr", "n_tset", "a_tset"]:
        if col in df:
            df[f"rev_rank_{col}"] = r[col].rank(ascending=False, method="min").astype(np.float32)
            df[f"rev_gap_{col}"] = (r[col].transform("max") - df[col]).astype(np.float32)
    df["rev_n_s1"] = r.pj.transform("size").astype(np.float32)
    return df
