"""Candidate generation: several retrievers, unioned with provenance, then pruned.

The pruned output IS candidate_pairs.tsv - the exact set the matcher scores.
All retrieval runs within a country block (when `within_country`), so work
scales with block size, never with |S1| x |S2+S3| globally.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# view name -> (record column, vectorizer kwargs)
VIEWS = {
    "name_char": ("name_core", dict(analyzer="char_wb", ngram_range=(3, 3))),
    "name_word": ("name_core", dict(analyzer="word", token_pattern=r"\S+")),
    "skel_char": ("name_skel", dict(analyzer="char_wb", ngram_range=(2, 3))),
    "addr_word": ("addr_core", dict(analyzer="word", token_pattern=r"\S+")),
    "addr_char": ("addr_clean", dict(analyzer="char_wb", ngram_range=(3, 3))),
    "name_addr_word": ("name_addr", dict(analyzer="word", token_pattern=r"\S+")),
}


class TextIndex:
    """L2-normalised TF-IDF matrices for every view, rows aligned with `recs`.

    Fitted on all records of ONE split (train or test) - unsupervised, provided data only.
    """

    def __init__(self, recs: pd.DataFrame):
        self.mats, self.idfs = {}, {}
        for view, (col, kw) in VIEWS.items():
            vec = TfidfVectorizer(lowercase=False, sublinear_tf=True, dtype=np.float32, **kw)
            try:
                self.mats[view] = vec.fit_transform(recs[col].tolist()).tocsr()
                self.idfs[view] = dict(zip(vec.get_feature_names_out(), vec.idf_))
            except ValueError:  # empty vocabulary (e.g. all addresses blank)
                from scipy.sparse import csr_matrix
                self.mats[view] = csr_matrix((len(recs), 1), dtype=np.float32)
                self.idfs[view] = {}

    def pair_cos(self, view, i, j, chunk=400_000) -> np.ndarray:
        """Cosine similarity for aligned index arrays i, j."""
        M = self.mats[view]
        out = np.empty(len(i), np.float32)
        for s in range(0, len(i), chunk):
            a, b = M[i[s:s + chunk]], M[j[s:s + chunk]]
            out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
        return out


def _topk(Q, P, k, min_sim, budget=2e7):
    """Chunked sparse top-k cosine. Returns (q_local, p_local, sim, rank)."""
    n_q, n_p = Q.shape[0], P.shape[0]
    if n_q == 0 or n_p == 0:
        return [np.empty(0, int)] * 2 + [np.empty(0, np.float32), np.empty(0, int)]
    k = min(k, n_p)
    PT = P.T.tocsc()
    step = max(1, int(budget // n_p))
    out = [[], [], [], []]
    for s in range(0, n_q, step):
        S = (Q[s:s + step] @ PT).toarray()
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k] if k < n_p else \
            np.tile(np.arange(n_p), (S.shape[0], 1))
        sims = np.take_along_axis(S, idx, 1)
        order = np.argsort(-sims, axis=1, kind="stable")
        idx, sims = np.take_along_axis(idx, order, 1), np.take_along_axis(sims, order, 1)
        ranks = np.broadcast_to(np.arange(1, k + 1), idx.shape)
        rows = np.broadcast_to(np.arange(s, s + S.shape[0])[:, None], idx.shape)
        m = sims >= min_sim
        for lst, arr in zip(out, (rows[m], idx[m], sims[m], ranks[m])):
            lst.append(arr)
    return [np.concatenate(x) for x in out]


def _record_keys(recs: pd.DataFrame, key: str) -> list:
    """Exact blocking keys per record (list of keys; empty list = no key)."""
    keys = []
    for bc, postal, skel, hn, un, addr, name in zip(recs.bc, recs.postal, recs.name_skel, recs.house_nums,
                                                    recs.units, recs.addr_core, recs.name_core):
        nums = " ".join(dict.fromkeys((hn + " " + un).split()))  # house + unit/shop numbers
        first_skel = skel.split()[0][:3] if skel else ""
        if key == "postal_name":
            keys.append([f"{bc}|{postal}|{first_skel}"] if postal and first_skel else [])
        elif key == "num_street_name":
            street = addr.split()[0] if addr else ""
            keys.append([f"{bc}|{n}|{street}|{name[:3]}" for n in nums.split()[:3]]
                        if nums and street and name else [])
        elif key == "name_exact":
            keys.append([f"{bc}|{name}"] if name else [])
        else:
            raise ValueError(f"unknown key {key}")
    return keys


def _key_retrieve(recs, q_idx, p_idx, key, max_bucket):
    rkeys = _record_keys(recs, key)
    buckets = {}
    for j in p_idx:
        for kk in rkeys[j]:
            buckets.setdefault(kk, []).append(j)
    qi, pj = [], []
    for i in q_idx:
        hits = set()
        for kk in rkeys[i]:
            b = buckets.get(kk, ())
            if 0 < len(b) <= max_bucket:
                hits.update(b)
        qi += [i] * len(hits)
        pj += list(hits)
    return np.array(qi, int), np.array(pj, int)


def run_blocking(recs: pd.DataFrame, ti: TextIndex, cfg: dict):
    """Return (union, candidates). Both have global row indices qi (S1) and pj (S2/S3).

    union      : every pair any retriever proposed, with per-retriever sim/rank columns
    candidates : union pruned to <= max_k per S1 and prune_score >= min_score
    """
    src = recs.src.values
    groups = recs.bc.values if cfg.get("within_country", True) else np.zeros(len(recs), int)
    parts = []
    for r in cfg["retrievers"]:
        name = r["name"]
        for g in np.unique(groups):
            q_idx = np.where((src == 1) & (groups == g))[0]
            p_idx = np.where((src != 1) & (groups == g))[0]
            if len(q_idx) == 0 or len(p_idx) == 0:
                continue
            if r["type"] == "tfidf":
                M = ti.mats[r["view"]]
                ql, pl, sim, rank = _topk(M[q_idx], M[p_idx], r["k"], r.get("min_sim", 0.0))
                parts.append(pd.DataFrame({"qi": q_idx[ql], "pj": p_idx[pl], "retr": name,
                                           "sim": sim, "rank": rank}))
            elif r["type"] == "key":
                qi, pj = _key_retrieve(recs, q_idx, p_idx, r["key"], r.get("max_bucket", 50))
                parts.append(pd.DataFrame({"qi": qi, "pj": pj, "retr": name,
                                           "sim": 1.0, "rank": 0}))
    long = pd.concat(parts, ignore_index=True)
    union = long[["qi", "pj"]].drop_duplicates().reset_index(drop=True)
    tfidf_names = [r["name"] for r in cfg["retrievers"] if r["type"] == "tfidf"]
    key_names = [r["name"] for r in cfg["retrievers"] if r["type"] == "key"]
    for r in cfg["retrievers"]:
        sub = long[long.retr == r["name"]].drop_duplicates(["qi", "pj"])
        if r["type"] == "tfidf":
            sub = sub.rename(columns={"sim": f"{r['name']}_sim", "rank": f"{r['name']}_rank"})
            union = union.merge(sub[["qi", "pj", f"{r['name']}_sim", f"{r['name']}_rank"]],
                                on=["qi", "pj"], how="left")
            union[f"{r['name']}_rank"] = union[f"{r['name']}_rank"].fillna(r["k"] + 1)
            # fill sims for pairs this retriever did not return with their true cosine
            miss = union[f"{r['name']}_sim"].isna().values
            if miss.any():
                union.loc[miss, f"{r['name']}_sim"] = ti.pair_cos(
                    r["view"], union.qi.values[miss], union.pj.values[miss])
        else:
            sub = sub.assign(**{f"{r['name']}_hit": 1})[["qi", "pj", f"{r['name']}_hit"]]
            union = union.merge(sub, on=["qi", "pj"], how="left")
            union[f"{r['name']}_hit"] = union[f"{r['name']}_hit"].fillna(0).astype(np.int8)
    union["n_retr"] = sum((union[f"{n}_rank"] <= next(r["k"] for r in cfg["retrievers"]
                                                       if r["name"] == n)).astype(int)
                          for n in tfidf_names) + sum(union[f"{n}_hit"] for n in key_names)
    prune = cfg.get("prune", {})
    sims = union[[f"{n}_sim" for n in tfidf_names]]
    # "mean" rewards agreement across name AND address views, which separates chain branches
    score = sims.mean(axis=1) if prune.get("score", "mean") == "mean" else sims.max(axis=1)
    if key_names:
        score = score + prune.get("key_bonus", 0.2) * union[[f"{n}_hit" for n in key_names]].max(axis=1)
    union["prune_score"] = score.astype(np.float32)
    union = union.sort_values(["qi", "prune_score"], ascending=[True, False], kind="stable")
    union["prune_rank"] = union.groupby("qi").cumcount() + 1
    keep = (union.prune_rank <= prune.get("max_k", 10)) & \
           (union.prune_score >= prune.get("min_score", 0.0))
    cands = union[keep].reset_index(drop=True)
    return union.reset_index(drop=True), cands
