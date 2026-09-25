"""Scalable candidate generation, one country at a time.

Two word-level TF-IDF retrievers:
  name : name core tokens + '~'-prefixed consonant-skeleton tokens (typos, transliteration)
  addr : address core tokens + '#'-prefixed house/unit numbers + '@'-prefixed postcode
Very common tokens (document frequency > max_df) are dropped from the retrieval vectors so
the sparse product stays cheap; top-k per query uses sparse_dot_topn (Apache-2.0, multi-threaded).
The union is pruned to max_k per S1 by the mean of the two cosines -> candidate_pairs.tsv.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize
from sparse_dot_topn import sp_matmul_topn

N_FEATURES = 2 ** 22  # hashed vocabulary size; no Python vocabulary dict -> low memory


def _prefix(series: pd.Series, p: str) -> pd.Series:
    return series.str.replace(r"(\S+)", p + r"\1", regex=True)


def _num_tokens(df) -> pd.Series:
    """'#1323 #132' - full numbers plus a 3-digit prefix, so a dropped trailing/middle digit
    (1323 vs 132, 5930 vs 593, 2777 vs 277) still shares a token."""
    out = []
    for h, u in zip(df.house_nums, df.units):
        toks = []
        for n in (h + " " + u).split():
            toks.append("#" + n)
            if len(n) >= 4:
                toks.append("#" + n[:3])
        out.append(" ".join(toks))
    return pd.Series(out, index=df.index, dtype="str")


def name_docs(df):
    return df.name_core + " " + _prefix(df.name_skel, "~")


def addr_docs(df):
    return df.addr_core + " " + _num_tokens(df) + " " + _prefix(df.postal, "@")


def name_addr_docs(df):
    """Name AND address in one vector: a true match must agree on both to rank high, which
    beats generic names ('Eye Clinic') and dense streets (many businesses x ~4.5 copies)."""
    return name_docs(df) + " " + addr_docs(df)


class HashedTfidf:
    """TF-IDF over hashed tokens: sublinear tf, smooth idf, tokens with df > max_df or
    df < 2 get weight 0 (dropped from retrieval), rows L2-normalised."""

    def __init__(self, max_df: int):
        self.hv = HashingVectorizer(token_pattern=r"\S+", lowercase=False, alternate_sign=False,
                                    norm=None, n_features=N_FEATURES, dtype=np.float32)
        self.max_df = max_df

    def fit_transform_pool(self, docs_pool, docs_other):
        Xp = self.hv.transform(docs_pool).tocsr()
        df = np.bincount(Xp.indices, minlength=N_FEATURES)
        Xo = self.hv.transform(docs_other).tocsr()
        df += np.bincount(Xo.indices, minlength=N_FEATURES)
        n = Xp.shape[0] + Xo.shape[0]
        del Xo
        idf = (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)
        idf[(df > self.max_df) | (df < 2)] = 0.0
        self.idf = idf
        return self._weight(Xp)

    def transform(self, docs):
        return self._weight(self.hv.transform(docs).tocsr())

    def _weight(self, X):
        X.data = (1 + np.log(X.data)) * self.idf[X.indices]
        X.eliminate_zeros()
        return l2_normalize(X, copy=False)


DOCS = {"name": name_docs, "addr": addr_docs, "name_addr": name_addr_docs}


def _rowwise_dot(A, B, i, j, chunk=500_000):
    out = np.empty(len(i), np.float32)
    for s in range(0, len(i), chunk):
        out[s:s + chunk] = np.asarray(A[i[s:s + chunk]].multiply(B[j[s:s + chunk]]).sum(axis=1)).ravel()
    return out


class CountryIndex:
    """Fitted vectorizers + pool matrices for one country; query S1 records in chunks."""

    def __init__(self, pool: pd.DataFrame, s1_all: pd.DataFrame, bcfg: dict, threads: int = 7):
        self.pool = pool.reset_index(drop=True)
        self.cfg, self.threads = bcfg, threads
        self.vecs, self.B = {}, {}
        for view in sorted({r["view"] for r in bcfg["retrievers"]}):
            max_df = max(r.get("max_df", 20000) for r in bcfg["retrievers"] if r["view"] == view)
            vec = HashedTfidf(max_df)
            self.B[view] = vec.fit_transform_pool(DOCS[view](self.pool), DOCS[view](s1_all))
            self.vecs[view] = vec
        # name-token IDF for features (own counts, so common tokens keep a LOW idf)
        df = Counter()
        for s in (self.pool.name_core, s1_all.name_core):
            for n in s:
                df.update(set(n.split()))
        N = len(self.pool) + len(s1_all)
        self.idf = {t: math.log((N + 1) / (c + 1)) + 1 for t, c in df.items()}
        key_s1 = s1_all.name_core.value_counts()
        key_pool = self.pool.name_core.value_counts()
        self.chain_s1, self.chain_pool = key_s1, key_pool

    def query(self, q: pd.DataFrame):
        """Return (union, cands) with qi = row in q, pj = row in self.pool."""
        q = q.reset_index(drop=True)
        A = {v: self.vecs[v].transform(DOCS[v](q)).tocsr() for v in self.vecs}
        parts = []
        for r in self.cfg["retrievers"]:
            C = sp_matmul_topn(A[r["view"]], self.B[r["view"]], top_n=r["k"],
                               threshold=r.get("min_sim", 0.0), sort=True, n_threads=self.threads)
            C = C.tocsr()
            rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
            part = pd.DataFrame({"qi": rows, "pj": C.indices.astype(np.int64), "s": C.data})
            part = part.sort_values(["qi", "s"], ascending=[True, False], kind="stable")
            part[f"rank_{r['name']}"] = part.groupby("qi").cumcount() + 1
            parts.append(part[["qi", "pj", f"rank_{r['name']}"]])
        union = parts[0]
        for p in parts[1:]:
            union = union.merge(p, on=["qi", "pj"], how="outer")
        for r in self.cfg["retrievers"]:
            union[f"rank_{r['name']}"] = union[f"rank_{r['name']}"].fillna(r["k"] + 1).astype(np.int16)
        qi, pj = union.qi.values, union.pj.values
        for v in self.vecs:  # exact cosine in every view for every union pair
            union[f"sim_{v}"] = _rowwise_dot(A[v], self.B[v], qi, pj)
        union["n_retr"] = sum((union[f"rank_{r['name']}"] <= r["k"]).astype(np.int8)
                              for r in self.cfg["retrievers"])
        sims = union[[f"sim_{v}" for v in self.vecs]]
        prune = self.cfg.get("prune", {})
        union["prune_score"] = (sims.mean(axis=1) if prune.get("score", "mean") == "mean"
                                else sims.max(axis=1)).astype(np.float32)
        union = union.sort_values(["qi", "prune_score"], ascending=[True, False], kind="stable")
        union["prune_rank"] = (union.groupby("qi").cumcount() + 1).astype(np.int16)
        keep = (union.prune_rank <= prune.get("max_k", 15)) & (union.prune_score >= prune.get("min_score", 0.0))
        return union.reset_index(drop=True), union[keep].reset_index(drop=True)
