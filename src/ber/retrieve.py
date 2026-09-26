"""Scalable candidate generation, one country at a time.

Two word-level TF-IDF retrievers:
  name : name core tokens + '~'-prefixed consonant-skeleton tokens (typos, transliteration)
  addr : address core tokens + '#'-prefixed house/unit numbers + '@'-prefixed postcode
Very common tokens (document frequency > max_df) are dropped from the retrieval vectors so
the sparse product stays cheap; top-k per query uses sparse_dot_topn (Apache-2.0, multi-threaded).
The union is pruned to max_k per S1 by the mean of the two cosines -> candidate_pairs.tsv.
"""
import math
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize
from sparse_dot_topn import sp_matmul_topn

from . import dictionaries as D

N_FEATURES = 2 ** 24  # hashed vocabulary size (bigrams need room); no vocabulary dict -> low memory


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


_US_ABBR = set(D.US_STATES.values())
_US_FULL = {k.upper(): v for k, v in D.US_STATES.items()}
_IN_REGION = set(D.IN_STATES.values()) | {
    "goa", "assam", "odisha", "jharkhand", "chhattisgarh", "tripura", "sikkim", "manipur",
    "meghalaya", "mizoram", "nagaland", "chandigarh", "puducherry", "andhrapradesh"}
_UPPER2 = re.compile(r"(?<![A-Z])([A-Z]{2})(?![A-Z])")  # standalone 2-letter codes


def region_of(df) -> pd.Series:
    """Coarse region per record: US state (from the RAW address, because the cleaned one maps
    CT->court, FL->floor), Indian state token, else the first 2 postcode digits (France: departement)."""
    out = []
    for bc, raw, core, postal in zip(df.bc, df.business_address, df.addr_core, df.postal):
        r = ""
        if bc == "us":
            up = raw.upper()
            codes = [c.lower() for c in _UPPER2.findall(up) if c.lower() in _US_ABBR]
            if codes:
                r = codes[-1]
            else:
                for full, ab in _US_FULL.items():
                    if full in up:
                        r = ab
                        break
        elif bc == "in":
            for t in reversed(core.split()):
                if t in _IN_REGION:
                    r = t
                    break
        if not r and postal:
            r = postal[:2]
        out.append(r)
    return pd.Series(out, index=df.index, dtype="str")


def _bigrams(series: pd.Series, sep="|", sort=False) -> pd.Series:
    out = []
    for s in series:
        t = s.split()
        pairs = zip(t, t[1:])
        out.append(" ".join((sep.join(sorted(p)) if sort else sep.join(p)) for p in pairs))
    return pd.Series(out, index=series.index, dtype="str")


def _tag(series: pd.Series, region: pd.Series) -> pd.Series:
    """'clark industries' + 'az' -> 'clark@az industries@az' (common words become specific)."""
    return pd.Series([" ".join(f"{t}@{r}" for t in s.split()) if r else "" for s, r in zip(series, region)],
                     index=series.index, dtype="str")


def _joined(series: pd.Series) -> pd.Series:
    """'emmy love dds' -> '=emmylove =emmylovedds': matches website/handle names written without
    spaces ('@emmylove', 'maguiresprairiecafe.com')."""
    out = []
    for s in series:
        t = [x for x in s.split() if x not in ("com", "www", "net", "org")]
        toks = []
        if len(t) >= 2:
            toks.append("=" + "".join(t[:2]))
        if len(t) >= 3:
            toks.append("=" + "".join(t))
        if len(t) == 1 and len(t[0]) >= 6:
            toks.append("=" + t[0])
        out.append(" ".join(toks))
    return pd.Series(out, index=series.index, dtype="str")


def name_docs(df):
    """Name unigrams (+ DBA part) + skeleton + order-free word pairs + joined forms + region tags."""
    names = (df.name_core + " " + df.dba_core).str.strip()
    return (names + " " + _prefix(df.name_skel, "~") + " " + _bigrams(df.name_core, sort=True)
            + " " + _bigrams(df.dba_core, sort=True) + " " + _joined(df.name_core) + " "
            + _joined(df.dba_core) + " " + _tag(df.name_core, df.region)
            + " " + _tag(_prefix(df.name_skel, "~"), df.region))  # '~svstk@bihar': translit/typos


def addr_docs(df):
    """Address unigrams + house numbers + adjacent word pairs + postcode."""
    return (df.addr_core + " " + _num_tokens(df) + " " + _bigrams(df.addr_core) + " "
            + _prefix(df.postal, "@"))


def name_addr_docs(df):
    """Name AND address in one vector: a true match must agree on both to rank high, which
    beats generic names ('Eye Clinic') and dense streets (many businesses x ~4.5 copies)."""
    return name_docs(df) + " " + addr_docs(df)


class HashedTfidf:
    """TF-IDF over hashed token counts: sublinear tf, smooth idf, tokens with df > max_df or
    df < 2 get weight 0 (dropped from retrieval), rows L2-normalised."""

    def __init__(self, max_df: int):
        self.max_df = max_df

    def fit_weight_pool(self, Xp, Xo):
        df = np.bincount(Xp.indices, minlength=N_FEATURES) + np.bincount(Xo.indices, minlength=N_FEATURES)
        n = Xp.shape[0] + Xo.shape[0]
        idf = (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)
        idf[(df > self.max_df) | (df < 2)] = 0.0
        self.idf = idf
        return self.weight(Xp)

    def weight(self, X):
        X.data = (1 + np.log(X.data)) * self.idf[X.indices]
        X.eliminate_zeros()
        return l2_normalize(X, copy=False)


_HV = HashingVectorizer(token_pattern=r"\S+", lowercase=False, alternate_sign=False, norm=None,
                        n_features=N_FEATURES, dtype=np.float32)
_DOC_COLS = ["bc", "business_address", "addr_core", "postal", "name_core", "dba_core", "name_skel",
             "house_nums", "units", "region"]


def _hash_chunk(df: pd.DataFrame, views):
    """Build every view's document and hash it (runs in a worker process)."""
    if "region" not in df:
        df = df.assign(region=region_of(df))
    name, addr = name_docs(df), addr_docs(df)
    docs = {"name": name, "addr": addr, "name_addr": name + " " + addr}
    return {v: _HV.transform(docs[v]).tocsr() for v in views}, df.region.values


def hash_views(df: pd.DataFrame, views, workers: int = 7, chunk: int = 100_000):
    """Parallel doc building + hashing -> ({view: count CSR}, region array), rows aligned with df."""
    cols = [c for c in _DOC_COLS if c in df]
    starts = list(range(0, len(df), chunk))
    if workers <= 1 or len(starts) <= 1:
        res = [_hash_chunk(df[cols].iloc[s:s + chunk], views) for s in starts]
    else:
        res = [None] * len(starts)
        with ProcessPoolExecutor(workers) as ex:
            pending = {}
            for k, s in enumerate(starts):
                if len(pending) >= workers * 2:  # bounded queue keeps memory flat
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        res[pending.pop(f)] = f.result()
                pending[ex.submit(_hash_chunk, df[cols].iloc[s:s + chunk], views)] = k
            for f in pending:
                res[pending[f]] = f.result()
    X = {v: sp.vstack([r[0][v] for r in res]).tocsr() for v in views}
    return X, np.concatenate([r[1] for r in res])


DOCS = {"name": name_docs, "addr": addr_docs, "name_addr": name_addr_docs}


def _rowwise_dot(A, B, i, j, chunk=500_000):
    out = np.empty(len(i), np.float32)
    for s in range(0, len(i), chunk):
        out[s:s + chunk] = np.asarray(A[i[s:s + chunk]].multiply(B[j[s:s + chunk]]).sum(axis=1)).ravel()
    return out


def prune_features(union: pd.DataFrame, views, retr, cand_src: np.ndarray,
                   q_indic: np.ndarray = None, c_indic: np.ndarray = None) -> pd.DataFrame:
    """Cheap per-pair features for the learned pruner (no string work, fully vectorised)."""
    F = pd.DataFrame(index=union.index)
    g = union.groupby("qi")
    for v in views:
        col = union[f"sim_{v}"]
        mx = g[f"sim_{v}"].transform("max")
        F[f"sim_{v}"] = col
        F[f"gap_{v}"] = mx - col
        F[f"qmax_{v}"] = mx
        F[f"qrank_{v}"] = g[f"sim_{v}"].rank(ascending=False, method="min")
    for r in retr:
        F[f"rank_{r['name']}"] = union[f"rank_{r['name']}"]
    F["n_retr"] = union.n_retr
    F["q_union"] = g.qi.transform("size")
    F["cand_src"] = cand_src
    if q_indic is not None:  # names in different scripts share few tokens: don't punish them
        F["q_indic"] = q_indic
        F["c_indic"] = c_indic
    return F.astype(np.float32)


class CountryIndex:
    """Fitted vectorizers + pool matrices for one country; query S1 records in chunks."""

    def __init__(self, pool: pd.DataFrame, s1_all: pd.DataFrame, bcfg: dict, threads: int = 7):
        self.pool = pool.reset_index(drop=True)
        self.cfg, self.threads = bcfg, threads
        self.vecs, self.B = {}, {}
        views = sorted({r["view"] for r in bcfg["retrievers"]})
        Xp, self.pool["region"] = hash_views(self.pool, views, threads)
        Xo, _ = hash_views(s1_all, views, threads)
        for view in views:
            max_df = max(r.get("max_df", 20000) for r in bcfg["retrievers"] if r["view"] == view)
            vec = HashedTfidf(max_df)
            self.B[view] = vec.fit_weight_pool(Xp[view], Xo[view])
            self.vecs[view] = vec
        del Xp, Xo
        # name-token IDF for features (own counts, so common tokens keep a LOW idf)
        df = Counter()
        for s in (self.pool.name_core, s1_all.name_core):
            for n in s:
                df.update(set(n.split()))
        N = len(self.pool) + len(s1_all)
        self.idf = {t: math.log((N + 1) / (c + 1)) + 1 for t, c in df.items()}
        model_path = bcfg.get("prune", {}).get("model")
        self.pruner = None
        if bcfg.get("prune", {}).get("score") == "model" and model_path:
            import lightgbm as lgb
            self.pruner = lgb.Booster(model_file=str(model_path))
        key_s1 = s1_all.name_core.value_counts()
        key_pool = self.pool.name_core.value_counts()
        self.chain_s1, self.chain_pool = key_s1, key_pool

    def query(self, q: pd.DataFrame):
        """Return (union, cands) with qi = row in q, pj = row in self.pool."""
        q = q.reset_index(drop=True)
        X, _ = hash_views(q, list(self.vecs), self.threads)
        A = {v: self.vecs[v].weight(X[v]) for v in self.vecs}
        parts = []
        for r in self.cfg["retrievers"]:
            C = sp_matmul_topn(A[r["view"]], self.B[r["view"]], top_n=r["k"],
                               threshold=r.get("min_sim", 0.0), sort=True, n_threads=self.threads)
            C = C.tocsr()
            rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
            part = pd.DataFrame({"qi": rows, "pj": C.indices.astype(np.int64), "s": C.data})
            part = part.sort_values(["qi", "s"], ascending=[True, False], kind="stable")
            part[f"rank_{r['name']}"] = part.groupby("qi").cumcount() + 1
            part[f"s_{r['name']}"] = part.s.astype(np.float32)
            parts.append(part[["qi", "pj", f"rank_{r['name']}", f"s_{r['name']}"]])
        union = parts[0]
        for p in parts[1:]:
            union = union.merge(p, on=["qi", "pj"], how="outer")
        for r in self.cfg["retrievers"]:
            union[f"rank_{r['name']}"] = union[f"rank_{r['name']}"].fillna(r["k"] + 1).astype(np.int16)
        qi, pj = union.qi.values, union.pj.values
        for v in self.vecs:  # exact cosine in every view: reuse the retriever's own score, compute the rest
            col = np.full(len(union), np.nan, np.float32)
            for r in self.cfg["retrievers"]:
                if r["view"] == v:
                    got = union[f"s_{r['name']}"].values
                    fill = np.isnan(col) & ~np.isnan(got)
                    col[fill] = got[fill]
            miss = np.isnan(col)
            if miss.any():
                col[miss] = _rowwise_dot(A[v], self.B[v], qi[miss], pj[miss])
            union[f"sim_{v}"] = col
        union = union.drop(columns=[f"s_{r['name']}" for r in self.cfg["retrievers"]])
        union["n_retr"] = sum((union[f"rank_{r['name']}"] <= r["k"]).astype(np.int8)
                              for r in self.cfg["retrievers"])
        sims = union[[f"sim_{v}" for v in self.vecs]]
        prune = self.cfg.get("prune", {})
        if self.pruner is not None:  # learned ranking of the union (see train_pruner.py)
            F = prune_features(union, list(self.vecs), self.cfg["retrievers"],
                               self.pool.src.values[union.pj.values],
                               q.name_indic.values[union.qi.values].astype(np.float32),
                               self.pool.name_indic.values[union.pj.values].astype(np.float32))
            union["prune_score"] = self.pruner.predict(F[self.pruner.feature_name()].to_numpy()).astype(np.float32)
        else:
            union["prune_score"] = (sims.mean(axis=1) if prune.get("score", "mean") == "mean"
                                    else sims.max(axis=1)).astype(np.float32)
        union = union.sort_values(["qi", "prune_score"], ascending=[True, False], kind="stable")
        union["prune_rank"] = (union.groupby("qi").cumcount() + 1).astype(np.int16)
        keep = (union.prune_rank <= prune.get("max_k", 15)) & (union.prune_score >= prune.get("min_score", 0.0))
        return union.reset_index(drop=True), union[keep].reset_index(drop=True)
