"""Turn pair probabilities into one match SET per S1 entity.

Decision rule (see README "Why three thresholds"):
  1. one-to-one: an S2/S3 record keeps only its best-scoring S1 (S1 is deduplicated).
  2. predict the top candidate if p >= t_first            (first match ~ symmetric risk)
  3. + best candidate of the other source if p >= t_other
  4. + any other candidate if p >= t_extra                 (extras: false merge ~2.7x a miss)
  otherwise the entity is predicted as a singleton (empty list).
Thresholds are chosen by grid search on the exact macro F0.5, picking the centre of
a flat region (smoothed argmax) rather than a sharp peak.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from .metrics import f05_vec


def prepare(df: pd.DataFrame, one_to_one: bool = True) -> pd.DataFrame:
    """df needs columns s1, cand, cand_src, p. Returns a sorted frame with decision helpers."""
    d = df[["s1", "cand", "cand_src", "p"]].copy()
    d["pa"] = d.p.values
    if one_to_one:
        best = d.groupby("cand").p.transform("max").values
        d.loc[d.p.values < best, "pa"] = 0.0
    d = d.sort_values(["s1", "pa"], ascending=[True, False], kind="stable").reset_index(drop=True)
    g = d.groupby("s1", sort=False)
    d["is_top"] = (g.cumcount() == 0).values
    d["top_p"] = g.pa.transform("first").values
    top_src = g.cand_src.transform("first").values
    other = d[d.cand_src.values != top_src]
    d["is_other_best"] = False
    d.loc[other.groupby("s1", sort=False).head(1).index, "is_other_best"] = True
    return d


def select(d: pd.DataFrame, t_first: float, t_other: float, t_extra: float) -> np.ndarray:
    pa = d.pa.values
    return (d.top_p.values >= t_first) & (pa > 0) & (
        d.is_top.values | (d.is_other_best.values & (pa >= t_other)) | (pa >= t_extra))


def to_map(d: pd.DataFrame, mask: np.ndarray) -> dict:
    sel = d[mask]
    return sel.groupby("s1", sort=False).cand.apply(list).to_dict()


class Scorer:
    """Fast macro-F0.5 of a selection mask over ALL S1 entities (incl. those with no candidates)."""

    def __init__(self, d: pd.DataFrame, truth_map: dict, target_singleton_rate: float = None):
        ids = list(truth_map)
        code = {s: i for i, s in enumerate(ids)}
        self.codes = d.s1.map(code).values
        self.y = np.fromiter((c in truth_map[s] for s, c in zip(d.s1, d.cand)), float, len(d))
        self.n_truth = np.array([len(truth_map[s]) for s in ids])
        self.E = len(ids)
        self.w = np.ones(self.E)
        if target_singleton_rate is not None:  # reweight to the test singleton prior
            sing = self.n_truth == 0
            s = sing.mean()
            self.w = np.where(sing, target_singleton_rate / max(s, 1e-9),
                              (1 - target_singleton_rate) / max(1 - s, 1e-9))

    def entity_f(self, mask):
        c = self.codes[mask]
        n_pred = np.bincount(c, minlength=self.E)
        tp = np.bincount(c, weights=self.y[mask], minlength=self.E)
        return f05_vec(n_pred, self.n_truth, tp)

    def __call__(self, mask) -> float:
        return float(np.average(self.entity_f(mask), weights=self.w))


def _rng(spec):
    lo, hi, step = spec
    return np.round(np.arange(lo, hi + 1e-9, step), 4)


def grid_search(d: pd.DataFrame, scorer: Scorer, grid: dict):
    tf, to, te = _rng(grid["t_first"]), _rng(grid["t_other"]), _rng(grid["t_extra"])
    S = np.full((len(tf), len(to), len(te)), np.nan)
    for a, t1 in enumerate(tf):
        for b, t2 in enumerate(to):
            for c, t3 in enumerate(te):
                S[a, b, c] = scorer(select(d, t1, t2, t3))
    smooth = uniform_filter(S, size=3, mode="nearest")
    a, b, c = np.unravel_index(np.nanargmax(smooth), S.shape)
    ra, rb, rc = np.unravel_index(np.nanargmax(S), S.shape)
    curve = pd.DataFrame([(t1, t2, t3, S[i, j, k], smooth[i, j, k])
                          for i, t1 in enumerate(tf) for j, t2 in enumerate(to)
                          for k, t3 in enumerate(te)],
                         columns=["t_first", "t_other", "t_extra", "score", "smoothed"])
    return {
        "t_first": float(tf[a]), "t_other": float(to[b]), "t_extra": float(te[c]),
        "score_at_choice": float(S[a, b, c]),
        "raw_best": {"t_first": float(tf[ra]), "t_other": float(to[rb]),
                     "t_extra": float(te[rc]), "score": float(S[ra, rb, rc])},
    }, curve
