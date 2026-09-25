"""Local replica of the competition metric plus blocking diagnostics.

Known from the problem statement: F0.5 per S1 entity, macro-averaged over ALL S1
entities; singletons score 1.0 for an empty prediction and 0.0 otherwise.
Inferred: a matched entity with an empty prediction scores 0 (recall = 0).
"""
import numpy as np
import pandas as pd


def f05(pred: set, truth: set) -> float:
    if not truth:
        return float(not pred)
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def f05_vec(n_pred, n_truth, tp) -> np.ndarray:
    """Vectorised per-entity F0.5 from counts."""
    n_pred, n_truth, tp = (np.asarray(x, float) for x in (n_pred, n_truth, tp))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, tp / n_pred, 0.0)
        r = np.where(n_truth > 0, tp / n_truth, 0.0)
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    return np.where(n_truth == 0, (n_pred == 0).astype(float), f)


def entity_scores(pred_map: dict, truth_map: dict) -> pd.Series:
    return pd.Series({k: f05(set(pred_map.get(k, ())), v) for k, v in truth_map.items()})


def breakdown(pred_map: dict, truth_map: dict, s1_country: dict = None) -> dict:
    s = entity_scores(pred_map, truth_map)
    n_true = pd.Series({k: len(v) for k, v in truth_map.items()})
    n_pred = pd.Series({k: len(pred_map.get(k, ())) for k in truth_map})
    out = {
        "macroF05": float(s.mean()),
        "n_entities": int(len(s)),
        "singleton_rate": float((n_true == 0).mean()),
        "singleton_acc": float(s[n_true == 0].mean()) if (n_true == 0).any() else None,
        "matched_F": float(s[n_true > 0].mean()) if (n_true > 0).any() else None,
        "F_1match": float(s[n_true == 1].mean()) if (n_true == 1).any() else None,
        "F_multi": float(s[n_true > 1].mean()) if (n_true > 1).any() else None,
        "n_pred_matches": int(n_pred.sum()),
        "n_pred_singletons": int((n_pred == 0).sum()),
    }
    tp = sum(len(set(pred_map.get(k, ())) & v) for k, v in truth_map.items())
    out["pair_P"] = tp / max(int(n_pred.sum()), 1)
    out["pair_R"] = tp / max(int(n_true.sum()), 1)
    if s1_country:
        c = pd.Series(s1_country).reindex(s.index)
        for cc, v in s.groupby(c):
            out[f"F05_{cc}"] = float(v.mean())
    return out


def paired_delta(scores_a: pd.Series, scores_b: pd.Series) -> tuple:
    """Mean per-entity difference (a - b) and its standard error on shared entities."""
    d = (scores_a - scores_b).dropna()
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(max(len(d), 1)))


def blocking_metrics(cand_map: dict, truth_map: dict, n_pool: int, union: pd.DataFrame = None,
                     id_of=None) -> dict:
    """Recall ceiling (pair + entity/oracle macro F0.5) and candidate-set size stats."""
    sizes = np.array([len(cand_map.get(k, ())) for k in truth_map])
    n_gt = sum(len(v) for v in truth_map.values())
    hit = sum(len(set(cand_map.get(k, ())) & v) for k, v in truth_map.items())
    oracle = {k: set(cand_map.get(k, ())) & v for k, v in truth_map.items()}
    out = {
        "block_pair_recall": hit / max(n_gt, 1),
        "entity_ceiling": float(entity_scores(oracle, truth_map).mean()),
        "cand_mean": float(sizes.mean()), "cand_median": float(np.median(sizes)),
        "cand_p95": float(np.percentile(sizes, 95)), "cand_max": int(sizes.max(initial=0)),
        "pct_zero_cand": float((sizes == 0).mean()),
        "reduction_ratio": 1 - sizes.sum() / max(len(truth_map) * n_pool, 1),
    }
    if union is not None and id_of is not None:
        # recall@k of the pre-prune union ordered by prune_score
        pos = set((s, c) for s, v in truth_map.items() for c in v)
        u = union.assign(s1=id_of[union.qi.values], cand=id_of[union.pj.values])
        u["is_pos"] = [(a, b) in pos for a, b in zip(u.s1, u.cand)]
        for k in (1, 2, 3, 5, 10, 20, 50):
            out[f"union_recall@{k}"] = float(u.is_pos[u.prune_rank <= k].sum() / max(n_gt, 1))
        out["union_recall_all"] = float(u.is_pos.sum() / max(n_gt, 1))
        out["union_cand_mean"] = float(len(u) / max(len(truth_map), 1))
    return out


def reliability(p: np.ndarray, y: np.ndarray, bins=10) -> dict:
    """Expected calibration error + per-bin table (the decision layer needs calibrated p)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append((float(edges[b]), int(m.sum()), float(p[m].mean()), float(y[m].mean())))
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return {"ece": float(ece), "bins": rows}
