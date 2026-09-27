"""LightGBM pair matcher (MIT license) with GroupKFold-by-S1 out-of-fold predictions."""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

NON_FEATURES = {"qi", "pj", "s1", "cand", "y", "p"}


RANK_SCALE_PREFIX = ("chain_", "sim_", "prune_score", "n_max_idf", "n_idf_overlap", "rev_n_s1", "ctx_n_",
                     "ctx_gap_sim", "ctx_gap_prune", "rev_gap_prune", "rev_gap_sim", "hop_best")


def rank_normalise(df: pd.DataFrame, cols, country) -> pd.DataFrame:
    """Per-country percentile of scale-dependent features, so an unseen country (France) is
    presented on the same scale as the training countries. In place on `cols`; returns df."""
    country = np.asarray(country)
    for c in cols:
        if c in df:
            df[c] = df[c].groupby(country).rank(pct=True).astype(np.float32).values
    return df


def rank_cols(fcols, mode):
    if mode == "all":
        return list(fcols)
    if mode == "scale":
        return [c for c in fcols if c.startswith(RANK_SCALE_PREFIX)]
    return []


def feature_cols(df: pd.DataFrame, drop=()) -> list:
    return [c for c in df.columns if c not in NON_FEATURES and c not in set(drop)]


def _clf(params, seed, n_estimators=None):
    p = dict(params)
    if n_estimators:
        p["n_estimators"] = n_estimators
    return lgb.LGBMClassifier(objective="binary", random_state=seed, verbose=-1, **p)


def train_oof(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, mcfg: dict):
    """Out-of-fold probabilities. Folds are grouped by S1 id so no entity leaks across folds."""
    seeds = mcfg.get("seeds", [42])
    cols = list(X.columns)
    X = X.to_numpy(np.float32)  # one compact copy; fancy-indexing numpy avoids pandas overhead
    oof = np.zeros(len(X))
    iters, imp = [], pd.Series(0.0, index=cols)
    codes, uniq = pd.factorize(pd.Series(groups))
    k = mcfg["cv"]["folds"]
    for seed in seeds:
        # group k-fold by S1 id with integer codes: every pair of one S1 entity lands in the
        # same fold (sklearn's GroupKFold(shuffle) is ~20 min per split on 2M string ids)
        fold_of_group = np.random.default_rng(seed).permutation(len(uniq)) % k
        fold_of_row = fold_of_group[codes]
        for fold in range(k):
            tr, va = np.where(fold_of_row != fold)[0], np.where(fold_of_row == fold)[0]
            t0 = time.time()
            m = _clf(mcfg["params"], seed)
            m.fit(X[tr], y[tr], eval_X=(X[va],), eval_y=(y[va],), feature_name=cols,
                  callbacks=[lgb.early_stopping(mcfg.get("early_stopping", 100), verbose=False)])
            oof[va] += m.predict_proba(X[va])[:, 1] / len(seeds)
            iters.append(int(m.best_iteration_ or mcfg["params"].get("n_estimators", 100)))
            imp += pd.Series(m.booster_.feature_importance("gain"), index=cols)
            print(f"   [model] seed {seed} fold {fold}: best_iter {iters[-1]}, {time.time() - t0:.0f}s", flush=True)
    return oof, iters, (imp / imp.sum()).sort_values(ascending=False)


def train_full(X, y, mcfg: dict, n_iter: int) -> list:
    """Fit one model per seed on all training pairs with the CV-chosen iteration count."""
    cols = list(X.columns)
    Xn = X.to_numpy(np.float32)
    return [_clf(mcfg["params"], s, n_estimators=n_iter).fit(Xn, y, feature_name=cols)
            for s in mcfg.get("seeds", [42])]


def predict(models, X) -> np.ndarray:
    Xn = X.to_numpy(np.float32) if hasattr(X, "to_numpy") else X
    return np.mean([m.predict_proba(Xn)[:, 1] for m in models], axis=0)
