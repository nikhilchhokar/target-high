"""LightGBM pair matcher (MIT license) with GroupKFold-by-S1 out-of-fold predictions."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

NON_FEATURES = {"qi", "pj", "s1", "cand", "y", "p"}


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
    oof = np.zeros(len(X))
    iters, imp = [], pd.Series(0.0, index=X.columns)
    for seed in seeds:
        gkf = GroupKFold(n_splits=mcfg["cv"]["folds"], shuffle=True, random_state=seed)
        for tr, va in gkf.split(X, y, groups):
            m = _clf(mcfg["params"], seed)
            m.fit(X.iloc[tr], y[tr], eval_X=(X.iloc[va],), eval_y=(y[va],),
                  callbacks=[lgb.early_stopping(mcfg.get("early_stopping", 100), verbose=False)])
            oof[va] += m.predict_proba(X.iloc[va])[:, 1] / len(seeds)
            iters.append(int(m.best_iteration_ or mcfg["params"].get("n_estimators", 100)))
            imp += pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
    return oof, iters, (imp / imp.sum()).sort_values(ascending=False)


def train_full(X, y, mcfg: dict, n_iter: int) -> list:
    """Fit one model per seed on all training pairs with the CV-chosen iteration count."""
    return [_clf(mcfg["params"], s, n_estimators=n_iter).fit(X, y) for s in mcfg.get("seeds", [42])]


def predict(models, X) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
