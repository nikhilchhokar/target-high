"""Diagnostic: which true pairs does blocking miss, and why? (one country, small query sample)"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from ber import pipeline, prep, retrieve, io
bc, nq = sys.argv[1], int(sys.argv[2])
cfg = pipeline.load_config("configs/base.yaml")
cfg["blocking"]["retrievers"] = [dict(r, k=50) for r in cfg["blocking"]["retrievers"]]
s1c = prep.load(cfg, "train", [1], pipeline.Q_COLS, bc=bc)
pool = prep.load(cfg, "train", [2, 3], pipeline.Q_COLS + ["business_name", "business_address"], bc=bc)
idx = retrieve.CountryIndex(pool, s1c, cfg["blocking"], 7)
q = s1c.sample(nq, random_state=1).reset_index(drop=True)
u, _ = idx.query(q)
truth = io.load_ground_truth("data/train/train_ground_truth.tsv", q.entity_id.tolist())
pid = {e: i for i, e in enumerate(idx.pool.entity_id)}
qi_of = {e: i for i, e in enumerate(q.entity_id)}
got = set(zip(u.qi, u.pj))
rows = []
for s, v in truth.items():
    for c in v:
        if c in pid:
            rows.append((qi_of[s], pid[c], (qi_of[s], pid[c]) in got))
        else:
            rows.append((qi_of[s], -1, False))
R = pd.DataFrame(rows, columns=["qi", "pj", "found"])
print(bc, "true pairs", len(R), "found by union(k=50)", R.found.mean().round(4), "pool id missing (other country?)", (R.pj < 0).mean().round(4))
M = R[~R.found & (R.pj >= 0)]
qi, pj = M.qi.values, M.pj.values
M = M.assign(sim_name=retrieve._rowwise_dot(idx.vecs["name"].transform(retrieve.DOCS["name"](q)), idx.B["name"], qi, pj),
             sim_addr=retrieve._rowwise_dot(idx.vecs["addr"].transform(retrieve.DOCS["addr"](q)), idx.B["addr"], qi, pj))
print("missed pairs: sim_name quantiles", M.sim_name.quantile([.25, .5, .75]).round(3).tolist(), " sim_addr", M.sim_addr.quantile([.25, .5, .75]).round(3).tolist())
print("share missed with both sims < 0.1:", ((M.sim_name < .1) & (M.sim_addr < .1)).mean().round(3))
raw = s1c.set_index("entity_id")
for r in M.sample(min(25, len(M)), random_state=2).itertuples():
    a = q.iloc[r.qi]; b = idx.pool.iloc[r.pj]
    print(f"\n[{r.sim_name:.2f}/{r.sim_addr:.2f}] S1: {a.name_core} | {a.addr_core} #{a.house_nums}\n            {b.entity_id[:2]}: {b.business_name} | {b.business_address}\n            norm: {b.name_core} | {b.addr_core} #{b.house_nums}")
