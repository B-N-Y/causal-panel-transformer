"""Dimension selection: rule v2 ('over-select, then prune') versus the elbow rule (c = 0.07), fresh seeds, same data.
Scenarios as in the end-to-end pipeline runs (true dimension: 4 without spurious columns, 5 with them).
Output: outputs/dimension_v2{tag}_s{seed}.csv"""
import os, sys, json, argparse, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))
import dimension as BY
from simulation import gen_panel
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=140); ap.add_argument("--E", type=int, default=4)
ap.add_argument("--varmode", default="scale"); ap.add_argument("--innov", default="gauss"); ap.add_argument("--nospurious", action="store_true")
ap.add_argument("--graph", default="fixed"); ap.add_argument("--kmax", type=int, default=8); ap.add_argument("--quick", action="store_true")
ap.add_argument("--tag", default=""); a = ap.parse_args()
T_ = 800 if a.quick else 6000
x, z, e, meta = gen_panel(seed=a.seed, E=a.E, T=T_, varmode=a.varmode, innov=a.innov, spurious=not a.nospurious, graph=a.graph)
kg = 4 if a.nospurious else 5
k2, i2 = BY.select_dimension_v2(x, e, kmax=a.kmax, fit_kw=dict(steps=30 if a.quick else 20000))
k1, i1 = BY.select_dimension(x, e, ks=range(2, 8) if not a.quick else (3, 4), seeds=(0, 1) if not a.quick else (0,), epochs=2 if a.quick else 200, c=0.07)
row = dict(seed=a.seed, E=a.E, varmode=a.varmode, innov=a.innov, spurious=not a.nospurious, graph=a.graph, k_true=kg,
           k_v2=k2, k_elbow=k1, v2_reproducible=i2["reproducible"], v2_canonical=json.dumps([round(v, 3) for v in i2["canonical"]]),
           v2_r2=json.dumps([round(v, 3) for v in i2["r2"]]))
print(row, flush=True)
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
pd.DataFrame([row]).to_csv(os.path.join(HERE, "outputs", f"dimension_v2{a.tag}_s{a.seed}.csv"), index=False)
