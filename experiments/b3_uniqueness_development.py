"""Uniqueness check of B3 recomputed on the development data (seeds 600-609 and 650-659), from the Stage-A latents saved by
experiments/evaluate_pipeline.py --save_latents (outputs/z/pipeline_stage_a{tag}_s{seed}.npz); same scan and rule as that script.
Usage: python experiments/b3_uniqueness_development.py --generator bs --E 2 --seed 650 --tag _b3bs
Output: outputs/b3_uniqueness_development{tag}_s{seed}.csv
"""
import os, sys, argparse, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "cpt"))
import b3_pattern as B3P
from simulation import gen_panel

ap = argparse.ArgumentParser()
ap.add_argument("--generator", default="panel", choices=["panel", "bs"]); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--E", type=int, default=4); ap.add_argument("--b3_scan", type=int, default=24)
a = ap.parse_args()
d = np.load(os.path.join(HERE, "outputs", "z", f"pipeline_stage_a{a.tag}_s{a.seed}.npz")); zA, e, w = d["zA"], d["e"], d["w"]
if a.generator == "bs":
    from simulation_bs import gen_bs
    _, _, _, meta = gen_bs(seed=a.seed, T=6000); base = meta["masks"]
else:
    _, _, _, meta = gen_panel(seed=a.seed, E=a.E, T=6000, varmode="none", innov="gauss", spurious=False)
    base = (np.abs(np.stack(meta["A_list"])) > 1e-9).astype(float)
masks = B3P.pattern_versions(base, a.seed)["correct"]
row = dict(generator=a.generator, seed=a.seed, tag=a.tag)
if zA.shape[1] == w.shape[1]:
    sols = B3P.b3_scan(zA, e, int(e.max()) + 1, masks, n_starts=a.b3_scan, seed=a.seed)
    n_s, n_y = B3P.b3_uniqueness(sols)
    row.update(n_matching=len(sols), b3uniq_n_classes=n_s, b3uniq_n_near=n_y, decision="NOT_IDENTIFIABLE" if n_s != 1 else "B3")
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
pd.DataFrame([row]).to_csv(os.path.join(HERE, "outputs", f"b3_uniqueness_development{a.tag}_s{a.seed}.csv"), index=False)
print(json.dumps(row))
