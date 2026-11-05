"""K1/K4: TDRL in the hard settings and in a learning-rate sensitivity check, on the same data as CPT-2.
TDRL is trained through tdrl_runner.run (full series, no held-out 20%, 200 epochs, batch 64, AdamW 5e-4, torch seed = data
seed; last-epoch model primary, oracle_* columns secondary).
Stage B columns: mcc_jd, mcc_ica, lever_design (lever implied by the design), MCC_design [primary], B [oracle upper bound].
Output: outputs/tdrl_hard_settings{tag}_s{seed}.csv"""
import os, sys, argparse, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))

import tdrl_runner as TD                                             # noqa: E402  (stubs + path set-up + run)
from stage_b_scoring import design_lever                        # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--E", type=int, default=4)
ap.add_argument("--varmode", default="scale"); ap.add_argument("--innov", default="gauss"); ap.add_argument("--graph", default="fixed")
ap.add_argument("--k", type=int, default=4); ap.add_argument("--D", type=int, default=12); ap.add_argument("--T", type=int, default=6000)
ap.add_argument("--obs_noise", type=float, default=0.05); ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--lr", type=float, default=5e-4, help="(K4) learning-rate sensitivity")
ap.add_argument("--savez", default=None, help="file prefix for saved latents (outputs/z/), for later re-scoring")
ap.add_argument("--quick", action="store_true"); ap.add_argument("--tag", default=""); a = ap.parse_args()
T_ = 800 if a.quick else a.T
row = TD.run(seed=a.seed, E=a.E, T=T_, k=a.k, D=a.D, varmode=a.varmode, innov=a.innov, epochs=2 if a.quick else a.epochs,
             lr=a.lr, eval_every=1 if a.quick else 10, graph=a.graph, spurious=False, obs_noise=a.obs_noise, savez=a.savez)
lev = design_lever(a.varmode, a.innov)
jd, ica = row["mcc_jd"], row["mcc_ica"]
row.update(method="TDRL", version="k1v2", lever_design=lev, MCC_design=jd if lev == "JD" else ica,
           B=round(float(np.nanmax([jd, ica])), 4), oracle_MCC_design=row.get("oracle_mcc_jd" if lev == "JD" else "oracle_mcc_ica"),
           time=row["sec"])
print(row, flush=True)
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
pd.DataFrame([row]).to_csv(os.path.join(HERE, "outputs", f"tdrl_hard_settings{a.tag}_s{a.seed}.csv"), index=False)
