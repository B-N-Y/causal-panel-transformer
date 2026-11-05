"""Random-rotation and PCA baselines for the raw MCC (Section 8.3 of the paper).

For each generator cell and seed, the true latent state z is transformed by Haar-distributed random orthogonal matrices,
which is exactly the freedom left when no lever is available (Theorem 5.2(c)), and the permutation-matched mean absolute
correlation (raw MCC) with z is recorded. The same is done for an i.i.d. Gaussian latent of the same dimension, and for the
first k principal components of the observations.
Outputs: one row per cell and seed (--out) and one row per random rotation (--draws_out), from which the 5th-95th
percentiles over all draws are computed.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ortho_group
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpt"))
from simulation import gen_panel, mcc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--draws", type=int, default=50, help="random orthogonal matrices per seed")
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--out", default=os.path.join("outputs", "random_rotation_baseline.csv"))
    ap.add_argument("--draws_out", default=None,
                    help="file for the per-draw values (default: random_rotation_draws.csv next to --out)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    draws_out = a.draws_out or os.path.join(os.path.dirname(a.out), "random_rotation_draws.csv")
    cells = [("none", "gauss"), ("scale", "gauss"), ("none", "t5"), ("both", "t5")]
    rows, draws = [], []
    for varmode, innov in cells:
        for s in range(a.seeds):
            x, z, e, meta = gen_panel(k=4, E=4, T=a.T, seed=s, nonlinear=True, spurious=False, innov=innov, varmode=varmode)
            k = z.shape[1]
            rng = np.random.default_rng(1000 + s)
            rot = [mcc(z, z @ ortho_group.rvs(k, random_state=rng).T)[0] for _ in range(a.draws)]
            iid = rng.standard_normal(z.shape)
            rot_iid = [mcc(iid, iid @ ortho_group.rvs(k, random_state=rng).T)[0] for _ in range(a.draws)]
            pca = mcc(z, PCA(n_components=k).fit_transform(x))[0]
            draws += [dict(varmode=varmode, innov=innov, seed=s, draw=i, rot=r_, rot_iid=q_)
                      for i, (r_, q_) in enumerate(zip(rot, rot_iid))]
            rows.append(dict(varmode=varmode, innov=innov, seed=s, T=a.T, draws=a.draws,
                             rot_mean=np.mean(rot), rot_max=np.max(rot), rot_iid_mean=np.mean(rot_iid), pca=pca))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.groupby(["varmode", "innov"])[["rot_mean", "rot_max", "rot_iid_mean", "pca"]].agg(["min", "max"]).round(3))
    print("overall: rotation mean per seed %.3f-%.3f, single draw max %.3f, iid %.3f, PCA %.3f-%.3f"
          % (df.rot_mean.min(), df.rot_mean.max(), df.rot_max.max(), df.rot_iid_mean.mean(), df.pca.min(), df.pca.max()))
    dd = pd.DataFrame(draws)
    dd.to_csv(draws_out, index=False)
    mean_shift = dd.varmode.isin(["mean", "both"])           # cells with regime-dependent mean shifts
    q = lambda v: "%.2f-%.2f" % tuple(np.percentile(v, [5, 95]))  # noqa: E731
    print("5th-95th percentiles over all draws: without mean shifts %s, with mean shifts %s, i.i.d. %s (mean %.3f)"
          % (q(dd.rot[~mean_shift]), q(dd.rot[mean_shift]), q(dd.rot_iid), dd.rot_iid.mean()))


if __name__ == "__main__":
    main()
