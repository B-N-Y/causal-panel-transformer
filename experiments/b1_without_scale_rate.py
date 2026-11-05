"""Rate at which lever B1 (joint diagonalization) meets the solved criterion WITHOUT regime-dependent innovation scales
(Section 8.2 and Appendix D). Same setting as Table 2 (affine copy with condition number 5, T=4000, k=4, E=4), on seeds not
used in Table 2, in the four cells without scale variation: with or without mean shifts, Gaussian or Student-t5 innovations.
A run is solved if the distance to the nearest permutation is below 0.25 (Appendix D). Runs on a CPU in a few minutes.
Usage: python experiments/b1_without_scale_rate.py --seeds 10 4010 --out results/B1_NO_SCALE/b1_without_scale.csv"""
import os, sys, argparse, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))
import simulation
from stage_b import ar_residuals, step_b1_jd
from scipy.optimize import linear_sum_assignment

K, E, T, KAPPA, NOISE = 4, 4, 4000, 5.0, 0.05


def affine_copy(z, seed):
    """Bounded affine copy, identical to estimator_free_validation.py (copy type 'bounded')."""
    rng = np.random.default_rng(1000 + seed); G = rng.standard_normal((K, K))
    U, _, Vt = np.linalg.svd(G); L = U @ np.diag(np.geomspace(1.0, KAPPA, K)) @ Vt
    rng = np.random.default_rng(1000 + seed); rng.standard_normal((K, K))
    return z @ L.T + NOISE * rng.standard_normal(z.shape)


def perm_distance(z, zr):
    X = np.column_stack([np.ones(len(z)), z]); M = np.linalg.lstsq(X, zr, rcond=None)[0][1:].T
    Mn = np.abs(M) / np.linalg.norm(M, axis=1, keepdims=True); r, c = linear_sum_assignment(-Mn)
    Pm = np.zeros_like(Mn); Pm[r, c] = 1.0; return float(np.linalg.norm(Mn - Pm))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs=2, default=[10, 4010])
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "B1_NO_SCALE", "b1_without_scale.csv")); a = ap.parse_args()
    rows = []
    for vm in ["none", "mean"]:
        for inn in ["gauss", "t5"]:
            for s in range(*a.seeds):
                x, z, e, meta = simulation.gen_panel(k=K, E=E, T=T, seed=s, nonlinear=True, innov=inn, varmode=vm)
                zh = affine_copy(z, s); res, ee = ar_residuals(zh, e, E, "linear"); zj = step_b1_jd(zh, res, ee, E)
                rows.append(dict(varmode=vm, innov=inn, seed=s, T=T, K=K, E=E, kappa=KAPPA, noise=NOISE,
                                 perm_distance=perm_distance(z, zj)))
    df = pd.DataFrame(rows); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); df.to_csv(a.out, index=False)
    df["solved"] = df.perm_distance < 0.25
    print(df.groupby(["varmode", "innov"]).agg(n=("solved", "size"), solved=("solved", "sum"), median_distance=("perm_distance", "median")))
