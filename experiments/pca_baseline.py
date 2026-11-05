"""Linear baseline: Stage A = principal components of the observations (k components, whitened), Stage B = the lever fixed by the
design of the cell in Table 3 (simulation.resolve_rotation: regime-wise AR residuals -> JD or pooled ICA; same choice as
stage_b_scoring.design_lever). Scoring as in experiments/evaluate_cpt.py (raw MCC, affine R^2, affine MCC, mcc_jd, mcc_ica, rho_min;
the first 200 steps are dropped); solved = MCC_design >= 0.90.
Data: the main cells of Table 3, same call as in run R10, so seeds 0-9 give the data of R10 and the competitors, and seeds 200-209
the data of R10F.
Usage: python experiments/pca_baseline.py --cell E4_scale_g --seed0 0 --n 10 [--tag _X] [--quick]
Cells: E2_scale_g, E4_scale_g, E4_none_g, E4_none_t5, E4_scale_g_rand. CPU; a few seconds per dataset.
Output: outputs/pca_baseline{tag}_{cell}_s{seed0}-{last seed}.csv
"""
import os, sys, time, argparse, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "cpt"))
from simulation import gen_panel, mcc, rotation_diagnostics, resolve_rotation
from stage_b_scoring import design_lever

CELLS = {"E2_scale_g": (2, "scale", "gauss", "fixed"), "E4_scale_g": (4, "scale", "gauss", "fixed"),
         "E4_none_g": (4, "none", "gauss", "fixed"), "E4_none_t5": (4, "none", "t5", "fixed"),
         "E4_scale_g_rand": (4, "scale", "gauss", "random")}


def rho_min(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    def isq(C):
        w, V = np.linalg.eigh(C); return V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    return float(np.linalg.svd(isq(np.cov(a.T)) @ (a.T @ b / len(a)) @ isq(np.cov(b.T)), compute_uv=False).min())


def pca_whitened(x, k):
    xc = x - x.mean(0); w, V = np.linalg.eigh(np.cov(xc.T)); o = np.argsort(w)[::-1][:k]
    return xc @ V[:, o] / np.sqrt(w[o]), float(w[o].sum() / w.sum())


ap = argparse.ArgumentParser()
ap.add_argument("--cell", required=True, choices=list(CELLS)); ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--n", type=int, default=10)
ap.add_argument("--k", type=int, default=4); ap.add_argument("--D", type=int, default=12); ap.add_argument("--T", type=int, default=6000)
ap.add_argument("--obs_noise", type=float, default=0.05); ap.add_argument("--quick", action="store_true"); ap.add_argument("--tag", default="")
a = ap.parse_args()
E, vm, inn, gr = CELLS[a.cell]; T_ = 1500 if a.quick else a.T; lev = design_lever(vm, inn)
rows = []
for s in range(a.seed0, a.seed0 + a.n):
    t0 = time.time()
    x, z, e, meta = gen_panel(seed=s, E=E, T=T_, varmode=vm, innov=inn, spurious=False, graph=gr, k=a.k, D=a.D, obs_noise=a.obs_noise)
    zh, share = pca_whitened(x.astype(np.float64), a.k)
    m, _ = mcc(z[200:], zh[200:]); r2, ma = rotation_diagnostics(z[200:], zh[200:]); rr = resolve_rotation(z[200:], zh[200:], e[200:], E)
    mt = rr["mcc_jd"] if lev == "JD" else rr["mcc_ica"]
    rows.append(dict(method="PCA (whitened)", cell=a.cell, seed=s, E=E, varmode=vm, innov=inn, graph=gr, k=a.k, D=a.D, T=T_,
                     obs_noise=a.obs_noise, spurious=False, explained_variance=round(share, 4), MCC=round(float(m), 4), r2_affine=round(r2, 4),
                     mcc_affine=round(ma, 4), mcc_jd=round(rr["mcc_jd"], 4), mcc_ica=round(rr["mcc_ica"], 4), lever_design=lev,
                     MCC_design=round(float(mt), 4), solved=bool(mt >= 0.90), rho_min=round(rho_min(z[200:], zh[200:]), 4),
                     time=round(time.time() - t0, 2)))
    print(rows[-1], flush=True)
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
out = os.path.join(HERE, "outputs", f"pca_baseline{a.tag}_{a.cell}_s{a.seed0}-{a.seed0 + a.n - 1}.csv")
pd.DataFrame(rows).to_csv(out, index=False); print("WROTE", out)
