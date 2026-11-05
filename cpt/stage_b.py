"""Stage-B procedures applied to a Stage-A estimate z_hat of the latent state (an affine copy of z up to estimation error):
  (0) residuals of a regime-wise linear autoregression (or of a nonlinear autoregression, or of a regression on a window of
      past values)
  (1) B1: joint diagonalization of the regime-wise residual covariances (Cardoso-Souloumiac)
  (2) B2: FastICA
  (3) B3: pattern fit (find Lambda such that the transition coefficients of Lambda^-1 z_hat lie in the given zero pattern)
Used as a library by the pipeline and the estimator-free validation. Command line: input outputs/z/<tag>_s<seed>.npz
(zt, zh, e, A), output outputs/stage_b.csv.
"""
import os, sys, glob, argparse, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))

from simulation import mcc, rotation_diagnostics, _joint_diag
from scipy.optimize import minimize
import pandas as pd


def ar_residuals(zh, e, E, mode="linear", lags=1, hidden=64, seed=0):
    """Regime-wise AR residuals. mode: linear | nonlinear (MLP) | multilag"""
    n, k = zh.shape
    if mode == "multilag":
        X0 = np.column_stack([zh[lags - 1 - t: n - 1 - t] for t in range(lags)])
        Y = zh[lags:]
        ee = e[lags:]
    else:
        X0, Y, ee = zh[:-1], zh[1:], e[1:]
    res = np.zeros_like(Y)
    if mode == "nonlinear":
        from sklearn.neural_network import MLPRegressor
        for r in range(E):
            idx = np.where(ee == r)[0]
            if len(idx) < 20 * k:
                continue
            m = MLPRegressor(hidden_layer_sizes=(hidden,), max_iter=600, random_state=seed,
                             early_stopping=True, n_iter_no_change=15).fit(X0[idx], Y[idx])
            res[idx] = Y[idx] - m.predict(X0[idx])
    else:
        for r in range(E):
            idx = np.where(ee == r)[0]
            if len(idx) < 5 * k:
                continue
            Xp = np.column_stack([np.ones(len(idx)), X0[idx]])
            B, *_ = np.linalg.lstsq(Xp, Y[idx], rcond=None)
            res[idx] = Y[idx] - Xp @ B
    return res, ee


def step_b1_jd(zh, res, ee, E):
    k = zh.shape[1]
    covs = [np.cov(res[ee == r].T) for r in range(E) if (ee == r).sum() >= 5 * k]
    if len(covs) < 2:
        return None
    Cp = np.cov(res.T); w, U = np.linalg.eigh(Cp)
    Wh = U @ np.diag(1 / np.sqrt(np.maximum(w, 1e-10))) @ U.T
    V = _joint_diag([Wh @ c @ Wh for c in covs])
    return zh @ Wh @ V


def step_b2_ica(zh, res, seed=0):
    from sklearn.decomposition import FastICA
    k = zh.shape[1]
    try:
        ica = FastICA(n_components=k, whiten="unit-variance", max_iter=3000,
                      random_state=seed).fit(res)
        return zh @ ica.components_.T
    except Exception:
        return None


def step_b2_ica_regime(zh, res, ee, seed=0, min_n=200):
    """B2 lever within a single regime (one regime suffices for non-Gaussian identification). If scales differ across
    regimes, the pooled residuals are a scale mixture and the components become dependent through the common regime variable,
    which violates the independence assumption of ICA. The residuals of the regime with the most observations are used.
    res, ee: output of ar_residuals (row aligned)."""
    from sklearn.decomposition import FastICA
    k = zh.shape[1]; r, counts = np.unique(ee, return_counts=True); r0 = r[np.argmax(counts)]
    m = ee == r0
    if m.sum() < min_n:
        return None
    try:
        ica = FastICA(n_components=k, whiten="unit-variance", max_iter=3000, random_state=seed).fit(res[m])
        return zh @ ica.components_.T
    except Exception:
        return None


def step_b3_pattern(zh, e, E, masks, restarts=4, seed=0, nonlin=False):
    """Search Lambda such that the regime-wise AR coefficients of Lambda^-1 zh lie in the given pattern (search over
    R^{k x k}; superseded by step_b3_pattern_v2)."""
    n, k = zh.shape
    zs = (zh - zh.mean(0)) / (zh.std(0) + 1e-9)

    def obj(v):
        L = v.reshape(k, k)
        d = np.linalg.det(L)
        if abs(d) < 1e-6:
            return 1e6
        u = zs @ np.linalg.inv(L).T
        tot = 0.0
        for r in range(E):
            idx = np.where(e[1:] == r)[0]
            if len(idx) < 5 * k:
                continue
            X = np.column_stack([np.ones(len(idx)), u[idx]])
            B, *_ = np.linalg.lstsq(X, u[idx + 1], rcond=None)
            A = B[1:].T
            tot += ((A * (1 - masks[r])) ** 2).sum() / ((A ** 2).sum() + 1e-9)
        return tot / E + 0.01 * ((np.linalg.svd(L, compute_uv=False) - 1) ** 2).sum()

    best = None
    for t in range(restarts):
        rng = np.random.default_rng(seed * 100 + t)
        x0 = (np.eye(k) + 0.15 * rng.standard_normal((k, k))).ravel()
        r = minimize(obj, x0, method="Powell",
                     options=dict(maxiter=30000, xtol=1e-6, ftol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return zs @ np.linalg.inv(best.x.reshape(k, k)).T, float(best.fun)


def step_b3_pattern_v2(zh, e, E, masks, restarts=12, seed=0):
    """Constructive B3 step.
    After whitening the residuals, the remaining freedom is ORTHOGONAL only. The search space is therefore
    so(k) with k(k-1)/2 parameters (6 for k=4, instead of k^2 = 16 for a search over R^{k x k}) and compact; a Powell search
    over R^{k x k} (step_b3_pattern) gets stuck in local minima.
    Parametrization: O = expm(skew(theta)); objective: off-pattern AR energy (scale normalized)."""
    from scipy.linalg import expm
    from scipy.optimize import minimize
    n, k = zh.shape
    idx_sk = [(i, j) for i in range(k) for j in range(i + 1, k)]

    def skew(th):
        X = np.zeros((k, k))
        for v, (i, j) in zip(th, idx_sk):
            X[i, j] = v; X[j, i] = -v
        return X

    # 1) whitening from the AR residuals (absorbs D and S)
    res, ee = ar_residuals(zh, e, E, mode="linear")
    C = np.cov(res.T)
    w, U = np.linalg.eigh(C)
    Wh = U @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-10))) @ U.T
    zw = (zh - zh.mean(0)) @ Wh

    # 2) off-pattern AR energy over orthogonal O
    Xs, Ys = {}, {}
    for r in range(E):
        ii = np.where(e[1:] == r)[0]
        if len(ii) >= 5 * k:
            Xs[r] = np.column_stack([np.ones(len(ii)), zw[ii]]); Ys[r] = zw[ii + 1]

    def obj(th):
        O = expm(skew(th))
        tot = 0.0
        for r, X in Xs.items():
            u0 = X.copy(); u0[:, 1:] = X[:, 1:] @ O
            B, *_ = np.linalg.lstsq(u0, Ys[r] @ O, rcond=None)
            A = B[1:].T
            tot += ((A * (1 - masks[r])) ** 2).sum() / ((A ** 2).sum() + 1e-9)
        return tot / max(len(Xs), 1)

    # ADAPTIVE RESTARTS: if dim g_S = 0 the solution is unique and the objective should be ~0; if the objective stays
    # high we are in a local minimum -> another round of restarts.
    best, bestf = None, np.inf
    rng = np.random.default_rng(seed)
    tol_ok = 0.02
    for rnd in range(3):
        starts = ([np.zeros(len(idx_sk))] if rnd == 0 else []) + \
                 [rng.uniform(-np.pi / 2, np.pi / 2, len(idx_sk)) for _ in range(restarts)]
        for th0 in starts:
            r_ = minimize(obj, th0, method="Powell", options=dict(maxiter=20000, xtol=1e-8, ftol=1e-10))
            if r_.fun < bestf:
                bestf, best = r_.fun, r_.x
        if bestf < tol_ok:
            break
    return zw @ expm(skew(best)), float(bestf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zdir", default=os.path.join(HERE, "outputs", "z"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "stage_b.csv"))
    ap.add_argument("--pre", default="linear,nonlinear", help="pre-processing modes (comma separated)")
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.zdir, "*.npz"))):
        d = np.load(f); zt, zh, e, A = d["zt"], d["zh"], d["e"], d["A"]
        E = int(e.max()) + 1
        masks = (np.abs(A) > 1e-9).astype(float)
        tag = os.path.basename(f)[:-4]
        r2a, maff = rotation_diagnostics(zt[200:], zh[200:])
        base = mcc(zt[200:], zh[200:])[0]
        for pre in a.pre.split(","):
            res, ee = ar_residuals(zh, e, E, mode=pre)
            out = dict(tag=tag, pre=pre, raw=round(base, 4), affine=round(maff, 4),
                       r2_affine=round(r2a, 4))
            z1 = step_b1_jd(zh, res, ee, E)
            out["b1_jd"] = round(mcc(zt[200:], z1[200:])[0], 4) if z1 is not None else np.nan
            z2 = step_b2_ica(zh, res)
            out["b2_ica"] = round(mcc(zt[200:], z2[200:])[0], 4) if z2 is not None else np.nan
            if pre == "linear":          # B3 does not depend on the pre-processing (fits its own AR)
                z3, fval = step_b3_pattern_v2(zh, e, E, masks)
                out["b3_pattern"] = round(mcc(zt[200:], z3[200:])[0], 4)
                out["b3_obj"] = round(fval, 5)
            rows.append(out); print(out, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    print("\nWritten:", a.out)
    if len(df):
        print(df.groupby("pre")[["raw", "affine", "b1_jd", "b2_ica"]].mean().round(3).to_string())
