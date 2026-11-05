"""Estimator-free validation of the identifiability results (V1-V6). No estimator is trained: Stage A is replaced by an
affine copy of the true latent, z_hat = z Lambda' + noise, and the Stage B steps are applied to it.
Diagnostics reported in every row:
  (i)   the affine copy has a bounded condition number: Lambda = U diag(s) V^T, s = geomspace(1, KAPPA); for comparison, an
        unconstrained Gaussian matrix (condition number not bounded) can be requested as copy type 'v1' (column 'copy' =
        'bounded' / 'v1');
  (ii)  condition number, rho_min (smallest canonical correlation between z and z_hat), affine R^2 and affine-fit MCC;
  (iii) Stage B margins: B1 = min_{i<j} (1 - |cos(p_i, p_j)|), where p_j is the cross-regime variance profile of component j
        (0: proportional profiles, not identifiable); B2 = innovation excess kurtosis (Gaussian 0, 6 for t5); B3 = smallest
        singular value of the operator L(A~) on so(k) (Lemma 5.6; 0: not locally identifiable);
  (iv)  distance of the resolved representation to a signed permutation: the affine regression matrix of z_resolved on z is
        row-normalised, and the Frobenius distance between |M| and the nearest permutation matrix is reported (0: exact resolution);
  (v)   all settings (T, K, E, seed, restarts, noise, KAPPA) as columns.
Output: outputs/estimator_free_validation/V*.csv and validation.log. Usage: python experiments/estimator_free_validation.py
(environment variables: VSEEDS, VT, VWHICH, VKAPPA, VCOPY). B2 is evaluated within a single regime (step_b2_ica_regime)."""
import os, sys, time, itertools, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))
import simulation
from simulation import mcc, rotation_diagnostics, resolve_rotation
from stage_b import step_b3_pattern_v2 as step_b3_pattern, ar_residuals, step_b1_jd, step_b2_ica_regime

OUT = os.path.join(HERE, "outputs", "estimator_free_validation"); os.makedirs(OUT, exist_ok=True)
SEEDS = int(os.environ.get("VSEEDS", 10)); T = int(os.environ.get("VT", 4000)); K = 4; E = 4
KAPPA = float(os.environ.get("VKAPPA", 5.0)); NOISE = 0.05; RESTARTS = 12
COPIES = tuple(os.environ.get("VCOPY", "bounded").split("-"))     # copy types; "bounded-v1" also runs the unconstrained copy
log = open(os.path.join(OUT, "validation.log"), "a")


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()


def lam(seed, copy, k=K):
    rng = np.random.default_rng(1000 + seed); G = rng.standard_normal((k, k))
    if copy == "v1":
        return G                                                     # unconstrained Gaussian matrix
    U, _, Vt = np.linalg.svd(G); return U @ np.diag(np.geomspace(1.0, KAPPA, k)) @ Vt


def affine_copy(z, seed, copy):
    L = lam(seed, copy, z.shape[1]); rng = np.random.default_rng(1000 + seed); rng.standard_normal((z.shape[1], z.shape[1]))
    return z @ L.T + NOISE * rng.standard_normal(z.shape), float(np.linalg.cond(L))


def rho_min(z, zh):
    zc, hc = z - z.mean(0), zh - zh.mean(0); n = len(z)
    Szz, Shh, Szh = zc.T @ zc / n, hc.T @ hc / n, zc.T @ hc / n
    iz = np.linalg.inv(np.linalg.cholesky(Szz)); ih = np.linalg.inv(np.linalg.cholesky(Shh))
    return float(np.linalg.svd(iz @ Szh @ ih.T, compute_uv=False).min())


def b1_margin(S):
    """S: (E, k) innovation scales; profile p_j = S[:, j]^2."""
    Pm = np.asarray(S) ** 2; Pm = Pm / np.linalg.norm(Pm, axis=0, keepdims=True)
    return float(min(1 - abs(Pm[:, i] @ Pm[:, j]) for i, j in itertools.combinations(range(Pm.shape[1]), 2)))


def b3_margin(A_list, masks):
    k = A_list[0].shape[0]; basis = []
    for i, j in itertools.combinations(range(k), 2):
        X = np.zeros((k, k)); X[i, j], X[j, i] = 1.0, -1.0; basis.append(X / np.sqrt(2))
    cols = []
    for X in basis:
        v = [(X @ A - A @ X)[m == 0] for A, m in zip(A_list, masks)]; cols.append(np.concatenate(v))
    return float(np.linalg.svd(np.column_stack(cols), compute_uv=False).min())


def perm_distance(z, zr):
    X = np.column_stack([np.ones(len(z)), z]); M = np.linalg.lstsq(X, zr, rcond=None)[0][1:].T   # zr ~ M z
    Mn = np.abs(M) / np.linalg.norm(M, axis=1, keepdims=True); r, c = linear_sum_assignment(-Mn)
    Pm = np.zeros_like(Mn); Pm[r, c] = 1.0; return float(np.linalg.norm(Mn - Pm))


def diagnostics(z, zh, cond, copy, s):
    r2, ma = rotation_diagnostics(z, zh)
    return dict(seed=s, copy=copy, condition_number=cond, rho_min=rho_min(z, zh), r2_affine=r2, mcc_affine=ma, raw=mcc(z, zh)[0],
                T=len(z), K=z.shape[1], E=E, kappa=KAPPA if copy == "bounded" else np.nan, noise=NOISE, restarts=RESTARTS)


def V12():
    rows = []
    for vm in ["none", "scale", "mean", "both"]:
        for inn in ["gauss", "t5"]:
            for s in range(SEEDS):
                x, z, e, meta = simulation.gen_panel(k=K, E=E, T=T, seed=s, nonlinear=True, innov=inn, varmode=vm)
                masks = (np.abs(np.stack(meta["A_list"])) > 1e-9).astype(float)
                margins = dict(b1_margin=b1_margin(meta["noise_scale"]), b2_kurtosis=0.0 if inn == "gauss" else 6.0, b3_margin=b3_margin(meta["A_list"], masks))
                for copy in COPIES:
                    zh, cond = affine_copy(z, s, copy)
                    res, ee = ar_residuals(zh, e, E, "linear"); zj = step_b1_jd(zh, res, ee, E); zi = step_b2_ica_regime(zh, res, ee, seed=s)
                    z3, f = step_b3_pattern(zh, e, E, masks, restarts=RESTARTS, seed=s)
                    rows.append(dict(test="V1V2", mode=vm, innov=inn, **diagnostics(z, zh, cond, copy, s), **margins,
                                     B1_jd=mcc(z, zj)[0], B2_ica=mcc(z, zi)[0], B3_pattern=mcc(z, z3)[0],
                                     B1_perm_distance=perm_distance(z, zj), B2_perm_distance=perm_distance(z, zi), B3_perm_distance=perm_distance(z, z3)))
                    P({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in rows[-1].items()})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "V1V2_levers.csv"), index=False); return df


def simulate_linear(A_list, S_list, b_list, e, seed):
    rng = np.random.default_rng(seed); Tn = len(e); z = np.zeros((Tn, K))
    for t in range(1, Tn):
        z[t] = A_list[e[t]] @ z[t - 1] + b_list[e[t]] + S_list[e[t]] * rng.standard_normal(K)
    return z


def V3():
    rows = []
    for s in range(SEEDS):
        x, z0, e, meta = simulation.gen_panel(k=K, E=E, T=T, seed=s, nonlinear=True, innov="gauss", varmode="scale")
        A_list = meta["A_list"]; rng = np.random.default_rng(s); base = np.array([0.3] * K); S_list = []
        for r in range(E):
            f = rng.uniform(0.5, 2.0, K); f[1] = f[0]; S_list.append(base * f)
        z = simulate_linear(A_list, S_list, [np.zeros(K)] * E, e, s)
        for copy in COPIES:
            zh, cond = affine_copy(z, s, copy); res, ee = ar_residuals(zh, e, E, "linear"); zj = step_b1_jd(zh, res, ee, E)
            Cm = np.array([[abs(np.corrcoef(z[:, a], zj[:, b])[0, 1]) for b in range(K)] for a in range(K)])
            r_, c_ = linear_sum_assignment(-Cm); per = Cm[r_, c_]
            Sb = np.array(S_list); margin_23 = b1_margin(Sb[:, [0, 2, 3]])        # separation margin outside the block {0,1}
            rows.append(dict(test="V3_block", **diagnostics(z, zh, cond, copy, s), b1_margin_all=b1_margin(Sb), b1_margin_off_block=margin_23,
                             jd_mcc=per.mean(), block01_corr=per[[0, 1]].mean(), split23_corr=per[[2, 3]].mean()))
            P({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in rows[-1].items()})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "V3_block.csv"), index=False); return df


def support_metrics(A_hat_list, A_true_list, thr=0.15):
    St = [np.abs(A) > 1e-9 for A in A_true_list]; Sh = [np.abs(A) > thr for A in A_hat_list]
    Ft = np.logical_and.reduce(St); Fh = np.logical_and.reduce(Sh); Ut = np.logical_or.reduce(St); Uh = np.logical_or.reduce(Sh)
    Vt = Ut & ~Ft; Vh = Uh & ~Fh
    f1 = lambda a, b: (a & b).sum() / max((a & b).sum() + 0.5 * ((~a & b).sum() + (a & ~b).sum()), 1e-9)
    return f1(Ft, Fh), f1(Vt, Vh), bool((Vt == Vh).all())


def V4():
    rows = []
    for vm, inn in [("scale", "gauss"), ("none", "t5"), ("both", "t5")]:
        for s in range(SEEDS):
            x, z, e, meta = simulation.gen_panel(k=K, E=E, T=T, seed=s, nonlinear=True, innov=inn, varmode=vm)
            for copy in COPIES:
                zh, cond = affine_copy(z, s, copy); res, ee = ar_residuals(zh, e, E, "linear")
                zr = step_b1_jd(zh, res, ee, E) if vm != "none" else step_b2_ica_regime(zh, res, ee, seed=s)

                def fit_A(zz):
                    out = []
                    for r in range(E):
                        idx = np.where(e[1:] == r)[0]; X = np.column_stack([np.ones(len(idx)), zz[idx]])
                        out.append(np.linalg.lstsq(X, zz[idx + 1], rcond=None)[0][1:].T)
                    return out
                m, perm = mcc(z, zr); zr_al = zr[:, perm]
                sg = np.sign([np.corrcoef(z[:, j], zr_al[:, j])[0, 1] for j in range(K)]); zr_al = zr_al * sg / zr_al.std(0) * z.std(0)
                fF, fV, vexact = support_metrics(fit_A(zr_al), meta["A_list"]); fF0, fV0, _ = support_metrics(fit_A(zh), meta["A_list"])
                rows.append(dict(test="V4_edge", mode=vm, innov=inn, **diagnostics(z, zh, cond, copy, s), b1_margin=b1_margin(meta["noise_scale"]),
                                 mcc_resolved=m, perm_distance=perm_distance(z, zr), F_f1=fF, V_f1=fV, V_exact=vexact, F_f1_raw=fF0, V_f1_raw=fV0))
                P({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in rows[-1].items()})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "V4_edge_types.csv"), index=False); return df


def V5():
    from sklearn.decomposition import FastICA
    rows = []
    for s in range(SEEDS):
        x, z, e, meta = simulation.gen_panel(k=K, E=E, T=T, seed=s, nonlinear=True, innov="t5", varmode="none")
        rng = np.random.default_rng(500 + s); c = rng.standard_t(5, (len(z), 2)) * 0.4; za = np.column_stack([z, c]); k6 = za.shape[1]
        G = rng.standard_normal((k6, k6)); U, _, Vt = np.linalg.svd(G); Lam = U @ np.diag(np.geomspace(1.0, KAPPA, k6)) @ Vt
        zh = za @ Lam.T + NOISE * rng.standard_normal(za.shape)
        res = zh[1:] - zh[:-1] @ np.linalg.lstsq(np.column_stack([np.ones(len(zh) - 1), zh[:-1]]), zh[1:], rcond=None)[0][1:]
        ica = FastICA(n_components=k6, whiten="unit-variance", max_iter=3000, random_state=s).fit(res); zr = zh @ ica.components_.T
        ac1 = np.array([np.corrcoef(zr[:-1, j], zr[1:, j])[0, 1] for j in range(k6)]); m6, perm = mcc(za, zr)
        pred = set(np.argsort(np.abs(ac1))[:2]); true = set(perm[[4, 5]])
        rows.append(dict(test="V5_spurious", seed=s, copy="bounded", condition_number=float(np.linalg.cond(Lam)), rho_min=rho_min(za, zh), mcc6=m6,
                         ac1_min2=np.sort(np.abs(ac1))[:2].mean(), ac1_top4=np.sort(np.abs(ac1))[2:].mean(), spurious_correct=pred == true,
                         T=T, kappa=KAPPA, noise=NOISE, note="non-Gaussian (t5) iid component; not a direct test of the Gaussian case"))
        P({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in rows[-1].items()})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "V5_spurious.csv"), index=False); return df


def V6():
    rows = []
    for s in range(SEEDS):
        x, z, e, meta = simulation.gen_panel(k=K, E=E, T=T, seed=s, nonlinear=True, innov="gauss", varmode="none")
        masks = (np.abs(np.stack(meta["A_list"])) > 1e-9).astype(float); b3m = b3_margin(meta["A_list"], masks)
        rng = np.random.default_rng(2000 + s); G = rng.standard_normal((K, K)); W = rng.standard_normal((K, K))
        U, _, Vt = np.linalg.svd(G); Lam = U @ np.diag(np.geomspace(1.0, KAPPA, K)) @ Vt
        for alpha in [0.0, 0.5, 1.0, 1.6, 2.2, 3.0]:
            zh = z @ Lam.T + alpha * np.tanh(z @ W); z3, f = step_b3_pattern(zh, e, E, masks, restarts=RESTARTS, seed=s)
            rows.append(dict(test="V6_threshold", alpha=alpha, **diagnostics(z, zh, float(np.linalg.cond(Lam)), "bounded", s), b3_margin=b3m,
                             B3=mcc(z, z3)[0], B3_perm_distance=perm_distance(z, z3)))
            P({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in rows[-1].items()})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "V6_threshold.csv"), index=False); return df


if __name__ == "__main__":
    which = os.environ.get("VWHICH", "12,3,4,5,6").replace("-", ",").split(",")
    t0 = time.time(); P(f"==== estimator-free validation {time.ctime()} seeds={SEEDS} T={T} kappa={KAPPA} which={which}")
    for w, f in (("12", V12), ("3", V3), ("4", V4), ("5", V5), ("6", V6)):
        if w in which:
            f(); P(f"---- {w} done {time.time() - t0:.0f}s")
    P(f"==== DONE {time.time() - t0:.0f}s")
