"""Latent dimension selection on held-out data.
select_dimension(x, e, ks, seeds): for each k an encoder is fitted and the loss on the last 20% of the time axis (not used in
training) is measured; the elbow rule below picks k*.
select_dimension_v2(x, e, kmax): over-select, then prune (reproducible directions with temporal structure).
"""
import numpy as np
import encoders as KD


def select_dimension(x, e, ks=range(2, 8), seeds=(0, 1), method="iVAE", c=0.07, epochs=200):
    """ELBOW rule. L(k) = seed-averaged held-out loss; Delta_k = L(k-1) - L(k) (improvement in held-out loss).
    k* = the first k with Delta_{k+1} < c * max Delta (stop once the improvement falls below the fraction c of the largest
    improvement). A two-fit reproducibility rule is not used: in a non-temporal iVAE the independent noise of spurious
    columns appears as a reproducible 'dimension'.
    The threshold c was tuned on separate calibration seeds. Under-selecting the dimension is costly (it can produce a wrong
    reported solution), whereas over-selecting is harmless."""
    curve = {}
    for k in ks:
        ls = [KD.fit(method, x, e, k, seed=s, epochs=epochs)[1]["split_loss"] for s in seeds]
        curve[k] = (float(np.mean(ls)), float(np.std(ls) / np.sqrt(len(ls))) if len(ls) > 1 else 0.0)
    ks_ = sorted(curve); L = np.array([curve[k][0] for k in ks_])
    d = -(np.diff(L))                                     # d[i] = L(k_i) - L(k_{i+1})
    kstar = ks_[-1]
    for i, di in enumerate(d):
        if di < c * d.max():
            kstar = ks_[i]; break
    return kstar, dict(curve=curve, rule=f"elbow c={c}")


def _structure_tests(u, e, lag=10):
    """Second-moment temporal-structure tests for one direction: (ii) Brown-Forsythe p-value (Levene, median-centred) for
    equality of variances across regimes, (iii) Ljung-Box (McLeod-Li) p-value on u^2."""
    from scipy import stats
    gr = [u[e == r] for r in np.unique(e) if (e == r).sum() >= 3]
    p_regime = float(stats.levene(*gr, center="median").pvalue) if len(gr) >= 2 else 1.0
    q = u ** 2 - (u ** 2).mean(); n = len(q); d = (q ** 2).sum()
    rho = np.array([(q[h:] * q[:-h]).sum() / d for h in range(1, lag + 1)])
    Q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lag + 1)))
    p_vol = float(stats.chi2.sf(Q, lag))
    return p_regime, p_vol


def select_dimension_v2(x, e, kmax=8, seeds=(0, 1), corr_threshold=0.9, r2_threshold=0.01, fit_kw=None, screen="level",
                        alpha_screen=0.01):
    """'OVER-SELECT, THEN PRUNE'. An extra dimension is harmless, a missing dimension is costly (wrong reported solution):
    (1) fit CPT-2 with kmax on two seeds; (2) the canonical correlations between the two solutions above corr_threshold
    define the REPRODUCIBLE directions; (3) among these, directions without temporal structure (not predictable from their
    past: VAR(1) R^2 < r2_threshold; in the model a white/spurious component, e.g. the iid noise of spurious columns) are
    removed. k* = number of remaining directions.
    screen: "level" (default; temporal structure in the level only) or "level_or_second_moment" (a direction is also kept if
    it shows temporal structure in the second moment: regime-dependent variance or volatility clustering, with p-values
    below alpha_screen; in return data the level of true shocks is unpredictable).
    Returns: k*, info (canonical correlations, per-direction R^2)."""
    import encoders as KD
    fit_kw = fit_kw or {}
    Z = [KD.fit_cpt2(x, e, kmax, seed=s, **fit_kw)[0] for s in seeds]
    a, b = (Z[0] - Z[0].mean(0)), (Z[1] - Z[1].mean(0))
    def isq(C):
        w, V = np.linalg.eigh(C); return V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    Wa, Wb = isq(np.cov(a.T)), isq(np.cov(b.T))
    U, s, Vt = np.linalg.svd(Wa @ (a.T @ b / len(a)) @ Wb)
    common = a @ Wa @ U[:, s > corr_threshold]                  # reproducible directions (coordinates of the first seed)
    r2 = []
    if common.shape[1]:
        X0 = np.column_stack([np.ones(len(common) - 1), common[:-1]]); Y1 = common[1:]
        B = np.linalg.lstsq(X0, Y1, rcond=None)[0]; R = Y1 - X0 @ B
        r2 = list(1 - R.var(0) / Y1.var(0))
    if screen == "level":                                       # default
        kstar = int(sum(r > r2_threshold for r in r2))
        return max(kstar, 1), dict(canonical=[float(v) for v in s], reproducible=int((s > corr_threshold).sum()),
                                   r2=[float(v) for v in r2], rule=f"v2 kmax={kmax} corr>{corr_threshold} r2>{r2_threshold}")
    # 'level_or_second_moment': a direction is kept if it shows temporal structure in the level (VAR(1) R^2) OR in the
    # second moment (regime-dependent variance, volatility clustering).
    assert screen == "level_or_second_moment", screen
    e1 = np.asarray(e)[: len(common)]
    pt = [_structure_tests((common[:, j] - common[:, j].mean()) / common[:, j].std(), e1) for j in range(common.shape[1])]
    keep = [bool(r > r2_threshold or pr < alpha_screen or po < alpha_screen) for r, (pr, po) in zip(r2, pt)]
    return max(int(sum(keep)), 1), dict(canonical=[float(v) for v in s], reproducible=int((s > corr_threshold).sum()),
                                        r2=[float(v) for v in r2], p_regime=[p[0] for p in pt], p_volatility=[p[1] for p in pt],
                                        kept=keep, rule=f"v2-second-moment kmax={kmax} corr>{corr_threshold} (r2>{r2_threshold}"
                                                        f" | p_regime<{alpha_screen} | p_volatility<{alpha_screen})")
