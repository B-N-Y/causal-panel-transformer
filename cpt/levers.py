"""Lever diagnostics and Stage B, computed WITHOUT access to the ground truth.
Input: Stage A output zh (T x k), observed regime e (T,), number of regimes E; optionally a known zero pattern (E x k x k bool).
Levers:
  B1: the regime profiles of innovation-scale ratios are pairwise distinct -> joint diagonalisation (JD)
  B2: non-Gaussian innovations (Comon) -> ICA
  B3: the zero pattern of A_e is known and dim g_S = 0 -> pattern matching
Tests:
  test_b1: statistic d = min_{i<j} ||Lc_i - Lc_j|| (log-scale profiles of the components after JD, centred across
           regimes). Null distribution: the labels of regime BLOCKS are permuted (breaking the regime-scale link) -> p-value.
  test_b2: Mardia multivariate kurtosis WITHIN each regime (each regime's own residuals, whitened with its own covariance);
           regimes are combined with Stouffer's method. (Pooled residuals are not used: scale variation creates a Gaussian
           scale mixture and triggers the test spuriously.)
Decision rule (`resolve`): returns the results and the reasoning; the ground truth is never accessed.
"""
import numpy as np
from itertools import combinations


# ---------------- shared: regime-wise AR(1) residuals ----------------
def _panel_pairs(zh, e, segment):
    """Panel mode: segment = per-row identifier of a contiguous unit segment (consecutive periods). Within-unit centring
    (fixed effects) and only consecutive row pairs within the same segment."""
    zc = zh - (pd_grp_mean(zh, segment))
    ok = segment[1:] == segment[:-1]
    return zc[:-1][ok], zc[1:][ok], e[1:][ok]


def pd_grp_mean(z, g):
    _, inv = np.unique(g, return_inverse=True); n = np.bincount(inv)
    return np.stack([np.bincount(inv, weights=z[:, j]) / n for j in range(z.shape[1])], 1)[inv]


def ar_residuals(zh, e, E, segment=None):
    """OLS of zh_t on [1, zh_{t-1}] within each regime; returns residuals and regime labels (t>=1).
    If segment is given, panel mode (_panel_pairs)."""
    if segment is not None:
        X0, Y, ee = _panel_pairs(zh, e, segment)
    else:
        zh = zh - zh.mean(0)
        X0, Y, ee = zh[:-1], zh[1:], e[1:]
    res = np.full_like(Y, np.nan)
    for r in range(E):
        idx = np.where(ee == r)[0]
        if len(idx) < 5 * zh.shape[1] + 2:
            continue
        X = np.column_stack([np.ones(len(idx)), X0[idx]])
        B, *_ = np.linalg.lstsq(X, Y[idx], rcond=None)
        res[idx] = Y[idx] - X @ B
    ok = ~np.isnan(res).any(1)
    return res[ok], ee[ok]


def _isqrt(C):
    w, U = np.linalg.eigh(C)
    return U @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ U.T


def joint_diag(mats, iters=100):
    """Cardoso-Souloumiac Jacobi JD (same algorithm as simulation._joint_diag; independent copy)."""
    C = np.array([m.copy() for m in mats]); k = C.shape[1]; V = np.eye(k)
    for _ in range(iters):
        changed = False
        for p in range(k - 1):
            for q in range(p + 1, k):
                g = np.array([[c[p, p] - c[q, q], c[p, q] + c[q, p]] for c in C])
                w, U = np.linalg.eigh(g.T @ g); x, y = U[:, np.argmax(w)]
                if x < 0:
                    x, y = -x, -y
                r = np.sqrt(x * x + y * y); cth = np.sqrt((x + r) / (2 * r)); sth = y / np.sqrt(2 * r * (x + r))
                if abs(sth) < 1e-10:
                    continue
                changed = True
                J = np.eye(k); J[p, p] = J[q, q] = cth; J[p, q] = -sth; J[q, p] = sth
                C = np.array([J.T @ c @ J for c in C]); V = V @ J
        if not changed:
            break
    return V, C


# ---------------- B1 ----------------
def _b1_stat(res, ee, E):
    regs = [r for r in range(E) if (ee == r).sum() > 10 * res.shape[1]]
    if len(regs) < 2:
        return np.nan, None
    W = _isqrt(np.cov(res.T))
    covs = [W @ np.cov(res[ee == r].T) @ W for r in regs]
    V, D = joint_diag(covs)
    L = np.log(np.array([np.diag(d) for d in D]).T)          # k x E' log-scale profiles
    Lc = L - L.mean(1, keepdims=True)
    d = min(np.linalg.norm(Lc[i] - Lc[j]) for i, j in combinations(range(len(Lc)), 2))
    return float(d), W @ V


def _segments(e):
    cut = np.flatnonzero(np.diff(e)) + 1
    return np.split(np.arange(len(e)), cut)


def test_b1(zh, e, E, n_perm=99, seed=0):
    res, ee = ar_residuals(zh, e, E)
    d0, _ = _b1_stat(res, ee, E)
    if np.isnan(d0):
        return dict(b1_stat=np.nan, b1_p=1.0)
    rng = np.random.default_rng(seed)
    segs = _segments(e)
    labs = np.array([e[s[0]] for s in segs])
    null = []
    for _ in range(n_perm):
        ep = np.empty_like(e)
        for s, l in zip(segs, rng.permutation(labs)):
            ep[s] = l
        rp, eep = ar_residuals(zh, ep, E)
        null.append(_b1_stat(rp, eep, E)[0])
    null = np.array([v for v in null if np.isfinite(v)])
    p = (1 + (null >= d0).sum()) / (1 + len(null))
    return dict(b1_stat=d0, b1_null90=float(np.percentile(null, 90)) if len(null) else np.nan, b1_p=float(p))


# ---------------- B2 ----------------
def test_b2(zh, e, E):
    """Within-regime Mardia kurtosis: b2 = mean((r' S^-1 r)^2); H0 (Gaussian): E[b2] = k(k+2), Var = 8k(k+2)/n."""
    res, ee = ar_residuals(zh, e, E)
    k = res.shape[1]; zs = []; b2s = []
    for r in range(E):
        R = res[ee == r]
        if len(R) < 10 * k:
            continue
        R = R - R.mean(0); Si = np.linalg.inv(np.cov(R.T))
        m = np.einsum("ij,jk,ik->i", R, Si, R)
        b2 = float(np.mean(m ** 2)); n = len(R)
        zs.append((b2 - k * (k + 2)) / np.sqrt(8 * k * (k + 2) / n)); b2s.append(b2)
    if not zs:
        return dict(b2_z=np.nan, b2_p=1.0)
    z = float(np.sum(zs) / np.sqrt(len(zs)))
    from scipy.stats import norm
    return dict(b2_z=z, b2_kurt_excess=float(np.mean(b2s) - k * (k + 2)), b2_p=float(norm.sf(z)))


# ---------------- Stage B transformations ----------------
def apply_b1(zh, e, E, segment=None):
    res, ee = ar_residuals(zh, e, E, segment)
    _, T = _b1_stat(res, ee, E)
    return None if T is None else (zh - zh.mean(0)) @ T


def apply_b2(zh, e, E, seed=0, segment=None, b2_filter=None):
    """b2_filter: None (default, plain ICA on the residuals) or one of the options in B2_FILTERS; see apply_b2_filter."""
    from sklearn.decomposition import FastICA
    if b2_filter is not None:
        res, ee = ar_residuals(zh, e, E, segment)
        return (zh - zh.mean(0)) @ _b2_T_filtered(res, ee, E, b2_filter, seed, segment)
    res, _ = ar_residuals(zh, e, E, segment)
    ica = FastICA(n_components=zh.shape[1], whiten="unit-variance", max_iter=3000, random_state=seed).fit(res)
    return (zh - zh.mean(0)) @ ica.components_.T


def apply_b3(zh, e, E, masks):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from stage_b import step_b3_pattern_v2
    z3, f = step_b3_pattern_v2(zh, e, E, masks.astype(float))
    return z3


# ---------------- decision rule ----------------
def resolve(zh, e, E, masks=None, alpha=0.01, rule="b1_first", n_perm=99, seed=0):
    """Chooses a lever without access to the ground truth and applies Stage B.
    rule: 'b1_first' | 'b2_first' | 'both_agree' (if both are significant, compare the two solutions; if they disagree, use B2).
    Returns: (z_resolved | None, info dict)."""
    t1 = test_b1(zh, e, E, n_perm=n_perm, seed=seed); t2 = test_b2(zh, e, E)
    info = dict(**t1, **t2)
    has1, has2 = t1["b1_p"] < alpha, t2["b2_p"] < alpha
    order = ["B1", "B2"] if rule != "b2_first" else ["B2", "B1"]
    choice = None
    if rule == "both_agree" and has1 and has2:
        z1, z2 = apply_b1(zh, e, E), apply_b2(zh, e, E, seed)
        C = np.abs(np.corrcoef(z1.T, z2.T)[:zh.shape[1], zh.shape[1]:])
        from scipy.optimize import linear_sum_assignment
        rr, cc = linear_sum_assignment(-C); agree = float(C[rr, cc].mean())
        info["b1_b2_agreement"] = agree
        info["selection"] = "B1+B2 agree" if agree > 0.95 else "B2 (disagree)"
        return (z1 if agree > 0.95 else z2), info
    for lv in order:
        if lv == "B1" and has1:
            choice = "B1"; break
        if lv == "B2" and has2:
            choice = "B2"; break
    if choice is None and masks is not None:
        choice = "B3"
    info["selection"] = choice or "none (affine only)"
    if choice == "B1":
        return apply_b1(zh, e, E), info
    if choice == "B2":
        return apply_b2(zh, e, E, seed), info
    if choice == "B3":
        return apply_b3(zh, e, E, masks), info
    return None, info


# =====================================================================================================
# Decision rule resolve_v2. The test_b1 / test_b2 above are not adequate decision rules in practice:
#   - test_b1 (distance of the two closest profiles + block permutation) is very conservative and rarely rejects;
#   - test_b2 (within-regime Mardia) has a high false-alarm rate: Stage A contamination makes residuals look non-Gaussian.
# Replacement: (i) SPLIT-HALF STABILITY for Stage B (ICASSO-style), (ii) CROSS-ENCODER canonical agreement for Stage A.
# Together, accepting cases with 'agreement >= .8 and stability >= .98' raised the success rate from a 47.5% base rate to
# 91% in our evaluation; '.9 / .99' gave 97%.
# =====================================================================================================
def _lever_T(res, ee, E, lever, seed=0, b2_filter=None, segment=None):
    if lever == "B1":
        return _b1_stat(res, ee, E)[1]
    if b2_filter is not None:                                    # optional B2 pre-filtering; the default path below is unchanged
        return _b2_T_filtered(res, ee, E, b2_filter, seed, segment)
    from sklearn.decomposition import FastICA
    return FastICA(n_components=res.shape[1], whiten="unit-variance", max_iter=3000, random_state=seed).fit(res).components_.T


def _matched_abs_corr(a, b):
    from scipy.optimize import linear_sum_assignment
    k = a.shape[1]
    C = np.abs(np.corrcoef(a.T, b.T)[:k, k:]); r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean())


def split_half_stability(zh, e, E, lever, segment=None, b2_filter=None):
    """Matched |corr| between the full-data solutions given by the Stage B transformations estimated separately on the two
    halves of the time axis. In panel mode (segment given) the halves are UNIT SEGMENTS (even / odd segment identifiers).
    b2_filter: for B2 the filter is applied SEPARATELY in each half (the volatility filter restarts in each half)."""
    if segment is not None:
        m = (segment % 2 == 0)
        r1, e1 = ar_residuals(zh[m], e[m], E, segment[m]); r2, e2 = ar_residuals(zh[~m], e[~m], E, segment[~m])
    else:
        h = len(zh) // 2
        r1, e1 = ar_residuals(zh[:h], e[:h], E); r2, e2 = ar_residuals(zh[h:], e[h:], E)
    if b2_filter is None:
        T1, T2 = _lever_T(r1, e1, E, lever), _lever_T(r2, e2, E, lever)
    else:
        T1, T2 = _lever_T(r1, e1, E, lever, b2_filter=b2_filter, segment=segment), _lever_T(r2, e2, E, lever, b2_filter=b2_filter, segment=segment)
    if T1 is None or T2 is None:
        return np.nan
    z0 = zh - (zh.mean(0) if segment is None else pd_grp_mean(zh, segment))   # panel: within-unit (fixed effects must not dominate stability)
    return _matched_abs_corr(z0 @ T1, z0 @ T2)


def cross_encoder_rho(zh_a, zh_b):
    """SMALLEST canonical correlation between the outputs of two independent encoders (same data, same time axis):
    a ground-truth-free indicator of a poor encoder's rho_min (Spearman .959 over 100 matched runs)."""
    a = zh_a - zh_a.mean(0); b = zh_b - zh_b.mean(0)
    M = _isqrt(np.cov(a.T)) @ (a.T @ b / len(a)) @ _isqrt(np.cov(b.T))
    return float(np.linalg.svd(M, compute_uv=False).min())


def resolve_v2(zh, e, E, zh_other=None, masks=None, t_cross=0.8, t_stab=0.98, segment=None, b2_filter=None):
    """Oracle-free decision: choose the lever by split-half stability (Algorithm 1 of the paper; the B3 uniqueness check is
    b3_pattern.b3_uniqueness, applied by the caller).
    Returns: (z_resolved | None, info). info['certificate'] in {'RELIABLE', 'SUSPECT', 'NOT_IDENTIFIABLE'}:
    'NOT_IDENTIFIABLE' means abstain (no lever reaches t_stab and no pattern is given); 'RELIABLE' and 'SUSPECT' both mean a
    reported solution and differ only in the cross-encoder agreement (info['cross_rho'] >= t_cross), a diagnostic that is not
    part of the procedure in the paper.
    b2_filter: None (default) or one of the options in B2_FILTERS: B2 stability and the B2 solution use filtered residuals
    (apply_b2_filter). If given, info['b2_filter'] is added; otherwise the info dict is unchanged."""
    info = {}
    if b2_filter is not None:
        info["b2_filter"] = b2_filter
    if zh.shape[1] == 1:                                         # k=1: no rotation ambiguity (scale only) -> Stage B not needed
        a_ = zh[:, 0] - zh[:, 0].mean()
        info["cross_rho"] = float(abs(np.corrcoef(a_, zh_other[:, 0])[0, 1])) if zh_other is not None else np.nan
        info["selection"] = "k=1"
        info["certificate"] = "SUSPECT"                         # one dimension: the lever cannot be tested and a dimension error cannot be ruled out -> never RELIABLE
        return zh - zh.mean(0), info
    cand = {}
    if E >= 2:
        cand["B1"] = split_half_stability(zh, e, E, "B1", segment)
    cand["B2"] = split_half_stability(zh, e, E, "B2", segment) if b2_filter is None else split_half_stability(zh, e, E, "B2", segment, b2_filter=b2_filter)
    info.update({f"stability_{k_}": v for k_, v in cand.items()})
    cand = {k_: v for k_, v in cand.items() if np.isfinite(v)}
    lever, stab = max(cand.items(), key=lambda kv: kv[1]) if cand else (None, np.nan)
    info["cross_rho"] = cross_encoder_rho(zh, zh_other) if zh_other is not None else np.nan
    if (lever is None or stab < t_stab) and masks is not None:
        lever = "B3"
    info["selection"] = lever
    z = apply_b1(zh, e, E, segment) if lever == "B1" else apply_b2(zh, e, E, segment=segment, b2_filter=b2_filter) if lever == "B2" else apply_b3(zh, e, E, masks) if lever == "B3" else None
    ok_b = lever == "B3" or (np.isfinite(stab) and stab >= t_stab)
    ok_a = (not np.isfinite(info["cross_rho"])) or info["cross_rho"] >= t_cross
    info["certificate"] = "RELIABLE" if (ok_a and ok_b and np.isfinite(info["cross_rho"])) else ("NOT_IDENTIFIABLE" if not ok_b else "SUSPECT")
    return z, info


# =====================================================================================================
# Optional PRE-FILTERING before B2 (ICA). With b2_filter=None the code paths above are unchanged.
# Problem: the pooled residual Lambda S_{e_t} eps_t is a SCALE MIXTURE; its components become dependent through the regime,
# a Gaussian shock looks non-Gaussian, and the information used by B1 (scale profiles) leaks into B2.
# In daily panels, volatility clustering produces the same dependence over time.
#   'regime': pooled whitening W (W Lambda S_bar = O orthogonal), then IN EACH REGIME the whitened residuals are re-whitened
#             with the symmetric inverse square root of their own covariance: C_e^{-1/2} W Lambda S_e eps = O eps (exact in
#             the population; C_e = O D_e^2 O', D_e = S_bar^{-1} S_e). The ICA input is then independent eps with the SAME
#             mixing O in ALL regimes: B2 works only if eps is non-Gaussian, and the scale information is removed (no
#             leakage). Solution: (zh - mean) W K'.
#   'ewma' : each residual COLUMN is divided by a univariate conditional variance, h_t = l h_{t-1} + (1-l) r_{t-1}^2,
#            l = 0.94 (RiskMetrics); initial value is the mean square of the first 30 observations. Uses the past only
#            (h_t does not see r_t).
#   'garch': same, with a per-column GARCH(1,1) h_t = w + a r_{t-1}^2 + b h_{t-1}, variance targeting (w = s^2 (1-a-b)),
#            Gaussian quasi-likelihood, L-BFGS-B; does not depend on the `arch` package.
#            CAVEAT: a per-column filter works in MIXED coordinates; dividing each column by a different, time-varying
#            number makes the effective mixing matrix time-dependent -> ICA can be stable but WRONG. These two options are
#            not used in the paper.
#   'ewma_common' / 'garch_common': a SINGLE (common) conditional variance: the same filter applied to
#            q_t = |pooled-whitened residual|^2 / k (garch_common: all coordinates conditionally N(0, h_t) with a common h_t
#            -> the likelihood has the same form in q_t); every row is divided by the same number, so the mixing matrix
#            is UNCHANGED. The appropriate form for a common volatility factor.
#   Combined: 'regime+ewma_common', 'regime+garch_common' (within-regime whitening first, then the common volatility filter).
#   For split-half stability the filters are run separately in each half.
# =====================================================================================================
B2_FILTERS = (None, "regime", "ewma", "garch", "ewma_common", "garch_common", "regime+ewma_common", "regime+garch_common", "single")
# "single": FastICA on the residuals of the regime with the most observations only (Theorem NG: one regime suffices); this
# option does not use apply_b2_filter.


def _ewma_h(x2, lam=0.94, n0=30):
    """x2: series of squares (or a variance proxy); h_t = lam h_{t-1} + (1-lam) x2_{t-1}, h_0 = mean of the first n0."""
    from scipy.signal import lfilter
    h0 = float(np.mean(x2[:max(1, min(n0, len(x2)))]))
    u = (1 - lam) * np.r_[h0, x2[:-1]]
    return lfilter([1.0], [1.0, -lam], u, zi=[lam * h0])[0]


def _garch11_h(x2):
    """GARCH(1,1) conditional variance for x2 = series of squares (or a variance proxy), zero mean, variance targeting.
    Returns (h, (a, b))."""
    from scipy.signal import lfilter
    from scipy.optimize import minimize
    s2 = float(x2.mean()); prev = np.r_[s2, x2[:-1]]

    def h_of(a, b):
        u = s2 * (1 - a - b) + a * prev
        return lfilter([1.0], [1.0, -b], u, zi=[b * s2])[0]

    def nll(th):
        p, q = th; a, b = p * q, p * (1 - q)
        h = np.maximum(h_of(a, b), 1e-12)
        return 0.5 * float(np.sum(np.log(h) + x2 / h)) / len(x2)

    r = minimize(nll, np.array([0.95, 0.05 / 0.95]), method="L-BFGS-B", bounds=[(1e-4, 0.999), (1e-4, 0.9999)])
    p, q = r.x; a, b = p * q, p * (1 - q)
    return np.maximum(h_of(a, b), 1e-12), (float(a), float(b))


def _garch11_var(x):
    """Convenience wrapper: GARCH(1,1) for a single series x (fitted on x^2)."""
    return _garch11_h(x ** 2)


def apply_b2_filter(res, ee, E, b2_filter, segment=None):
    """ICA input y and back-transformation M: the B2 transformation is T = M @ K' (K = FastICA components_ fitted on y)."""
    if b2_filter not in B2_FILTERS or b2_filter is None:
        raise ValueError(f"b2_filter={b2_filter!r}: {B2_FILTERS}")
    k = res.shape[1]; y, M = res, np.eye(k)
    for part in b2_filter.split("+"):
        if part == "regime":
            W = _isqrt(np.cov(y.T)); u = y @ W; y = u.copy()
            for r in range(E):
                m = ee == r
                if m.sum() > k:
                    y[m] = u[m] @ _isqrt(np.cov(u[m].T))
            M = W
            continue
        if segment is not None:
            raise ValueError("the volatility filter is not defined in panel mode (segment): it needs a within-unit time order")
        filt = _ewma_h if part.startswith("ewma") else (lambda v: _garch11_h(v)[0])
        if part.endswith("_common"):
            q = np.mean((y @ _isqrt(np.cov(y.T))) ** 2, 1)
            y = y / np.sqrt(filt(q))[:, None]
        else:
            y = y / np.sqrt(np.column_stack([filt(y[:, j] ** 2) for j in range(k)]))
    return y, M


def _b2_T_filtered(res, ee, E, b2_filter, seed=0, segment=None):
    from sklearn.decomposition import FastICA
    if b2_filter == "single":
        r = int(np.bincount(ee.astype(int), minlength=E).argmax())
        return FastICA(n_components=res.shape[1], whiten="unit-variance", max_iter=3000,
                       random_state=seed).fit(res[ee == r]).components_.T
    y, M = apply_b2_filter(res, ee, E, b2_filter, segment)
    ica = FastICA(n_components=res.shape[1], whiten="unit-variance", max_iter=3000, random_state=seed).fit(y)
    return M @ ica.components_.T
