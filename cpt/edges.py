"""Edge typing + weights + uncertainty.
Input: z after Stage B (T x k), regime e. In each regime z_t ~ [1, z_{t-1}] by OLS -> Â_e with CLASSICAL OLS standard errors
(in the model class the innovations are independent over time). Multiple testing: Holm (all regime x input tests jointly).
  edge present     : H0 Â_e[i,j] = 0 rejected by Holm
  FIXED            : edge present in all regimes and none of the regime-pair difference tests (Holm) is significant
  REGIME_DEPENDENT : edge present in some regimes and absent in others, or at least one regime-pair difference is significant
  isolated coordinate: a component with no cross edge (other than its self-edge) in any regime; returned under the key
                     'spurious_latent' (name kept for compatibility). It may be a confounder without effects on the other
                     coordinates or an autonomous latent factor, which the data do not distinguish (paper, Section 6).
Scale: weights are reported under the unit-innovation normalization (pooled residual std = 1 per component).
Classical standard errors with Holm correction are used rather than a regime-block bootstrap: with only a few blocks per regime,
the block bootstrap produced false positives even on the true latent.
"""
import numpy as np
from scipy.stats import norm


def _holm(p, alpha):
    flat = p.ravel(); o = np.argsort(flat); m = len(flat); rej = np.zeros(m, bool)
    for i, j in enumerate(o):
        if flat[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    return rej.reshape(p.shape)


def ols(z, e, E, segment=None):
    """Regime-wise OLS of z_t on [1, z_{t-1}]. If segment (unit labels of a panel) is given: within-unit demeaning, only
    consecutive pairs within the same unit, and UNIT-CLUSTERED standard errors."""
    k = z.shape[1]; A = np.full((E, k, k), np.nan); S = np.full((E, k, k), np.nan); res = []
    if segment is not None:
        from levers import pd_grp_mean
        z = z - pd_grp_mean(z, segment); valid = np.r_[False, segment[1:] == segment[:-1]]
    for r in range(E):
        idx = np.where(e[1:] == r)[0] + 1
        if segment is not None:
            idx = idx[valid[idx]]
        if len(idx) < 5 * k:
            continue
        X = np.column_stack([np.ones(len(idx)), z[idx - 1]]); Y = z[idx]
        B, *_ = np.linalg.lstsq(X, Y, rcond=None); R = Y - X @ B
        XtXi = np.linalg.inv(X.T @ X)
        if segment is None:
            s2 = (R ** 2).sum(0) / (len(idx) - k - 1)
            S[r] = np.sqrt(np.outer(s2, np.diag(XtXi)[1:]))
        else:                                               # sandwich estimator, clusters = units, each equation separately
            _, g = np.unique(segment[idx], return_inverse=True); G = g.max() + 1
            for i in range(k):
                U = np.zeros((G, X.shape[1])); np.add.at(U, g, X * R[:, [i]])
                V = XtXi @ (U.T @ U) @ XtXi * G / max(G - 1, 1)
                S[r, i] = np.sqrt(np.diag(V)[1:])
        A[r] = B[1:].T; res.append(R)
    return A, S, np.concatenate(res)


def normalize(z, e, E, segment=None):
    _, _, R = ols(z - z.mean(0), e, E, segment)
    return (z - z.mean(0)) / R.std(0)


def type_edges(z, e, E, alpha=0.05, min_weight=0.10, segment=None, **_):
    """Returns A (regime-wise weights), se, lower/upper (1 - alpha intervals), present (edge present, E x k x k),
    type (FIXED / REGIME_DEPENDENT / none, k x k) and spurious_latent (indices of isolated coordinates).
    segment: optional unit labels for panel data (see ols)."""
    z = normalize(z, e, E, segment); k = z.shape[1]
    A, S, _ = ols(z, e, E, segment)
    ok = ~np.isnan(A[:, 0, 0])
    p = np.where(np.isnan(A), 1.0, 2 * norm.sf(np.abs(A / S)))
    # Statistical significance + PRACTICAL magnitude (|A| >= min_weight on the unit-innovation scale). On an estimated latent,
    # small leakage produces tiny cross edges that are 'significant' in large samples; the magnitude threshold removes them.
    # The threshold was chosen on a validation set and should be confirmed on independent data.
    exists = _holm(p, alpha) & ok[:, None, None] & (np.abs(np.nan_to_num(A)) >= min_weight)
    pairs = [(a, b) for a in range(E) for b in range(a + 1, E) if ok[a] and ok[b]]
    if pairs:
        pd_ = np.array([2 * norm.sf(np.abs((A[a] - A[b]) / np.sqrt(S[a] ** 2 + S[b] ** 2))) for a, b in pairs])
        differs = _holm(pd_, alpha).any(0)
    else:
        differs = np.zeros((k, k), bool)
    types = np.full((k, k), "none", dtype=object)
    for i in range(k):
        for j in range(k):
            v = exists[ok, i, j]
            if v.any():
                types[i, j] = "FIXED" if (v.all() and not differs[i, j]) else "REGIME_DEPENDENT"
    off = ~np.eye(k, dtype=bool)
    spurious = [i for i in range(k) if not (exists[:, i, :][:, off[i]].any() or exists[:, :, i][:, off[:, i]].any())]
    z95 = norm.ppf(1 - alpha / 2)
    return dict(A=A, se=S, lower=A - z95 * S, upper=A + z95 * S, present=exists, type=types, spurious_latent=spurious)
