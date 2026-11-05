"""Optional Stage-A pre-projection by regime contrast (for high-dimensional observations; never accesses ground truth).
For every regime r other than a reference regime, the directions of the observation space along which the step x_t - x_{t-1}
varies most in regime r relative to the reference regime: the generalized eigenvectors of (Cov(dx | e = r), Cov(dx | e = ref))
with the largest eigenvalues. Under lever B1 the innovation scale of some latent components differs across regimes, and these
directions concentrate on them; the projection replaces a variance-ordered reduction (PCA), which favours directions of large
variance whether or not they carry the latent state.
"""
import numpy as np
from scipy.linalg import eigh


def regime_contrast(x, e, m, ref=None, fit_frac=0.8, ridge=1e-6):
    """x: (T, D) observations; e: (T,) observed regime; m: directions per non-reference regime.
    ref: reference regime (default: the most frequent one in the fitted part).
    fit_frac: the directions are estimated on the first fit_frac of the series (the held-out part of the encoders is not used).
    Returns xp (T, m (E - 1)), standardized on the fitted part, and info {"W", "ref", "eigenvalues"}."""
    x = np.asarray(x, np.float64); T, D = x.shape; E = int(e.max()) + 1; cut = int(T * fit_frac)
    dx = np.diff(x[:cut], axis=0); ee = e[1:cut]
    if ref is None:
        ref = int(np.bincount(e[:cut], minlength=E).argmax())
    S0 = np.cov(dx[ee == ref].T); S0 = S0 + ridge * np.trace(S0) / D * np.eye(D)
    W, lam = [], []
    for r in range(E):
        if r == ref:
            continue
        ev, V = eigh(np.cov(dx[ee == r].T), S0); o = np.argsort(ev)[::-1][:m]
        W.append(V[:, o]); lam.append(ev[o])
    W = np.concatenate(W, axis=1); xp = x @ W
    xp = (xp - xp[:cut].mean(0)) / xp[:cut].std(0)
    return xp.astype(np.float32), dict(W=W, ref=ref, eigenvalues=np.concatenate(lam))


def regime_information(z, e, fit_frac=0.8, ridge=1e-6):
    """Held-out information about the observed regime in the innovations of a Stage-A latent z (T, k); never uses ground truth.
    A pooled AR(1) with intercept is fitted on the first fit_frac of the series; the residuals r_t are classified by regime-wise
    Gaussians N(m_e, S_e) (quadratic discriminant, class priors = fitted frequencies). Returns the information gain in nats on the
    held-out part: H(prior) - cross-entropy. Invariant to invertible affine maps of z (up to the ridge), so it compares Stage-A
    candidates that are identified only up to an affine map. Under lever B1 the regime changes the innovation covariance, so a
    latent that has captured the latent state carries this information in its residuals."""
    z = np.asarray(z, np.float64); T, k = z.shape; E = int(e.max()) + 1; cut = int(T * fit_frac)
    Z1 = np.c_[np.ones(T - 1), z[:-1]]; B = np.linalg.lstsq(Z1[:cut - 1], z[1:cut], rcond=None)[0]
    r = z[1:] - Z1 @ B; ee = e[1:]; tr, te = np.arange(cut - 1), np.arange(cut - 1, T - 1)
    pri = np.bincount(ee[tr], minlength=E) / len(tr); logp = np.full((len(te), E), -np.inf)
    for c in range(E):
        rc = r[tr][ee[tr] == c]
        if len(rc) <= k + 1:
            continue
        m = rc.mean(0); S = np.cov(rc.T) + ridge * np.trace(np.cov(r[tr].T)) / k * np.eye(k)
        L = np.linalg.cholesky(S); u = np.linalg.solve(L, (r[te] - m).T)
        logp[:, c] = np.log(pri[c]) - 0.5 * (u ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * k * np.log(2 * np.pi)
    mx = logp.max(1, keepdims=True); lse = mx[:, 0] + np.log(np.exp(logp - mx).sum(1))
    ce = float(np.mean(lse - logp[np.arange(len(te)), ee[te]]))
    h0 = float(-np.mean(np.log(np.maximum(pri[ee[te]], 1e-12))))
    return h0 - ce


def regime_concentration(x, e, ref=0, fit_frac=0.8, n_null=20, seed=0):
    """Are the regimes single-target? For every regime r != ref, the generalized eigenvalues lambda_i of
    (Cov(dx | e = r), Cov(dx | e = ref)) on the first fit_frac of the series. Null: the steps of the reference regime are split
    n_null times at random into a subset of size min(n_r, n_ref / 2) and the rest; q = largest |log lambda| over these splits.
    Excess f_i = max(0, |log lambda_i| - q); concentration c_r = f_(1) / sum f (nan if no excess).
    Returns {r: dict(c, n_excess, q, top)}; never uses ground truth. ref: the observational regime, known from the design."""
    x = np.asarray(x, np.float64); T, D = x.shape; E = int(e.max()) + 1; cut = int(T * fit_frac)
    dx = np.diff(x[:cut], axis=0); ee = e[1:cut]; d0 = dx[ee == ref]; rng = np.random.default_rng(seed)

    def logev(a, b):
        S0 = np.cov(b.T); S0 = S0 + 1e-6 * np.trace(S0) / D * np.eye(D)
        return np.abs(np.log(np.clip(eigh(np.cov(a.T), S0, eigvals_only=True), 1e-12, None)))
    out = {}
    for r in range(E):
        if r == ref:
            continue
        dr = dx[ee == r]; m = min(len(dr), len(d0) // 2); q = 0.0
        for _ in range(n_null):
            idx = rng.permutation(len(d0)); q = max(q, logev(d0[idx[:m]], d0[idx[m:]]).max())
        f = np.maximum(0.0, logev(dr, d0) - q); s = f.sum()
        out[r] = dict(c=float(f.max() / s) if s > 0 else float("nan"), n_excess=int((f > 0).sum()), q=float(q),
                      top=float(np.sort(f)[::-1][0]))
    return out
