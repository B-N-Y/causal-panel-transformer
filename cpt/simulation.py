"""
Synthetic panel generator, evaluation metrics and simple reference estimators.
==============================================================================
The generator simulates z_t = A_{e_t} z_{t-1} + b_{e_t} + S_{e_t} eps_t,  x_t = g(z_t) + noise, with an observed regime e_t,
fixed and regime-dependent edges, an optional confounder without dynamics and optional regime variability of the innovation
scales and means. The experiments of the paper call gen_panel and the metrics (mcc, rotation_diagnostics) from other scripts.

The command line runs a small demonstration with simple reference estimators and reports the MCC of the latent coordinates,
the correlation of the estimated edge weights with the true ones and the detection of regime-dependent edges. Raw MCC alone
does not establish identification (Section 8.3 of the paper).

Usage:
  python cpt/simulation.py --sanity                 # numpy-only check of the measurement machinery (no torch needed)
  python cpt/simulation.py                          # demonstration: E in {1,2,4,6} x 5 seeds (requires torch)
  python cpt/simulation.py --E 4 --seeds 3          # quick run
  python cpt/simulation.py --endogenous             # endogenous regimes
  python cpt/simulation.py --linear                 # linear g

Output: results_demo*.csv in the working directory and a summary table.
Requirements: numpy, scipy, pandas; torch for the demonstration (CPU suffices, GPU is faster).
"""

import argparse, sys, time
import numpy as np
from scipy.optimize import linear_sum_assignment

# ----------------------------- Synthetic generator -----------------------------

def make_adjacency(k, E, rng, w_fixed=0.7, w_regime=0.8):
    """List of A_e: fixed edges (identical in all regimes) + regime-specific edges.
    Returns: A_list (E x k x k), fixed_mask, regime_mask, index of the crisis regime."""
    A_base = np.zeros((k, k))
    fixed_mask = np.zeros((k, k), dtype=bool)
    # fixed edges: sub-diagonal chain + one cross edge
    for j in range(k - 1):
        A_base[j + 1, j] = w_fixed * rng.choice([-1, 1])
        fixed_mask[j + 1, j] = True
    A_base[0, k - 1] = 0.4 * rng.choice([-1, 1]); fixed_mask[0, k - 1] = True
    # diagonal (persistence): fixed
    for j in range(k):
        A_base[j, j] = 0.5
        fixed_mask[j, j] = True

    regime_mask = np.zeros((k, k), dtype=bool)
    A_list = []
    crisis = E - 1  # the last regime is the "crisis" regime
    # regime-specific edge: active only in the crisis regime (e.g. a liquidity channel)
    ri, rj = 0, 1
    if k >= 3:
        ri, rj = 2, 0
    regime_mask[ri, rj] = True
    for e in range(E):
        A_e = A_base.copy()
        if E > 1 and e == crisis:
            A_e[ri, rj] = w_regime  # edge that switches on only in the crisis regime
        # sparse mechanism shift: small weight changes across regimes (in few entries)
        if E > 1:
            A_e[1, 0] = A_base[1, 0] * (1.0 + 0.5 * (e / max(E - 1, 1)))
        A_list.append(A_e)
    return A_list, fixed_mask, regime_mask, crisis


def make_adjacency_random(k, E, rng, p_edge=0.35):
    """graph="random": a DIFFERENT topology for each seed, to test generalization beyond the fixed graph (chain + one
    cross edge, nearly circulant). Returns the same objects/masks as make_adjacency (compatible with edge_recovery):
    diagonal persistence U(0.2,0.7); off-diagonal fixed edges with probability p_edge, |w|~U(0.3,0.8), random sign
    (at least one edge); 1-2 regime-specific edges |w|~U(0.5,0.9) that switch on in a random 'crisis' regime;
    sparse mechanism shift: the weight of one fixed edge changes with the regime by the factor (1 + 0.5 e/(E-1))."""
    A_base = np.zeros((k, k)); fixed_mask = np.zeros((k, k), dtype=bool)
    for j in range(k):
        A_base[j, j] = rng.uniform(0.2, 0.7); fixed_mask[j, j] = True
    off = [(i, j) for i in range(k) for j in range(k) if i != j]
    sel = rng.random(len(off)) < p_edge
    if not sel.any():
        sel[rng.integers(len(off))] = True
    for (i, j), s in zip(off, sel):
        if s:
            A_base[i, j] = rng.uniform(0.3, 0.8) * rng.choice([-1, 1]); fixed_mask[i, j] = True
    free = [ij for ij, s in zip(off, sel) if not s]
    regime_mask = np.zeros((k, k), dtype=bool)
    n_reg = min(len(free), 1 if k < 3 else int(rng.integers(1, 3)))
    reg_edges = [free[t] for t in rng.choice(len(free), size=n_reg, replace=False)] if n_reg else []
    reg_w = [rng.uniform(0.5, 0.9) * rng.choice([-1, 1]) for _ in reg_edges]
    for (i, j) in reg_edges:
        regime_mask[i, j] = True
    crisis = int(rng.integers(0, E)) if E > 1 else 0
    shift = [ij for ij, s in zip(off, sel) if s]
    si, sj = shift[int(rng.integers(len(shift)))]
    A_list = []
    for e in range(E):
        A_e = A_base.copy()
        if E > 1 and e == crisis:
            for (i, j), w in zip(reg_edges, reg_w):
                A_e[i, j] = w
        if E > 1:
            A_e[si, sj] = A_base[si, sj] * (1.0 + 0.5 * (e / max(E - 1, 1)))
        A_list.append(A_e)
    return A_list, fixed_mask, regime_mask, crisis


def random_mlp(rng, d_in, d_out, hidden=32):
    W1 = rng.normal(0, 1.0 / np.sqrt(d_in), (d_in, hidden))
    W2 = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, d_out))
    def f(z):
        h = np.maximum(0.2 * (z @ W1), z @ W1)  # leaky relu
        return h @ W2
    return f


def gen_panel(k=4, E=4, T=6000, D=12, seed=0, nonlinear=True, endogenous=False,
              spurious=True, bestcase=False, innov="t5", varmode=None, graph="fixed", obs_noise=0.05,
              conf_mode="ar1", delta=None):
    """z_t = A_{e_t} z_{t-1} + eps ;  x_t = g(z_t) + noise.
    Spurious relation: two additional observed dimensions are driven by a common confounder (no edge).
    endogenous=True: the regime is determined by the magnitude of z (violates exogeneity; stress test).
    bestcase=True: regime-dependent noise scales + latent mean shifts, a best-case environment variability that
    guarantees the 'sufficient variability' condition of Khemakhem et al. The three edge types, heavy tails and
    regime-block structure are kept unchanged.
    varmode: None (follows bestcase) | "none" | "scale" | "mean" | "both" (regime variability in the innovation scale
    s_e and/or the mean shift b_e) | "scale_delta" | "single" (single-target regimes, see below).
    innov: "t5" (Student-t, 5 degrees of freedom; default) | "gauss".
    graph: "fixed" | "random" (see make_adjacency_random).
    obs_noise: standard deviation of the observation noise.
    conf_mode: "ar1" (default) | "iid" (confounder without temporal structure)."""
    rng = np.random.default_rng(seed)
    A_list, fixed_mask, regime_mask, crisis = make_adjacency(k, E, rng)
    # graph="random": the topology comes from a SEPARATE stream; the main stream makes the same draws as for the fixed graph
    # (call above), so the shock/regime/mixing sequences are identical for both graph types (matched comparison).
    if graph == "random":
        A_list, fixed_mask, regime_mask, crisis = make_adjacency_random(
            k, E, np.random.default_rng([seed, 7919]))
    elif graph != "fixed":
        raise ValueError(f"graph={graph!r}: expected 'fixed' or 'random'")
    # stabilization: common scaling in all regimes (relative weights are preserved)
    rho = max(np.abs(np.linalg.eigvals(A)).max() for A in A_list)
    if rho > 0.95:
        A_list = [A * (0.9 / rho) for A in A_list]

    # best-case variability: regime-wise, component-wise noise scale and mean shift.
    # varmode separates variability in the SCALE (s_e) from variability in the MEAN (b_e).
    # varmode=None -> follows bestcase (bestcase: both; otherwise neither). The RNG order is preserved.
    if varmode is None:
        varmode = "both" if bestcase else "none"
    noise_scale = (0.3 * rng.uniform(0.5, 2.0, (E, k)) if varmode in ("scale", "both")
                   else np.full((E, k), 0.3))                # s_e (component-wise)
    mean_shift = (rng.normal(0, 0.8, (E, k)) if varmode in ("mean", "both")
                  else np.zeros((E, k)))                     # b_e (component-wise)
    # varmode="scale_delta": s_{j,e} = 0.3 (1 + delta q_{j,e}), q ~ U(-1, 1) fixed per seed; delta >= 0 tunes the strength of B1
    # continuously (delta = 0: no lever, proportional profiles). q comes from a separate stream, so the main RNG is consumed exactly
    # as with varmode="none": the data with delta = 0 are bit-identical to varmode="none", and delta changes only the scales.
    if varmode == "scale_delta":
        if delta is None or not (0.0 <= float(delta) < 1.0):
            raise ValueError(f"varmode='scale_delta' needs 0 <= delta < 1 (got {delta!r})")
        q = np.random.default_rng([seed, 8111]).uniform(-1.0, 1.0, (E, k))
        noise_scale = 0.3 * (1.0 + float(delta) * q)
    elif delta is not None:
        raise ValueError("delta is used only with varmode='scale_delta'")
    # varmode="single": single-target regimes. Regime 0 is observational (all scales 0.3); regime r >= 1 multiplies only the scale of
    # component (r - 1) mod k by 3. Deterministic: the main RNG is consumed exactly as with varmode="none".
    if varmode == "single":
        noise_scale = np.full((E, k), 0.3)
        for r in range(1, E):
            noise_scale[r, (r - 1) % k] *= 3.0

    z = np.zeros((T, k)); e = np.zeros(T, dtype=int)
    seg = max(T // (E * 6), 50)  # regime block length
    cur = rng.integers(0, E)
    conf = np.zeros(T)  # confounder
    for t in range(1, T):
        if endogenous:
            # endogenous regime: depends on the magnitude of z (no exogeneity)
            cur = int(min(E - 1, np.abs(z[t - 1]).mean() // 0.8))
        elif t % seg == 0:
            cur = rng.integers(0, E)
        e[t] = cur
        # innov="gauss": Gaussian innovations (removes the non-Gaussianity lever). Default: t5.
        raw = rng.standard_normal(k) if innov == "gauss" else rng.standard_t(df=5, size=k)
        eps = raw * noise_scale[cur]  # heavy tail (t5) / Gaussian
        z[t] = A_list[cur] @ z[t - 1] + mean_shift[cur] + eps
        # conf_mode: 'ar1' (default) | 'iid' (no temporal structure in the confounder).
        # With iid the same draw is scaled to the stationary variance of the AR(1) -> same RNG order and marginal variance.
        _c = rng.normal(0, 0.5)
        conf[t] = (0.8 * conf[t - 1] + _c) if conf_mode == "ar1" else _c / np.sqrt(1 - 0.8 ** 2)

    if nonlinear:
        g = random_mlp(rng, k, D - (2 if spurious else 0))
        x_main = g(z)
    else:
        M = rng.normal(0, 1, (k, D - (2 if spurious else 0)))
        x_main = z @ M
    # obs_noise (signal-to-noise ablation): default 0.05; standard normal * scale -> the RNG draws are the same for every value
    x_main = x_main + obs_noise * rng.standard_normal(x_main.shape)

    if spurious:
        # two observed dimensions driven only by the confounder: a SPURIOUS relation between them
        s1 = 1.0 * conf + rng.normal(0, 0.2, T)
        s2 = -0.9 * conf + rng.normal(0, 0.2, T)
        x = np.column_stack([x_main, s1, s2])
    else:
        x = x_main
    meta = dict(A_list=A_list, fixed_mask=fixed_mask, regime_mask=regime_mask,
                crisis=crisis, spurious_cols=(D - 2, D - 1) if spurious else None, graph=graph,
                noise_scale=noise_scale, mean_shift=mean_shift, conf=conf.astype(np.float32), conf_mode=conf_mode)
    if varmode == "scale_delta":                         # only with the new option; the default meta is unchanged
        meta.update(delta=float(delta), q=q)
    return x.astype(np.float32), z.astype(np.float32), e, meta

# ----------------------------- Metrics -----------------------------

def mcc(z_true, z_hat):
    """Mean |correlation| under the best permutation matching, and the matching (column index for each true component).
    z_hat may have more columns than z_true (model dimension > true dimension; rectangular matching)."""
    k, kh = z_true.shape[1], z_hat.shape[1]
    C = np.zeros((k, kh))
    for a in range(k):
        for b_ in range(kh):
            c = np.corrcoef(z_true[:, a], z_hat[:, b_])[0, 1]
            C[a, b_] = 0.0 if np.isnan(c) else abs(c)
    row, col = linear_sum_assignment(-C)
    # if the model dimension < true dimension, unmatched true components count as 0 (average over k)
    return C[row, col].sum() / k, col


def rotation_diagnostics(z_true, z_hat):
    """Rotation-resolution diagnostic.
    r2_affine : mean OLS R^2 of each z_j on [1, z_hat] -> subspace/affine recovery.
    mcc_affine: MCC after fitting z_hat to z by the best affine map -> upper bound reached if the rotation were FULLY resolved.
    raw MCC ~ mcc_affine  => rotation resolved (identifiability signature);
    raw MCC << mcc_affine => only the subspace is found, the rotation is open (baseline effect)."""
    n, k = z_true.shape
    Xd = np.column_stack([np.ones(n), z_hat])
    W, *_ = np.linalg.lstsq(Xd, z_true, rcond=None)
    fit = Xd @ W
    ss_res = ((z_true - fit) ** 2).sum(0); ss_tot = ((z_true - z_true.mean(0)) ** 2).sum(0)
    r2 = float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-12)))
    m_aff, _ = mcc(z_true, fit)
    return r2, float(m_aff)


def _joint_diag(mats, iters=50):
    """Cardoso-Souloumiac Jacobi joint diagonalization (real symmetric matrices).
    Returns: orthogonal V such that V.T @ C @ V is approximately diagonal (for each C)."""
    C = np.array([m.copy() for m in mats]); k = C.shape[1]; V = np.eye(k)
    for _ in range(iters):
        changed = False
        for p_ in range(k - 1):
            for q in range(p_ + 1, k):
                g = np.array([[c[p_, p_] - c[q, q], c[p_, q] + c[q, p_]] for c in C])
                G = g.T @ g
                w, U = np.linalg.eigh(G); x, y = U[:, np.argmax(w)]
                if x < 0: x, y = -x, -y
                r = np.sqrt(x * x + y * y); cth = np.sqrt((x + r) / (2 * r)); sth = y / np.sqrt(2 * r * (x + r))
                if abs(sth) < 1e-9: continue
                changed = True
                J = np.eye(k); J[p_, p_] = J[q, q] = cth; J[p_, q] = -sth; J[q, p_] = sth
                C = np.array([J.T @ c @ J for c in C]); V = V @ J
        if not changed: break
    return V


def resolve_rotation(z_true, z_hat, e, E):
    """CONSTRUCTIVE test of the scale-ratio and non-Gaussianity levers.
    Take the regime-wise AR residuals of the affinely recovered z_hat;
      (JD)  jointly diagonalize the regime innovation covariances  -> scale-ratio lever (B1)
      (ICA) FastICA on the pooled residuals                        -> non-Gaussianity lever (B2, Comon)
    Apply the resulting transformation to z_hat and measure MCC again. Prediction: JD brings the raw MCC close to
    mcc_affine only under scale variability, ICA only under non-Gaussian innovations."""
    from sklearn.decomposition import FastICA
    zh = z_hat - z_hat.mean(0); k = zh.shape[1]
    res = np.zeros_like(zh[1:]); covs = []
    for r in range(E):
        idx = np.where(e[1:] == r)[0]
        if len(idx) < 5 * k: continue
        Xp = np.column_stack([np.ones(len(idx)), zh[idx]]); Y = zh[idx + 1]
        B, *_ = np.linalg.lstsq(Xp, Y, rcond=None); rr = Y - Xp @ B
        res[idx] = rr; covs.append(np.cov(rr.T))
    # whiten (pooled), then resolve the orthogonal freedom
    Cp = np.cov(res.T); w, U = np.linalg.eigh(Cp); Wh = U @ np.diag(1 / np.sqrt(np.maximum(w, 1e-10))) @ U.T
    out = {}
    if len(covs) >= 2:
        V = _joint_diag([Wh @ c @ Wh for c in covs])
        out["mcc_jd"] = float(mcc(z_true, zh @ Wh @ V)[0])
    else:
        out["mcc_jd"] = np.nan
    try:
        ica = FastICA(n_components=k, whiten="unit-variance", max_iter=2000, random_state=0).fit(res)
        out["mcc_ica"] = float(mcc(z_true, zh @ ica.components_.T)[0])
    except Exception:
        out["mcc_ica"] = np.nan
    return out


def dim_g_S(A_list, tol=1e-9):
    """Dimension of g_S = {X in so(k): (X A_e - A_e X)_ij = 0 for all (i,j) not in supp(A_e), for all e}.
    dim g_S = k(k-1)/2 - rank(linear map). If it is 0, the B3 (pattern) lever identifies the rotation (locally)."""
    k = A_list[0].shape[0]; rows = []
    basis = []
    for p_ in range(k):
        for q in range(p_ + 1, k):
            X = np.zeros((k, k)); X[p_, q], X[q, p_] = 1.0, -1.0; basis.append(X)
    for A in A_list:
        zero = np.abs(A) <= tol
        cols = [(X @ A - A @ X)[zero] for X in basis]
        rows.append(np.column_stack(cols))
    M = np.vstack(rows)
    return int(len(basis) - np.linalg.matrix_rank(M, tol=1e-8))


def tdrl_condition(meta):
    """Check of the Gaussian, scale-only special case of Theorem 2 of TDRL (Yao et al., 2022), derived here (the paper
    gives no explicit count): the matrix [1/s_e^2 | 1] (E x (k+1)) has full column rank (k+1) and every A_e is nonsingular.
    Returns: (rank_ok, min |det A_e|). Note: with t5 innovations the route of TDRL Corollary 2 (independent of the number
    of regimes) applies."""
    S = np.asarray(meta["noise_scale"]); E, k = S.shape
    M = np.column_stack([1.0 / S ** 2, np.ones(E)])
    rank_ok = bool(np.linalg.matrix_rank(M) == k + 1)
    mdet = float(min(abs(np.linalg.det(A)) for A in meta["A_list"]))
    return rank_ok, mdet


def edge_recovery(z_hat, e, meta, perm, z_true=None):
    """Regime-wise OLS on the aligned z_hat: z_hat_t ~ Â_e z_hat_{t-1}; compared with A_e.
    Returns: weight correlation (fixed edges), detection of the regime-dependent edge."""
    # alignment: zh[:, a] = z_hat[:, perm[a]]  (perm is the col output of mcc)
    zh = z_hat[:, perm] if perm is not None else z_hat
    if z_true is not None:
        # scale-indeterminacy correction: identifiability holds up to permutation and per-dimension scale; to avoid the
        # distortion A[i,j] -> (s_i/s_j) A[i,j] in OLS, z_hat is rescaled to the std of the true z (the sign is
        # absorbed by |.|)
        zh = zh / (zh.std(0, keepdims=True) + 1e-8) * z_true.std(0, keepdims=True)
    A_list = meta["A_list"]; k = zh.shape[1]; E = len(A_list)
    fixed_mask, regime_mask, crisis = meta["fixed_mask"], meta["regime_mask"], meta["crisis"]
    A_hat = []
    for ee in range(E):
        idx = np.where(e[1:] == ee)[0] + 1
        if len(idx) < 5 * k:
            A_hat.append(np.full((k, k), np.nan)); continue
        X, Y = zh[idx - 1], zh[idx]
        A_hat.append((np.linalg.pinv(X) @ Y).T)
    # sign indeterminacy (each latent may be +/-) -> ABSOLUTE weights are compared
    wt, wh = [], []
    for ee in range(E):
        if np.isnan(A_hat[ee]).any(): continue
        wt += list(np.abs(A_list[ee][fixed_mask])); wh += list(np.abs(A_hat[ee][fixed_mask]))
    wcorr = np.corrcoef(wt, wh)[0, 1] if len(wt) > 2 else np.nan
    # regime-dependent edge: large in the crisis regime and small in the other regimes?
    det = np.nan
    if E > 1 and not np.isnan(A_hat[crisis]).any():
        ri, rj = np.argwhere(regime_mask)[0]
        w_true = abs(A_list[crisis][ri, rj])
        in_crisis = abs(A_hat[crisis][ri, rj])
        others = [abs(A_hat[ee][ri, rj]) for ee in range(E)
                  if ee != crisis and not np.isnan(A_hat[ee]).any()]
        det = float(in_crisis > 0.5 * w_true and
                    (len(others) == 0 or max(others) < 0.5 * in_crisis))
    return wcorr, det

# ----------------------------- Estimators -----------------------------

def fit_torch(x, e, k, E, epochs=60, seed=0, verbose=False):
    """iVAE-lite: encoder [x, onehot(e)] -> z ; decoder z -> x ;
    regime-indexed linear dynamic prior (learned A_e) + mechanism-sparsity penalty."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(x, device=dev)
    Eoh = torch.eye(E, device=dev)[torch.tensor(e, device=dev)]
    D = x.shape[1]
    enc = nn.Sequential(nn.Linear(D + E, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                        nn.Linear(64, k)).to(dev)
    dec = nn.Sequential(nn.Linear(k, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                        nn.Linear(64, D)).to(dev)
    A = nn.Parameter(torch.randn(E, k, k, device=dev) * 0.1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + [A], lr=2e-3)
    ei = torch.tensor(e, device=dev)
    for ep in range(epochs):
        opt.zero_grad()
        z = enc(torch.cat([X, Eoh], 1))
        rec = ((dec(z) - X) ** 2).mean()
        zp = torch.einsum("tij,tj->ti", A[ei[1:]], z[:-1])
        dyn = ((z[1:] - zp) ** 2).mean()
        sparse = (A - A.mean(0, keepdim=True)).abs().mean()  # mechanism sparsity
        var_pen = (z.var(0) - 1).abs().mean()                # prevents collapse
        loss = rec + 1.0 * dyn + 0.05 * sparse + 0.1 * var_pen
        loss.backward(); opt.step()
        if verbose and ep % 20 == 0:
            print(f"  ep{ep:3d} loss {loss.item():.4f} rec {rec.item():.4f} dyn {dyn.item():.4f}")
    with torch.no_grad():
        z = enc(torch.cat([X, Eoh], 1)).cpu().numpy()
    return z

def fit_ivae(x, e, k, E, epochs=4000, seed=0, lr=2e-3, verbose=False,
             return_prior=False):
    """iVAE (Khemakhem et al., 2020, auxiliary variable = regime): conditional VAE trained by the ELBO.
    q(z|x,e) Gaussian encoder; p(x|z) Gaussian decoder (learned observation log-variance);
    p(z|e) FACTORIZED prior conditioned on the environment, with a learned component-wise mean (pm) and log-variance (plv)
    per regime. This prior is the mechanism that breaks rotation invariance: the KL term forces the latent axes to align
    with the axes that differ across regimes."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(x, device=dev)
    ei = torch.tensor(e, device=dev)
    Eoh = torch.eye(E, device=dev)[ei]
    D = x.shape[1]
    enc = nn.Sequential(nn.Linear(D + E, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                        nn.Linear(64, 2 * k)).to(dev)
    dec = nn.Sequential(nn.Linear(k, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                        nn.Linear(64, D)).to(dev)
    obs_lv = nn.Parameter(torch.zeros(D, device=dev))        # observation log-variance
    pm = nn.Parameter(torch.zeros(E, k, device=dev))          # prior mean (per regime)
    plv = nn.Parameter(torch.zeros(E, k, device=dev))         # prior log-variance (per regime)
    params = list(enc.parameters()) + list(dec.parameters()) + [obs_lv, pm, plv]
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        h = enc(torch.cat([X, Eoh], 1))
        mu, lv = h[:, :k], h[:, k:].clamp(-8, 4)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        xh = dec(z)
        # -log p(x|z) (Gaussian, constants dropped)
        nll = 0.5 * (((X - xh) ** 2) / obs_lv.exp() + obs_lv).sum(1).mean()
        # KL( q(z|x,e) || p(z|e) ): analytic, factorized Gaussian
        pmt, plvt = pm[ei], plv[ei].clamp(-8, 4)
        kl = 0.5 * (plvt - lv + (lv.exp() + (mu - pmt) ** 2) / plvt.exp() - 1).sum(1).mean()
        loss = nll + kl
        loss.backward(); opt.step()
        if verbose and ep % 500 == 0:
            print(f"  ep{ep:4d} elbo-loss {loss.item():.3f} nll {nll.item():.3f} kl {kl.item():.3f}")
    with torch.no_grad():
        h = enc(torch.cat([X, Eoh], 1))
        zh = h[:, :k].cpu().numpy()  # posterior mean
    if return_prior:
        return zh, pm.detach().cpu().numpy(), plv.detach().cpu().numpy()
    return zh


def fit_temporal_ivae(x, e, k, E, epochs=4000, seed=0, lr=2e-3, hidden=64,
                      verbose=False, return_loss=False):
    """Temporal iVAE: conditional VAE trained by the ELBO, with a prior CONDITIONED ON THE PAST:
    p(z_t | z_{t-1}, e) = N(A_e z_{t-1} + b_e, diag(exp(plv_e))).
    In the panel SCM the latent components factorize conditionally on the past (the innovations are independent across
    components); this is the empirical counterpart of history-conditioned factorization.
    Uses temporal structure (A_e z_{t-1}) and regime variability (b_e, plv_e) together."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(x, device=dev)
    ei = torch.tensor(e, device=dev)
    Eoh = torch.eye(E, device=dev)[ei]
    D = x.shape[1]
    enc = nn.Sequential(nn.Linear(D + E, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                        nn.SiLU(), nn.Linear(hidden, 2 * k)).to(dev)
    dec = nn.Sequential(nn.Linear(k, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                        nn.SiLU(), nn.Linear(hidden, D)).to(dev)
    obs_lv = nn.Parameter(torch.zeros(D, device=dev))
    Ad = nn.Parameter(torch.randn(E, k, k, device=dev) * 0.1)  # regime-wise dynamics
    b = nn.Parameter(torch.zeros(E, k, device=dev))
    plv = nn.Parameter(torch.zeros(E, k, device=dev))
    params = list(enc.parameters()) + list(dec.parameters()) + [obs_lv, Ad, b, plv]
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        h = enc(torch.cat([X, Eoh], 1))
        mu, lv = h[:, :k], h[:, k:].clamp(-8, 4)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        xh = dec(z)
        nll = 0.5 * (((X - xh) ** 2) / obs_lv.exp() + obs_lv).sum(1).mean()
        # history-conditioned prior (t>=1); the z_{t-1} sample is detached (the prior treats the past as data)
        pm_t = torch.einsum("tij,tj->ti", Ad[ei[1:]], z[:-1].detach()) + b[ei[1:]]
        plv_t = plv[ei[1:]].clamp(-8, 4)
        mu1, lv1 = mu[1:], lv[1:]
        kl = 0.5 * (plv_t - lv1 + (lv1.exp() + (mu1 - pm_t) ** 2) / plv_t.exp() - 1).sum(1).mean()
        loss = nll + kl
        loss.backward(); opt.step()
        if verbose and ep % 500 == 0:
            print(f"  ep{ep:4d} loss {loss.item():.3f} nll {nll.item():.3f} kl {kl.item():.3f}")
    with torch.no_grad():
        h = enc(torch.cat([X, Eoh], 1))
        mu, lv = h[:, :k], h[:, k:].clamp(-8, 4)
        xh = dec(mu)
        nll_f = 0.5 * (((X - xh) ** 2) / obs_lv.exp() + obs_lv).sum(1).mean()
        pm_t = torch.einsum("tij,tj->ti", Ad[ei[1:]], mu[:-1]) + b[ei[1:]]
        plv_t = plv[ei[1:]].clamp(-8, 4)
        kl_f = 0.5 * (plv_t - lv[1:] + (lv[1:].exp() + (mu[1:] - pm_t) ** 2)
                      / plv_t.exp() - 1).sum(1).mean()
        zh = mu.cpu().numpy()
    if return_loss:
        return zh, float(nll_f + kl_f)  # unsupervised selection criterion (for choosing among restarts)
    return zh


def fit_pca(x, k):
    """Baseline: first k PCA components of the centred x (no environment, no training)."""
    xc = x - x.mean(0)
    U, S, Vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ Vt[:k].T


def test_prior():
    """Unit test: is the conditional prior of the iVAE set up correctly?
    On best-case data with E=4, after short training (a) the regime-wise prior means should move clearly apart;
    (b) the aligned prior means should be highly correlated with the regime-wise empirical means of the true z."""
    print("== PRIOR UNIT TEST (iVAE conditional prior) ==")
    x, z, e, meta = gen_panel(k=4, E=4, T=6000, seed=0, nonlinear=True, bestcase=True)
    zh, pm, plv = fit_ivae(x, e, 4, 4, epochs=1500, seed=0, return_prior=True)
    m, perm = mcc(z[200:], zh[200:])
    # (a) differentiation across regimes: the means are clearly away from the initial value (0)
    dists = [np.linalg.norm(pm[a] - pm[b]) for a in range(4) for b in range(a + 1, 4)]
    print(f"  MCC (short training): {m:.3f}")
    print(f"  pairwise L2 distance of prior means: min={min(dists):.3f} max={max(dists):.3f}")
    assert max(dists) > 0.5, "prior means do not differ across regimes: the KL path is broken"
    # (b) alignment with the truth: pm[e, perm] vs the true regime-wise mean of z
    true_means = np.stack([z[e == ee].mean(0) for ee in range(4)])  # E x k
    signs = np.array([np.sign(np.corrcoef(z[200:, a], zh[200:, perm[a]])[0, 1])
                      for a in range(4)])  # component-wise sign alignment
    aligned = pm[:, perm] * signs
    c = np.corrcoef(true_means.ravel(), aligned.ravel())[0, 1]
    print(f"  correlation of prior means with true regime means: {abs(c):.3f}")
    assert abs(c) > 0.6, f"prior means are not aligned with the true regime structure ({c:.3f})"
    print("== PRIOR TEST PASSED: the conditional prior learns the regime structure. ==")


# ----------------------------- Run logic -----------------------------

def run_config(E, seed, args):
    x, z, e, meta = gen_panel(k=args.k, E=E, T=args.T, D=args.D, seed=seed,
                              nonlinear=not args.linear, endogenous=args.endogenous,
                              bestcase=getattr(args, "bestcase", False),
                              varmode=getattr(args, "varmode", None))
    est = getattr(args, "estimator", "v1")
    hid = getattr(args, "hidden", 64)
    if est == "ivae":
        zh = fit_ivae(x, e, args.k, E, epochs=args.epochs, seed=seed)
    elif est == "tivae":
        zh = fit_temporal_ivae(x, e, args.k, E, epochs=args.epochs, seed=seed, hidden=hid)
    elif est == "pca":
        zh = fit_pca(x, args.k)
    else:
        zh = fit_torch(x, e, args.k, E, epochs=args.epochs, seed=seed)
    m, perm = mcc(z[200:], zh[200:])
    wcorr, det = edge_recovery(zh, e, meta, perm, z_true=z)
    r2a, maff = rotation_diagnostics(z[200:], zh[200:])
    rr = resolve_rotation(z[200:], zh[200:], e[200:], E)
    run_config.last_diag = (r2a, maff, rr["mcc_jd"], rr["mcc_ica"])
    return m, wcorr, det


def sanity():
    """Numpy-only check of the MACHINERY: do the generator and the metrics work correctly?
    (NOT a scientific result: a test of the measurement machinery.)"""
    print("== SANITY: generator + metric machinery ==")
    x, z, e, meta = gen_panel(k=4, E=4, T=4000, seed=1, nonlinear=True)
    print(f"panel: x{x.shape}, z{z.shape}, regime counts: {np.bincount(e)}")
    # 1) MCC machinery: build z_hat with a known permutation + noise -> MCC should be high
    rng = np.random.default_rng(2)
    P = rng.permutation(4)
    z_hat = z[:, P] * rng.choice([-1, 1], 4) + rng.normal(0, 0.1, z.shape)
    m, perm = mcc(z, z_hat)
    assert m > 0.95, f"MCC machinery is wrong: {m}"
    print(f"MCC (known perm + noise): {m:.3f}  OK")
    # 2) edge-recovery machinery: OLS on the true z -> weight correlation should be high
    wcorr, det = edge_recovery(z_hat, e, meta, perm)
    print(f"edge weight correlation: {wcorr:.3f}  (>0.9 expected)")
    print(f"regime-dependent edge detection: {det}  (1.0 expected)")
    assert wcorr > 0.9 and det == 1.0, "edge machinery is wrong"
    # 3) E=1 check: there is no regime-dependent edge; the machinery must not crash
    x1, z1, e1, meta1 = gen_panel(k=4, E=1, T=2000, seed=3)
    m1, p1 = mcc(z1, z1 + np.random.default_rng(4).normal(0, 0.1, z1.shape))
    print(f"E=1 run OK (MCC machinery check {m1:.3f})")
    print("== SANITY PASSED: the measurement machinery is correct. Run with torch for the demonstration. ==")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--E", type=int, default=None, help="single value of E (default: {1,2,4,6})")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--D", type=int, default=12)
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--endogenous", action="store_true", help="endogenous regimes")
    ap.add_argument("--estimator", choices=["v1", "ivae", "tivae", "pca"], default="v1",
                    help="v1: rec+dyn MSE autoencoder; ivae: ELBO iVAE; "
                         "tivae: temporal iVAE with history-conditioned prior; pca: baseline")
    ap.add_argument("--Egrid", type=str, default=None,
                    help="comma-separated list of E, e.g. 1,2,4,6,10 (default: {1,2,4,6})")
    ap.add_argument("--hidden", type=int, default=64, help="hidden width of encoder/decoder")
    ap.add_argument("--bestcase", action="store_true",
                    help="generator with regime-dependent noise scale + mean shift")
    ap.add_argument("--varmode", type=str, default=None, choices=["none", "scale", "mean", "both"],
                    help="regime variability in the scale and/or the mean only (None: follows --bestcase)")
    ap.add_argument("--tag", type=str, default="", help="suffix of the output file")
    ap.add_argument("--test-prior", action="store_true",
                    help="unit test: is the iVAE conditional prior set up correctly")
    args = ap.parse_args()

    if args.sanity:
        sanity(); return
    if args.test_prior:
        test_prior(); return

    try:
        import torch  # noqa
    except ImportError:
        print("torch not found. First run:  pip install torch  (CPU suffices)"); sys.exit(1)
    import pandas as pd

    if args.Egrid:
        E_grid = [int(v) for v in args.Egrid.split(",")]
    else:
        E_grid = [args.E] if args.E else [1, 2, 4, 6]
    # the output file name depends on the mode
    tag = "".join([f"_{args.estimator}" if args.estimator != "v1" else "",
                   "_bestcase" if args.bestcase else "",
                   "_endogenous" if args.endogenous else "",
                   "_linear" if args.linear else "",
                   f"_v{args.varmode}" if args.varmode else "", args.tag])
    out_csv = f"results_demo{tag}.csv"
    rows = []
    for E in E_grid:
        for s in range(args.seeds):
            t0 = time.time()
            m, wcorr, det = run_config(E, s, args)
            rows.append(dict(E=E, seed=s, MCC=m, edge_wcorr=wcorr, regime_edge_det=det,
                             sec=round(time.time() - t0, 1),
                             k=args.k, D=args.D, T=args.T, epochs=args.epochs,
                             linear=args.linear, endogenous=args.endogenous,
                             estimator=args.estimator, bestcase=args.bestcase,
                             hidden=args.hidden, varmode=args.varmode,
                             r2_affine=round(run_config.last_diag[0], 4), mcc_affine=round(run_config.last_diag[1], 4),
                             mcc_jd=round(run_config.last_diag[2], 4), mcc_ica=round(run_config.last_diag[3], 4)))
            print(f"E={E} seed={s}  MCC={m:.3f}  wcorr={wcorr:.3f}  "
                  f"regime-edge={det}  ({rows[-1]['sec']}s)")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nResults written: {out_csv}")
    summ = df.groupby("E")[["MCC", "edge_wcorr", "regime_edge_det"]].mean().round(3)
    print("\n== SUMMARY (mean over seeds) ==\n", summ)



if __name__ == "__main__":
    main()


def gen_panel_multi(N=20, k=4, E=4, T=1000, D=12, seed=0, nonlinear=True,
                    endogenous=False, spurious=True, bestcase=False,
                    hetero=0.15):
    """PANEL generator (a multi-unit counterpart of gen_panel; gen_panel itself is unchanged).
    N units share the SAME mechanisms (A_e), with small unit-level heterogeneity (hetero)
    and INDEPENDENT innovations. The regime sequence is COMMON to all units (macro regime).
    Rationale: with a single series, recovery can be limited by the amount of data; panel replication (N>1)
    relaxes this limit through shared mechanisms.
    Returns: x (N,T,D), z (N,T,k), e (T,), meta"""
    rng = np.random.default_rng(seed)
    A_list, fixed_mask, regime_mask, crisis = make_adjacency(k, E, rng)
    rho = max(np.abs(np.linalg.eigvals(A)).max() for A in A_list)
    if rho > 0.95:
        A_list = [A * (0.9 / rho) for A in A_list]      # spectral scaling as in gen_panel
    if bestcase:
        scales = 0.3 * rng.uniform(0.5, 2.0, size=(E, k))
        shifts = rng.normal(0, 0.8, size=(E, k))
    else:
        scales = np.full((E, k), 0.3); shifts = np.zeros((E, k))
    # COMMON macro regime sequence
    e = np.zeros(T, dtype=int); seg = max(T // (E * 6), 25); cur = rng.integers(0, E)
    for t in range(1, T):
        if t % seg == 0:
            cur = rng.integers(0, E)
        e[t] = cur
    g = random_mlp(rng, k, D - (2 if spurious else 0)) if nonlinear else None
    M = None if nonlinear else rng.normal(0, 1, (k, D - (2 if spurious else 0)))
    Z = np.zeros((N, T, k)); X = []
    for i in range(N):
        # unit heterogeneity: small deviation in the mechanisms (sharing is preserved)
        Ai = [A * (1 + hetero * rng.normal(0, 1, A.shape)) for A in A_list]
        z = np.zeros((T, k)); conf = np.zeros(T)
        for t in range(1, T):
            eps = rng.standard_t(df=5, size=k) * scales[e[t]]
            z[t] = Ai[e[t]] @ z[t - 1] + shifts[e[t]] + eps
            conf[t] = 0.8 * conf[t - 1] + rng.normal(0, 0.5)
        Z[i] = z
        xm = g(z) if nonlinear else z @ M
        xm = xm + rng.normal(0, 0.05, xm.shape)
        if spurious:
            s1 = 1.0 * conf + rng.normal(0, 0.2, T)
            s2 = -0.9 * conf + rng.normal(0, 0.2, T)
            xm = np.column_stack([xm, s1, s2])
        X.append(xm)
    meta = dict(A_list=A_list, fixed_mask=fixed_mask, regime_mask=regime_mask,
                crisis=crisis, spurious_cols=(D - 2, D - 1) if spurious else None,
                N=N, hetero=hetero)
    return np.stack(X).astype(np.float32), Z.astype(np.float32), e, meta


def gen_panel_units(N=10, T_unit=500, k=4, E=4, D=12, seed=0, varmode="scale", innov="gauss", graph="fixed",
                    obs_noise=0.05, seg=50, burn=100):
    """PANEL with SHARED parameters (Section 8.6 of the paper; gen_panel and gen_panel_multi are unchanged).
    The transition matrices, innovation scales and mixing are drawn once per seed, from streams that do not depend on N or
    T_unit, so gen_panel_units(N=1, T_unit=N*T_unit, seed=s) is a single series with the same parameters as the panel
    gen_panel_units(N, T_unit, seed=s). Units have independent innovations and independent regime paths (blocks of seg
    periods, each with a regime drawn uniformly), and each unit starts after a burn-in of burn periods, so that its initial
    state has a positive density (A-z). Intercepts are zero; no spurious columns.
    Returns: x (N*T_unit, D), z (N*T_unit, k), e (N*T_unit,), unit (N*T_unit,), meta"""
    rp = np.random.default_rng([seed, 5501])                          # parameter stream
    A_list, fixed_mask, regime_mask, crisis = make_adjacency(k, E, rp)
    if graph == "random":
        A_list, fixed_mask, regime_mask, crisis = make_adjacency_random(k, E, np.random.default_rng([seed, 7919]))
    elif graph != "fixed":
        raise ValueError(f"graph={graph!r}: expected 'fixed' or 'random'")
    rho = max(np.abs(np.linalg.eigvals(A)).max() for A in A_list)
    if rho > 0.95:
        A_list = [A * (0.9 / rho) for A in A_list]
    if varmode == "scale":
        noise_scale = 0.3 * rp.uniform(0.5, 2.0, (E, k))
    elif varmode == "none":
        noise_scale = np.full((E, k), 0.3)
    else:
        raise ValueError(f"varmode={varmode!r}: expected 'scale' or 'none'")
    g = random_mlp(np.random.default_rng([seed, 5502]), k, D)
    X, Z, EE, U = [], [], [], []
    for i in range(N):
        ru = np.random.default_rng([seed, 5503, i])                    # unit stream: regimes, innovations, noise
        n = T_unit + burn; e = np.zeros(n, dtype=int); z = np.zeros((n, k))
        cur = ru.integers(0, E)
        for t in range(n):
            if t % seg == 0:
                cur = ru.integers(0, E)
            e[t] = cur
            raw = ru.standard_normal(k) if innov == "gauss" else ru.standard_t(df=5, size=k)
            z[t] = (A_list[cur] @ z[t - 1] if t > 0 else 0.0) + raw * noise_scale[cur]
        z, e = z[burn:], e[burn:]
        X.append(g(z) + obs_noise * ru.standard_normal((T_unit, D))); Z.append(z); EE.append(e); U.append(np.full(T_unit, i))
    meta = dict(A_list=A_list, fixed_mask=fixed_mask, regime_mask=regime_mask, crisis=crisis, noise_scale=noise_scale,
                N=N, T_unit=T_unit, seg=seg, burn=burn)
    return (np.concatenate(X).astype(np.float32), np.concatenate(Z).astype(np.float32), np.concatenate(EE),
            np.concatenate(U), meta)
