"""End-to-end helpers for lever B3.
pattern_versions: correct / missing_edge / extra_free patterns (same rule and RNG as experiments/b3_misspecified_pattern.py).
off_pattern_energy: value of the B3 objective (stage_b.step_b3_pattern_v2) at a solution, recomputed from the solution.
b3_scan: same objective with several starts; returns the distinct local minima that fit the pattern.
solution_class: compares a solution with the correct solution and the spurious solutions of Counterexample BS.
b3_uniqueness: uniqueness check of Section 7 (uses no ground truth).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def pattern_versions(masks, seed):
    rng = np.random.default_rng(seed)
    k = masks.shape[1]; off = ~np.eye(k, dtype=bool)
    present = masks.any(0) & off
    absent = (~masks.any(0)) & off
    out = {"correct": masks.copy()}
    if present.any():
        i, j = np.argwhere(present)[rng.integers(present.sum())]; m = masks.copy(); m[:, i, j] = 0; out["missing_edge"] = m
    if absent.any():
        i, j = np.argwhere(absent)[rng.integers(absent.sum())]; m = masks.copy(); m[:, i, j] = 1; out["extra_free"] = m
    return out


def _regime_X_Y(zc, e, E, k):
    out = {}
    for r in range(E):
        ii = np.where(e[1:] == r)[0]
        if len(ii) >= 5 * k:
            out[r] = (np.column_stack([np.ones(len(ii)), zc[ii]]), zc[ii + 1])
    return out


def off_pattern_energy(z3, e, E, masks):
    """Objective of step_b3_pattern_v2 at the solution z3: regime mean of ||A_r (1 - m_r)||^2 / ||A_r||^2 (A_r: [1, z_t] -> z_{t+1})."""
    k = z3.shape[1]; zc = z3 - z3.mean(0); tot = []
    for r, (X, Y) in _regime_X_Y(zc, e, E, k).items():
        A = np.linalg.lstsq(X, Y, rcond=None)[0][1:].T
        tot.append(((A * (1 - masks[r])) ** 2).sum() / ((A ** 2).sum() + 1e-9))
    return float(np.mean(tot)) if tot else np.nan


def _whiten(zh, e, E):
    k = zh.shape[1]; zc = zh - zh.mean(0); res = []
    for r, (X, Y) in _regime_X_Y(zc, e, E, k).items():
        res.append(Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0])
    w, U = np.linalg.eigh(np.cov(np.concatenate(res).T))
    return zc @ (U @ np.diag(1 / np.sqrt(np.maximum(w, 1e-10))) @ U.T)


def b3_scan(zh, e, E, masks, n_starts=24, seed=0, tol=0.02):
    """Same whitening and objective as step_b3_pattern_v2 (Powell over so(k)), from the identity and n_starts random starts.
    Returns the local minima with f < tol as a list [(f, z_solution)] sorted by f."""
    from scipy.linalg import expm
    from scipy.optimize import minimize
    k = zh.shape[1]; zw = _whiten(zh, e, E); XY = _regime_X_Y(zw, e, E, k)
    iu = [(i, j) for i in range(k) for j in range(i + 1, k)]

    def sk(th):
        X = np.zeros((k, k))
        for v, (i, j) in zip(th, iu):
            X[i, j] = v; X[j, i] = -v
        return X

    def obj(th):
        O = expm(sk(th)); tot = 0.0
        for r, (X, Y) in XY.items():
            u0 = X.copy(); u0[:, 1:] = X[:, 1:] @ O
            A = np.linalg.lstsq(u0, Y @ O, rcond=None)[0][1:].T
            tot += ((A * (1 - masks[r])) ** 2).sum() / ((A ** 2).sum() + 1e-9)
        return tot / max(len(XY), 1)
    rng = np.random.default_rng(seed); out = []
    for th0 in [np.zeros(len(iu))] + [rng.uniform(-np.pi / 2, np.pi / 2, len(iu)) for _ in range(n_starts)]:
        r_ = minimize(obj, th0, method="Powell", options=dict(maxiter=20000, xtol=1e-8, ftol=1e-10))
        if r_.fun < tol:
            out.append((float(r_.fun), zw @ expm(sk(r_.x))))
    return sorted(out, key=lambda t: t[0])


def transformation(z, zr):
    """zr ~ M z (affine regression); |M| with normalized rows."""
    X = np.column_stack([np.ones(len(z)), z]); M = np.linalg.lstsq(X, zr, rcond=None)[0][1:].T
    return np.abs(M) / np.linalg.norm(M, axis=1, keepdims=True)


def perm_distance(z, zr):
    Mn = transformation(z, zr); r, c = linear_sum_assignment(-Mn); P = np.zeros_like(Mn); P[r, c] = 1.0
    return float(np.linalg.norm(Mn - P))


def solution_class(z, zr, M_spurious, threshold=0.25):
    """Whether the solution is the correct one (signed identity; in BS the automorphism group is the sign group) or a spurious M_i.
    Returns (class, {candidate: distance}); class is 'correct', 'spurious1', 'spurious2' or 'other' (nearest candidate above threshold)."""
    Mn = transformation(z, zr); k = Mn.shape[0]
    cand = {"correct": np.eye(k)}
    for i, M in enumerate(M_spurious):
        Ma = np.abs(M); cand[f"spurious{i + 1}"] = Ma / np.linalg.norm(Ma, axis=1, keepdims=True)
    d = {c: float(np.linalg.norm(Mn - A)) for c, A in cand.items()}
    best = min(d, key=d.get)
    return (best if d[best] <= threshold else "other"), d


def b3_uniqueness(sols, f_ratio=3.0, f_add=0.001, threshold=0.25):
    """Uniqueness check of B3 (uses no ground truth). sols: output of b3_scan. Near-optimal: f <= f_ratio f* + f_add. Two solutions
    are distinct if the transformation between them is farther than threshold from a signed permutation. B3 is accepted only
    with exactly one class; with no solution (n_classes = 0) or two or more classes it abstains. Returns (n_classes, n_near_optimal)."""
    if not sols:
        return 0, 0
    fs = sols[0][0]; near = [z_ for f_, z_ in sols if f_ <= f_ratio * fs + f_add]
    reps = []
    for z_ in near:
        if not any(perm_distance(t, z_) <= threshold for t in reps):
            reps.append(z_)
    return len(reps), len(near)
