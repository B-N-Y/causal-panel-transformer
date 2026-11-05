"""Generator of Counterexample BS (Appendix B of the paper): k = 4, E = 2, S_1 = S_2 (no B1), Gaussian innovations (no B2).
In the coordinates (1, 2, 3, 4):
    A_1 = [[C_2, 0, p], [0', beta_3, 0], [0', 0, beta_4]]     (C_2 2x2, p a 2-vector; 8 zeros)
    A_2 : every entry except (1, 3) is free (drawn dense)       (1 zero)            Z = 9 > n = 6
Conditions (Appendix B, (i)-(iii)): the eigenvalues of C_2 are real and distinct and its off-diagonal entries are nonzero;
beta_3 != beta_4; beta_3, beta_4 are outside spec(C_2); (A_2)_{23} != 0 (local rigidity). The pattern is then locally rigid, but by
(iii) two SPURIOUS orthogonal solutions M_1, M_2 fit it (rotations on J = {1, 2, 4} that fix the third coordinate).
Observations: same nonlinear mixing as gen_panel (simulation.random_mlp: one hidden layer of 32 units, LeakyReLU 0.2, D = 12), same
regime-path construction (piecewise constant, seg = max(T // (6E), 50)) and same observation noise (0.05). No spurious block.
Usage: x, z, e, meta = gen_bs(seed); meta['masks'] (E x k x k, 1 = free), meta['M_spurious'] (the two spurious solutions, in z space).
"""
import numpy as np
from simulation import random_mlp

J = [0, 1, 3]                                  # coordinates of the spurious rotation (0-based; {1, 2, 4} in the paper)


def bs_pattern(k=4):
    """Zero pattern of BS: masks[e, i, j] = 1 free, 0 structural zero."""
    m1 = np.zeros((k, k))
    m1[np.ix_([0, 1], [0, 1])] = 1; m1[[0, 1], 3] = 1; m1[2, 2] = 1; m1[3, 3] = 1
    m2 = np.ones((k, k)); m2[0, 2] = 0
    return np.stack([m1, m2])


def _draw_parameters(rng, max_tries=10000):
    for _ in range(max_tries):
        a, d = rng.uniform(0.2, 0.7, 2)
        b, c = rng.uniform(0.25, 0.7, 2) * rng.choice([-1, 1], 2)
        disc = (a - d) ** 2 + 4 * b * c
        if disc < 0.04:                        # real, distinct eigenvalues (gap >= 0.2)
            continue
        C2 = np.array([[a, b], [c, d]]); lam = np.linalg.eigvals(C2).real
        b3, b4 = rng.uniform(0.2, 0.7, 2)
        if abs(b3 - b4) < 0.15 or min(abs(b3 - lam).min(), abs(b4 - lam).min()) < 0.1:
            continue
        p = rng.uniform(0.3, 0.7, 2) * rng.choice([-1, 1], 2)
        A1 = np.zeros((4, 4)); A1[:2, :2] = C2; A1[:2, 3] = p; A1[2, 2] = b3; A1[3, 3] = b4
        A2 = rng.uniform(-0.5, 0.5, (4, 4)); A2[np.diag_indices(4)] = rng.uniform(0.2, 0.7, 4); A2[0, 2] = 0.0
        if abs(A2[1, 2]) < 0.25:
            continue
        return A1, A2
    raise RuntimeError("could not draw BS parameters")


def spurious_solutions(A1, A2):
    """The two spurious M of Appendix B (iii), one for each eigenvalue of C_2. Returns [M_1, M_2], each 4x4 orthogonal;
    M A_1 M' and M A_2 M' fit the BS pattern. (A is the whitened transition matrix: since S = s I, A~ = A.)"""
    C2, p, b4 = A1[:2, :2], A1[:2, 3], A1[3, 3]
    a = A2[J, 2]                                # A~_2[J, 3]
    lam, V = np.linalg.eig(C2)
    v4 = np.r_[-np.linalg.solve(C2 - b4 * np.eye(2), p), 1.0]
    out = []
    for i in range(2):
        v1 = np.r_[V[:, i].real, 0.0]
        w4 = np.cross(v1, v4); w4 /= np.linalg.norm(w4)
        w1 = np.cross(w4, a); w1 /= np.linalg.norm(w1)   # intersection of P (the plane orthogonal to w4) with the line orthogonal to a
        w2 = np.cross(w4, w1)
        M = np.zeros((4, 4)); M[np.ix_(J, J)] = np.stack([w1, w2, w4]); M[2, 2] = 1.0
        out.append(M)
    return out


def off_pattern_energy(A_list, masks):
    """Same measure as the B3 objective: regime mean of ||A_e (1 - m_e)||^2 / ||A_e||^2."""
    return float(np.mean([((A * (1 - m)) ** 2).sum() / ((A ** 2).sum() + 1e-12) for A, m in zip(A_list, masks)]))


def b3_margin(A_list, masks):
    """Rigidity margin of Lemma 5.6: smallest singular value of L(A~) on so(k)."""
    from itertools import combinations
    k = A_list[0].shape[0]; cols = []
    for i, j in combinations(range(k), 2):
        X = np.zeros((k, k)); X[i, j], X[j, i] = 1.0, -1.0; X /= np.sqrt(2)
        cols.append(np.concatenate([(X @ A - A @ X)[m == 0] for A, m in zip(A_list, masks)]))
    return float(np.linalg.svd(np.column_stack(cols), compute_uv=False).min())


def gen_bs(seed=0, T=6000, D=12, s=0.3, obs_noise=0.05, margin_min=0.15):
    """BS generator. Returns (x, z, e, meta) in the format of gen_panel (no spurious block; meta['conf'] is zero).
    margin_min: a draw is accepted if the rigidity margin of Lemma 5.6 is at least this large at the true solution AND at both
    spurious solutions, so that the three solutions are well isolated (the margin is 0.21-0.24 on the pattern of gen_panel).
    Rejected draws are redrawn from the same stream."""
    k, E = 4, 2
    rng = np.random.default_rng([seed, 5303])            # stream separate from gen_panel
    masks = bs_pattern(k)
    for n_draws in range(1, 1001):
        A1, A2 = _draw_parameters(rng)
        A_list = [A1, A2]
        rho = max(np.abs(np.linalg.eigvals(A)).max() for A in A_list)
        if rho > 0.95:                                   # same common rescaling as gen_panel (pattern and conditions are kept)
            A_list = [A * (0.9 / rho) for A in A_list]
        Ms = spurious_solutions(*A_list)
        margins = [b3_margin(A_list, masks)] + [b3_margin([M @ A @ M.T for A in A_list], masks) for M in Ms]
        if min(margins) >= margin_min:
            break
    else:
        raise RuntimeError("margin_min not reached")
    noise_scale = np.full((E, k), s)                     # S_1 = S_2 = s I
    z = np.zeros((T, k)); e = np.zeros(T, dtype=int)
    seg = max(T // (E * 6), 50); cur = rng.integers(0, E)
    for t in range(1, T):
        if t % seg == 0:
            cur = rng.integers(0, E)
        e[t] = cur
        z[t] = A_list[cur] @ z[t - 1] + s * rng.standard_normal(k)
    g = random_mlp(rng, k, D)
    x = g(z) + obs_noise * rng.standard_normal((T, D))
    nz = np.abs(np.stack(A_list)) > 1e-9
    meta = dict(A_list=A_list, fixed_mask=nz.all(0) & np.isclose(A_list[0], A_list[1]), regime_mask=nz.any(0) & ~nz.all(0),
                crisis=None, spurious_cols=None, graph="bs", noise_scale=noise_scale, mean_shift=np.zeros((E, k)),
                conf=np.zeros(T, dtype=np.float32), conf_mode=None, masks=masks, M_spurious=Ms, generator="bs",
                bs_spurious_energy=[off_pattern_energy([M @ A @ M.T for A in A_list], masks) for M in Ms],
                bs_spurious_perm_distance=[float(np.linalg.norm(np.abs(M) - np.eye(k))) for M in Ms],
                b3_margin=margins, margin_min=margin_min, n_draws=n_draws)
    return x.astype(np.float32), z.astype(np.float32), e, meta
