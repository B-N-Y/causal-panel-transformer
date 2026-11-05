"""Nearly proportional profiles, estimator-free level (Section 8.5).
Generator: gen_panel(varmode="scale_delta", delta): s_{j,e} = 0.3 (1 + delta q_{j,e}), q ~ U(-1, 1) fixed per seed; delta = 0 is the
no-lever case (data bit-identical to varmode="none"), small delta gives nearly proportional profiles. Gaussian innovations (no B2),
E = 4, k = 4, no spurious block. No estimator: an affine copy of the true z (condition number 5, noise 0.05, rng 1000 + seed).
The decision rule is that of the pipeline (same functions as levers.resolve_v2): split-half stability of B1 and of single-regime B2;
the larger one is reported if it passes t_stab = 0.98, otherwise NOT_IDENTIFIABLE. The stabilities are stored.
Usage: python experiments/b1_delta_estimator_free.py --delta 0.1 --seed0 700 --n 20 [--innov gauss] [--T 6000] [--tag _X]
Output: outputs/b1_delta_estimator_free{tag}_d{delta}.csv (one row per seed). CPU; a few seconds per seed.
"""
import os, sys, time, argparse, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "cpt"))
import levers as KL
import b3_pattern as B3P
from simulation import gen_panel, mcc

ap = argparse.ArgumentParser()
ap.add_argument("--delta", type=float, required=True); ap.add_argument("--seed0", type=int, default=700); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--innov", default="gauss"); ap.add_argument("--E", type=int, default=4); ap.add_argument("--T", type=int, default=6000)
ap.add_argument("--kappa", type=float, default=5.0); ap.add_argument("--noise", type=float, default=0.05)
ap.add_argument("--t_stab", type=float, default=0.98); ap.add_argument("--quick", action="store_true"); ap.add_argument("--tag", default="")
a = ap.parse_args()
K = 4; T_ = 1500 if a.quick else a.T


def affine_copy(z, seed):
    g = np.random.default_rng(1000 + seed); G = g.standard_normal((K, K)); U, _, Vt = np.linalg.svd(G)
    L = U @ np.diag(np.geomspace(1.0, a.kappa, K)) @ Vt
    return z @ L.T + a.noise * g.standard_normal(z.shape), float(np.linalg.cond(L))


def b1_margin(S):
    P = np.asarray(S) ** 2; P = P / np.linalg.norm(P, axis=0, keepdims=True)
    return float(min(1 - abs(P[:, i] @ P[:, j]) for i, j in itertools.combinations(range(P.shape[1]), 2)))


rows = []
for s in range(a.seed0, a.seed0 + a.n):
    t0 = time.time()
    x, z, e, meta = gen_panel(k=K, E=a.E, T=T_, seed=s, varmode="scale_delta", delta=a.delta, innov=a.innov, spurious=False)
    zh, cond = affine_copy(z, s)
    st1 = KL.split_half_stability(zh, e, a.E, "B1"); st2 = KL.split_half_stability(zh, e, a.E, "B2", b2_filter="single")
    lever = "B1" if st1 >= st2 else "B2"; stab = max(st1, st2)
    z1 = KL.apply_b1(zh, e, a.E); z2 = KL.apply_b2(zh, e, a.E, b2_filter="single")
    m1, m2 = mcc(z[200:], z1[200:])[0], mcc(z[200:], z2[200:])[0]
    r = dict(seed=s, delta=a.delta, innov=a.innov, E=a.E, K=K, T=T_, kappa=a.kappa, noise=a.noise, condition_number=cond, t_stab=a.t_stab,
             generator="gen_panel(varmode=scale_delta)", q_stream="default_rng([seed, 8111])", b1_margin=b1_margin(meta["noise_scale"]),
             stability_B1=st1, stability_B2=st2, selection=lever, stability_selected=stab,
             decision=(lever if stab >= a.t_stab else "NOT_IDENTIFIABLE"),
             mcc_B1=m1, perm_distance_B1=B3P.perm_distance(z[200:], z1[200:]), mcc_B2=m2, perm_distance_B2=B3P.perm_distance(z[200:], z2[200:]),
             mcc_selected=(m1 if lever == "B1" else m2), time=round(time.time() - t0, 2))
    r["solved_B1_perm"] = r["perm_distance_B1"] <= 0.25; r["solved_B1_mcc"] = m1 >= 0.90; r["solved_selected"] = r["mcc_selected"] >= 0.90
    rows.append(r); print({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items()}, flush=True)
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
out = os.path.join(HERE, "outputs", f"b1_delta_estimator_free{a.tag}_d{a.delta:g}.csv")
pd.DataFrame(rows).to_csv(out, index=False); print("WROTE", out)
