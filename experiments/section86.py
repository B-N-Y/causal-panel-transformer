"""Panels, sample size and misspecified regimes (Section 8.6 of the paper).

One CPT fit per call (latent dimension fixed at the true value), followed by Stage B with the design lever (B1, joint
diagonalization) and by the lever selection of Section 7 (split-half stability, t_stab = 0.98, B2 on a single regime).
Modes:
  panel        : gen_panel_units with N units of T_unit periods sharing parameters, fitted with the unit structure (blocks);
                 with --single, the same parameters as one series of N*T_unit periods.
  sample_size  : gen_panel (main B1 cell) with T periods.
  regime_noise : gen_panel (main B1 cell, T = 6000); the estimator receives corrupted regime labels:
                 --noise flip --level p (each period gets a uniformly drawn other regime with probability p) or
                 --noise shift --level d (every label delayed by d periods).
The truth is always the uncorrupted latent state; the first 20 periods of every unit are excluded from all metrics.
Output: one CSV row per call in --out_dir.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "cpt"))
import encoders as KD  # noqa: E402
import levers as KL  # noqa: E402
from simulation import gen_panel, gen_panel_units, mcc, rotation_diagnostics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["panel", "sample_size", "regime_noise"])
ap.add_argument("--seed", type=int, default=900)
ap.add_argument("--E", type=int, default=4); ap.add_argument("--k", type=int, default=4)
ap.add_argument("--varmode", default="scale"); ap.add_argument("--innov", default="gauss")
ap.add_argument("--N", type=int, default=10); ap.add_argument("--T_unit", type=int, default=500)
ap.add_argument("--single", action="store_true", help="panel mode: one series with the same parameters and N*T_unit periods")
ap.add_argument("--T", type=int, default=6000)
ap.add_argument("--noise", choices=["none", "flip", "shift"], default="none"); ap.add_argument("--level", type=float, default=0.0)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--out_dir", default="outputs")
ap.add_argument("--tag", default="")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

if a.mode == "panel":
    N, Tu = (1, a.N * a.T_unit) if a.single else (a.N, a.T_unit)
    x, z, e, unit, meta = gen_panel_units(N=N, T_unit=Tu, k=a.k, E=a.E, seed=a.seed, varmode=a.varmode, innov=a.innov)
    e_obs = e
else:
    T_ = a.T if a.mode == "sample_size" else 6000
    x, z, e, meta = gen_panel(seed=a.seed, E=a.E, T=T_, varmode=a.varmode, innov=a.innov, spurious=False, k=a.k)
    unit = np.zeros(len(x), dtype=int); N = 1
    e_obs = e.copy()
    if a.mode == "regime_noise" and a.noise == "flip":
        r = np.random.default_rng([a.seed, 6601])
        hit = r.random(len(e)) < a.level
        e_obs[hit] = (e[hit] + r.integers(1, a.E, hit.sum())) % a.E        # a different regime, uniformly
    elif a.mode == "regime_noise" and a.noise == "shift":
        d = int(a.level); e_obs[d:] = e[:-d] if d > 0 else e; e_obs[:d] = e[0]

blocks = None if N == 1 else list(np.bincount(unit))
segment = None if N == 1 else unit
zh, inf = KD.fit("CPT2", x, e_obs, a.k, seed=0, steps=a.steps, **({"blocks": blocks} if blocks is not None else {}))

keep = np.concatenate([np.where(unit == i)[0][20:] for i in np.unique(unit)])
m_raw, _ = mcc(z[keep], zh[keep]); r2, m_aff = rotation_diagnostics(z[keep], zh[keep])
zb1 = KL.apply_b1(zh, e_obs, a.E, segment)
mcc_b1 = np.nan if zb1 is None else mcc(z[keep], zb1[keep])[0]
zr, info = KL.resolve_v2(zh, e_obs, a.E, segment=segment, b2_filter="single")
mcc_res = np.nan if zr is None else mcc(z[keep], zr[keep])[0]
row = dict(mode=a.mode, seed=a.seed, E=a.E, k=a.k, varmode=a.varmode, innov=a.innov, N=N, T_unit=len(x) // N, T_total=len(x),
           single=a.single, noise=a.noise, level=a.level, share_mislabeled=float((e_obs != e).mean()), steps=a.steps,
           MCC=m_raw, r2_affine=r2, mcc_affine=m_aff, mcc_b1=mcc_b1, solved_b1=bool(mcc_b1 >= 0.90),
           selection=info.get("selection"), certificate=info.get("certificate"),
           stability_B1=info.get("stability_B1"), stability_B2=info.get("stability_B2"),
           reported=info.get("certificate") != "NOT_IDENTIFIABLE", mcc_resolved=mcc_res, time=inf["time"], split_loss=inf["split_loss"])
name = {"panel": f"panel_N{a.N}_T{a.T_unit}{'_single' if a.single else ''}",
        "sample_size": f"T{a.T}", "regime_noise": f"{a.noise}{a.level:g}"}[a.mode]
out = os.path.join(a.out_dir, f"s86_{a.mode}_{name}_{a.varmode}_{a.innov}{a.tag}_s{a.seed}.csv")
pd.DataFrame([row]).to_csv(out, index=False)
print(f"WROTE {out}: r2_aff {r2:.4f} MCC_B1 {mcc_b1:.3f} selection {row['selection']} {row['certificate']} "
      f"stab B1 {row['stability_B1']:.3f} B2 {row['stability_B2']:.3f} time {inf['time']:.0f}s")
