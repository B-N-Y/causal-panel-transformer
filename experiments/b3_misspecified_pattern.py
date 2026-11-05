"""B3 with a misspecified zero pattern (estimator-free, setting of Section 8.2; basis of the statement in Section 9 of the paper).

Same setting as the estimator-free validation (experiments/estimator_free_validation.py): Stage A is replaced by an affine copy
of the true latent (bounded condition number), Gaussian innovations and no scale variation, so B3 is the only lever.
B3 is applied with four versions of the zero pattern:
  correct      : the true pattern;
  missing_edge : one true cross edge (present in at least one regime) is declared absent in every regime;
  extra_free   : one true zero (a cross entry absent in every regime) is left free;
  transposed   : the transpose of the true pattern (same number of zeros).
Recorded per version: residual off-pattern energy of the B3 solution, MCC, distance to the nearest permutation, solved
(distance <= 0.25).

Usage: python experiments/b3_misspecified_pattern.py [--seed S]   (default: seeds 400-409; about one minute per seed on a CPU)
Output: outputs/b3_misspecified_pattern_s{S}.csv with --seed, otherwise outputs/b3_misspecified_pattern.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "experiments"))
import estimator_free_validation as TD  # noqa: E402

T = 4000


def pattern_versions(masks, rng):
    k = masks.shape[1]; off = ~np.eye(k, dtype=bool)
    present = masks.any(0) & off                               # cross edges present in at least one regime
    absent = (~masks.any(0)) & off                             # cross entries absent in every regime
    out = {"correct": masks.copy()}
    if present.any():
        i, j = np.argwhere(present)[rng.integers(present.sum())]; m = masks.copy(); m[:, i, j] = 0; out["missing_edge"] = m
    if absent.any():
        i, j = np.argwhere(absent)[rng.integers(absent.sum())]; m = masks.copy(); m[:, i, j] = 1; out["extra_free"] = m
    out["transposed"] = np.transpose(masks, (0, 2, 1)).copy()
    return out


def main():
    ap = argparse.ArgumentParser(description="B3 with a misspecified zero pattern (estimator-free).")
    ap.add_argument("--seed", type=int, default=None, help="a single seed (default: seeds 400-409)")
    a = ap.parse_args()
    seeds = [a.seed] if a.seed is not None else range(400, 410)
    rows = []
    for s in seeds:
        x, z, e, meta = TD.simulation.gen_panel(k=TD.K, E=TD.E, T=T, seed=s, nonlinear=True, innov="gauss", varmode="none")
        masks = (np.abs(np.stack(meta["A_list"])) > 1e-9).astype(float)
        zh, cond = TD.affine_copy(z, s, "bounded")
        rng = np.random.default_rng(s)
        for name, m in pattern_versions(masks, rng).items():
            z3, f = TD.step_b3_pattern(zh, e, TD.E, m, restarts=TD.RESTARTS, seed=s)
            dist = TD.perm_distance(z, z3)
            rows.append(dict(seed=s, T=T, pattern=name, n_zeros=int((m == 0).sum()), off_pattern_energy=float(f),
                             mcc=TD.mcc(z, z3)[0], perm_distance=dist, solved=dist <= 0.25,
                             kappa=TD.KAPPA, noise=TD.NOISE, restarts=TD.RESTARTS))
            print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    suffix = f"_s{a.seed}" if a.seed is not None else ""
    d.to_csv(os.path.join(HERE, "outputs", f"b3_misspecified_pattern{suffix}.csv"), index=False)
    print(d.groupby("pattern").agg(n=("seed", "size"), solved=("solved", "sum"), energy_mean=("off_pattern_energy", "mean"),
                                   energy_min=("off_pattern_energy", "min"), energy_max=("off_pattern_energy", "max"),
                                   mcc=("mcc", "mean")).round(4))


if __name__ == "__main__":
    main()
