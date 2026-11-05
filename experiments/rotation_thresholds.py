"""Dataset-specific random-rotation thresholds for the solved criterion (Sections 8.1 and 8.3 of the paper).

For every dataset that enters a solved count, the true latent state (z, or [z, confounder] when the isolated component is
present) is regenerated and transformed by Haar-distributed random orthogonal matrices, which is the freedom left when no lever
is available (Theorem 5.2(c)). The permutation-matched mean absolute correlation (raw MCC) of these rotations gives the value
that an unidentified solution would reach by chance on that dataset. A run then counts as solved under the stricter criterion
if its MCC after Stage B is at least max(0.90, 99th percentile of the rotation MCC of its dataset).

The datasets are read from the result files given by --results (glob patterns relative to results/); the generator settings
come from the columns of each file. Output: one row per dataset with the 95th and 99th percentiles, the maximum and the share
of rotations with MCC >= 0.90.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import ortho_group

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "cpt"))
from simulation import gen_panel  # noqa: E402

DEFAULT = ["R10/cpt2_stage_a_*.csv", "R10F/cpt2_stage_a_*.csv", "R14/cpt2_stage_a_3c_*.csv", "PIPE150/*.csv",
           "PIPE160/*.csv", "DELTA_PIPE/*.csv"]


def matched_abs_corr(a, b):
    """Raw MCC of b against a: mean |corr| after the Hungarian assignment (same value as simulation.mcc)."""
    a = (a - a.mean(0)) / a.std(0); b = (b - b.mean(0)) / b.std(0)
    C = np.abs(a.T @ b) / len(a)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum() / a.shape[1])


def config_of(row, T_default):
    get = lambda c, d=None: row[c] if c in row and pd.notna(row[c]) else d  # noqa: E731
    delta, spurious = get("delta"), bool(get("spurious", False))
    k = int(get("k_true") - 1) if spurious else int(get("k", get("k_true")))   # pipeline files: k_true counts the confounder
    return dict(seed=int(row["seed"]), E=int(row["E"]), k=k, T=int(get("T", T_default)), varmode=str(row["varmode"]),
                innov=str(row["innov"]), graph=str(get("graph", "fixed")), spurious=spurious,
                delta=None if delta is None else float(delta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=DEFAULT)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--T_default", type=int, default=6000)
    ap.add_argument("--out", default=os.path.join("outputs", "rotation_thresholds.csv"))
    a = ap.parse_args()
    cfgs = {}
    for pat in a.results:
        for f in sorted(glob.glob(os.path.join(HERE, "results", pat))):
            for _, row in pd.read_csv(f).iterrows():
                c = config_of(row, a.T_default)
                cfgs[tuple(sorted(c.items(), key=lambda kv: kv[0]))] = c
    rows = []
    for c in cfgs.values():
        kw = dict(k=c["k"], E=c["E"], T=c["T"], seed=c["seed"], varmode=c["varmode"], innov=c["innov"], graph=c["graph"],
                  spurious=c["spurious"])
        if c["varmode"] == "scale_delta":
            kw["delta"] = c["delta"]
        x, z, e, meta = gen_panel(**kw)
        w = np.column_stack([z, meta["conf"]]) if c["spurious"] else z
        w = w[200:].astype(np.float64)
        rng = np.random.default_rng([c["seed"], c["k"], c["E"], c["T"], 4242])
        rot = np.array([matched_abs_corr(w, w @ ortho_group.rvs(w.shape[1], random_state=rng).T) for _ in range(a.draws)])
        rows.append(dict(c, dim=w.shape[1], draws=a.draws, rot_q95=np.quantile(rot, 0.95), rot_q99=np.quantile(rot, 0.99),
                         rot_max=rot.max(), share_ge_090=float((rot >= 0.90).mean())))
    df = pd.DataFrame(rows).sort_values(["varmode", "innov", "E", "k", "T", "graph", "spurious", "seed"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} datasets; share of rotations with MCC >= 0.90: mean {df.share_ge_090.mean():.3f}, "
          f"max {df.share_ge_090.max():.3f}; 99th percentile: {df.rot_q99.min():.3f}-{df.rot_q99.max():.3f}")


if __name__ == "__main__":
    main()
