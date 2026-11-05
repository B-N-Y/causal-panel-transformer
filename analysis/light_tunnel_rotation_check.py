"""Random-rotation check of Section 8.7 (light tunnel, training sequence citris_1). The Stage-A estimate of run LT_RC (regime
contrast, saved as zA in the .npz output of experiments/real_data_citris.py) is whitened and rotated by 1,000 Haar-distributed
orthogonal matrices; for each rotation the MCC with the five recorded factors is computed. Reported: median, 99th percentile and
largest MCC over the rotations, the MCC of the regime-contrast coordinates themselves and that of the Stage-B solution (z).
Usage: python analysis/light_tunnel_rotation_check.py outputs/LT_RC/citris_RC_k5_s0.npz data/light_tunnel/citris_1_truth.csv \
           results/LT_RC/rotation_check.csv"""
import sys
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import ortho_group

NPZ, TRUTH, OUT = sys.argv[1:4]
Z = pd.read_csv(TRUTH)[["red", "green", "blue", "pol_1", "pol_2"]].to_numpy(float)
s = np.load(NPZ); zA, z = s["zA"], s["z"]


def mcc(a, b):
    C = np.abs(np.corrcoef(a.T, b.T)[:5, 5:]); r, c = linear_sum_assignment(-C); return float(C[r, c].mean())


W = zA - zA.mean(0); W = W @ np.linalg.inv(np.linalg.cholesky(np.cov(W.T))).T
rng = np.random.default_rng(0)
m = np.array([mcc(Z, W @ ortho_group.rvs(5, random_state=rng)) for _ in range(1000)])
row = dict(n_rotations=1000, rotation_mcc_median=np.median(m), rotation_mcc_q99=np.quantile(m, 0.99), rotation_mcc_max=m.max(),
           mcc_regime_contrast=mcc(Z, zA), mcc_stage_b=mcc(Z, z))
pd.DataFrame([row]).to_csv(OUT, index=False); print(row)
