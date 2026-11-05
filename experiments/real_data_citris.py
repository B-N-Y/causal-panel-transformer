"""Real data with a recorded latent state: light tunnel of the Causal Chambers, camera images (experiment citris_1 of dataset
lt_crl_benchmark_v1; Gamella, Bing and Runge 2025, ICML; Gamella, Peters and Buhlmann 2025, Nature Machine Intelligence).
Data (not redistributed here): https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/lt_crl_benchmark_v1.zip
(MD5 8ec492afffd0142abd70b1fa7082b104, CC BY 4.0). --images is the array of the 100,000 images of citris_1 in the order of
citris_1.csv (uint8, 100000 x 64 x 64 x 3); --truth is a CSV with the columns flag, red, green, blue, pol_1, pol_2 of citris_1.csv.
Latent state (never seen by the methods): the brightness of the red, green and blue light and the angles of the two polarizers.
Observed: the images, reduced to their first 256 principal components (unsupervised; 99.6% of the pixel variance), each
standardized (--scale pc) or all divided by the standard deviation of the first one (--scale common: relative variances kept).
Regime: flag (0 no intervention; 1-5 intervention on red, green, blue, polarizer 1, polarizer 2), E = 6.
--encoder CPT2 | iVAE | TDRL fits Stage A with the latent dimension 5 (--k_fixed) or, for CPT2 without --k_fixed, with the
dimension selected by the pipeline; --encoder PCA uses the first 5 principal components; --encoder RC uses the regime-contrast projection with one
direction per intervention regime (contrast.regime_contrast, no training). --rc_m m: the encoders see the regime-contrast
projection with m directions per regime instead of the 256 components. Stage B: lever selection of Section 7
(B2 on a single regime). Evaluation against the latent state: Stage-A R^2, MCC (Pearson and Spearman) of the resolved state,
solved criterion of Section 8.1, lagged edges between the matched components.
Output: <out_dir>/citris_<tag>_s<seed>.csv (one row) and .npz."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import ortho_group, rankdata

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "cpt"))
sys.path.insert(0, os.path.join(HERE, "experiments"))

TRUTH = ["red", "green", "blue", "pol_1", "pol_2"]


def features(images, cache, p=256, scale="pc"):
    """scale: "pc" standardizes every component; "common" divides all components by the standard deviation of the first one,
    so that the components keep their relative variances."""
    if os.path.exists(cache):
        return np.load(cache)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(np.load(images).reshape(100000, -1), dtype=torch.float32, device=dev) / 255.
    X -= X.mean(0)
    ev, V = torch.linalg.eigh(X.T @ X / len(X))
    F = (X @ V[:, torch.argsort(ev, descending=True)[:p]]).cpu().numpy()
    F = ((F - F.mean(0)) / (F.std(0) if scale == "pc" else F[:, 0].std())).astype(np.float32)
    tmp = cache + f".{os.getpid()}.npy"; np.save(tmp, F); os.replace(tmp, cache)
    return F


def mcc(z_true, z_hat, spearman=False):
    if spearman:
        z_true = np.apply_along_axis(rankdata, 0, z_true); z_hat = np.apply_along_axis(rankdata, 0, z_hat)
    k = z_true.shape[1]
    C = np.nan_to_num(np.abs(np.corrcoef(z_true.T, z_hat.T)[:k, k:]))
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum() / k), list(zip(r, c)), C[r, c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--truth", required=True)
    ap.add_argument("--encoder", choices=["CPT2", "iVAE", "TDRL", "PCA", "RC"], default="CPT2")
    ap.add_argument("--k_fixed", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=16, help="iVAE/TDRL: 16 epochs of the 80,000 training rows = 20,000 updates")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--scale", choices=["pc", "common"], default="pc")
    ap.add_argument("--rc_m", type=int, default=None)
    ap.add_argument("--w_rc", type=float, default=None, help="CPT2: weight of the regime-contrast training term")
    ap.add_argument("--rows", default=None, help="start:end, a contiguous block of the series (robustness across time blocks)")
    a = ap.parse_args()
    if a.encoder != "CPT2" and a.k_fixed is None:
        ap.error("--encoder iVAE, TDRL, PCA and RC need --k_fixed")
    import levers as KL
    from pipeline import run_pipeline
    from rotation_thresholds import matched_abs_corr
    os.makedirs(a.out_dir, exist_ok=True)
    x = features(a.images, os.path.join(a.out_dir, "citris_x_pca256" + ("" if a.scale == "pc" else "_" + a.scale) + ".npy"), scale=a.scale)
    d = pd.read_csv(a.truth); z_true = d[TRUTH].to_numpy(float); e = d.flag.to_numpy(int); E = int(e.max()) + 1
    if a.rows:
        r0, r1 = map(int, a.rows.split(":")); x, z_true, e = x[r0:r1], z_true[r0:r1], e[r0:r1]
    if a.encoder in ("PCA", "RC"):
        if a.encoder == "PCA":
            zA = x[:, :a.k_fixed] * 1.0                  # x holds the principal components in order
        else:
            from contrast import regime_contrast
            zA = regime_contrast(x, e, 1)[0].astype(float)
        z, info = KL.resolve_v2(zA, e, E, b2_filter="single")
        out = dict(zA=zA, z=z, stage_b=info, dimension=dict(k_selected=a.k_fixed))
        if z is not None and info["certificate"] != "NOT_IDENTIFIABLE":
            import edges as KN
            out["edges"] = KN.type_edges(z, e, E)
    else:
        out = run_pipeline(x, e, seed=a.seed, encoders=(a.encoder, a.encoder), cpt2_steps=a.steps, epochs=a.epochs,
                           b2_filter="single", k_fixed=a.k_fixed, **({"rc_m": a.rc_m} if a.rc_m is not None else {}),
                           **({"cpt2_w_rc": a.w_rc} if a.w_rc else {}))
    k = int(out["dimension"]["k_selected"]); sb = out["stage_b"]; z = out["z"]
    reported = z is not None and sb.get("certificate") != "NOT_IDENTIFIABLE"
    zA = out["zA"]; A1 = np.c_[np.ones(len(zA)), zA]
    res = z_true - A1 @ np.linalg.lstsq(A1, z_true, rcond=None)[0]; r2 = 1 - res.var(0) / z_true.var(0)
    rng = np.random.default_rng([0, 5, E, len(z_true), 4242]); zs = (z_true - z_true.mean(0)) / z_true.std(0)
    q99 = float(np.quantile([matched_abs_corr(zs, zs @ ortho_group.rvs(5, random_state=rng).T) for _ in range(200)], .99))
    tag = f"{a.encoder}_{'k' + str(a.k_fixed) if a.k_fixed else 'pipeline'}" + (f"_rc{a.rc_m}" if a.rc_m else "") + (f"_r{a.rows.split(':')[0]}" if a.rows else "") + (f"_wrc{a.w_rc:g}" if a.w_rc else "")
    row = dict(encoder=a.encoder, k_fixed=a.k_fixed, seed=a.seed, scale=a.scale, rc_m=a.rc_m, rows=a.rows, w_rc=a.w_rc, steps=a.steps, epochs=a.epochs, T=len(x), D=x.shape[1], E=E,
               k_selected=k, stageA_r2_mean=float(r2.mean()), **{f"stageA_r2_{n}": float(v) for n, v in zip(TRUTH, r2)},
               selection=sb.get("selection"), certificate=sb.get("certificate"), reported=reported,
               stability_B1=sb.get("stability_B1"), stability_B2=sb.get("stability_B2"), cross_rho=sb.get("cross_rho"),
               rot_q99=q99, solved_threshold=max(0.90, q99))
    save = dict(zA=zA)
    if z is not None:
        m, pairs, cs = mcc(z_true, z); ms, _, _ = mcc(z_true, z, spearman=True)
        row.update(mcc=m, mcc_spearman=ms, solved=bool(reported and m >= row["solved_threshold"]),
                   **{f"corr_{TRUTH[i]}": float(c) for (i, _), c in zip(pairs, cs)})
        save.update(z=z, match=np.array(pairs))
        if reported:
            P = out["edges"]["present"]; idx = dict(pairs)                    # true index -> estimated index
            row["edges_present"] = ";".join(f"{TRUTH[j]}->{TRUTH[i]}" for i in idx for j in idx
                                            if P[:, idx[i], idx[j]].any())   # P[e, to, from] (edges.type_edges: A[e] = B.T)
            save.update(A=out["edges"]["A"], present=P)
    stem = os.path.join(a.out_dir, f"citris_{tag}_s{a.seed}")
    pd.DataFrame([row]).to_csv(stem + ".csv", index=False); np.savez(stem + ".npz", **save)
    print(row)


if __name__ == "__main__":
    main()
