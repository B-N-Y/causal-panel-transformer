"""R10: evaluation of CPT-2 (the ground truth is used only for scoring).
--mode stage_a  : k=4, no spurious columns; raw MCC, affine R2, affine MCC, Stage B (max of JD/ICA; resolve_rotation). Same seeds
                  and cells as R1/R5/R6 -> paired comparison with iVAE/TDRL/earlier CPT.
--mode whitened : spurious columns on (true dynamics = z (4) + confounder AR(1) = 5; the spurious columns' own iid noise gives
                  2 white sources). Configurations: (k=5, white 0) | (k=7, white 0) | (k=7, white 2). Score: how well the dynamic
                  block (the first k-kw coordinates, excluding the white block) carries w5 = [z, conf] (affine R2, worst-direction
                  canonical correlation rho_min, Stage B MCC).
Output: outputs/cpt2_{mode}{tag}_s{seed}.csv.
"""
import os, sys, argparse, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))

import encoders as KD
from simulation import gen_panel, mcc, rotation_diagnostics, resolve_rotation


def rho_min(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    def isq(C):
        w, V = np.linalg.eigh(C); return V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    return float(np.linalg.svd(isq(np.cov(a.T)) @ (a.T @ b / len(a)) @ isq(np.cov(b.T)), compute_uv=False).min())


def score(w, zh, e, E):
    m, _ = mcc(w[200:], zh[200:]); r2, ma = rotation_diagnostics(w[200:], zh[200:])
    rr = resolve_rotation(w[200:], zh[200:], e[200:], E)
    return dict(MCC=round(float(m), 4), r2_affine=round(r2, 4), mcc_affine=round(ma, 4), mcc_jd=round(rr["mcc_jd"], 4),
                mcc_ica=round(rr["mcc_ica"], 4), B=round(float(np.nanmax([rr["mcc_jd"], rr["mcc_ica"]])), 4),
                rho_min=round(rho_min(w[200:], zh[200:]), 4))


ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["stage_a", "whitened"], required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--E", type=int, default=4)
ap.add_argument("--varmode", default="scale"); ap.add_argument("--innov", default="gauss")
ap.add_argument("--graph", default="fixed"); ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--quick", action="store_true"); ap.add_argument("--tag", default="")
ap.add_argument("--lags", type=int, default=1); ap.add_argument("--nl", type=int, default=0)
ap.add_argument("--k", type=int, default=4); ap.add_argument("--D", type=int, default=12); ap.add_argument("--T", type=int, default=6000)
ap.add_argument("--flow", type=int, default=0, help=">0: CPT-2s selection among 3 candidates (linear, 2 lags + NL, 2 lags + NL + flow of this width)")
ap.add_argument("--enc", default="attn", choices=["attn", "gru", "mlp"], help="encoder ablation; attn = CPT-2")
ap.add_argument("--prior", default="gauss", choices=["gauss", "student"], help="family of the transition prior (prior ablation)")
ap.add_argument("--obs_noise", type=float, default=0.05); ap.add_argument("--ivae", action="store_true", help="iVAE comparison on the same data")
a = ap.parse_args()
T_ = 1500 if a.quick else a.T; steps = 30 if a.quick else a.steps
rows = []
if a.mode == "stage_a":
    x, z, e, meta = gen_panel(seed=a.seed, E=a.E, T=T_, varmode=a.varmode, innov=a.innov, spurious=False, graph=a.graph,
                              k=a.k, D=a.D, obs_noise=a.obs_noise)
    if a.flow:
        zh, inf = KD.fit("CPT2s", x, e, a.k, seed=0, steps=steps, enc_type=a.enc, candidates=((1, 0), (2, 64), (2, 64, a.flow)))
    else:
        zh, inf = KD.fit("CPT2", x, e, a.k, seed=0, steps=steps, mech_lags=a.lags, mech_nl=a.nl, enc_type=a.enc,
                         **({"prior_family": a.prior} if a.prior != "gauss" else {}))
    r = dict(mode=a.mode, enc=a.enc, flow=a.flow, selected_transition=inf.get("selected_transition", ""), seed=a.seed, E=a.E, varmode=a.varmode, innov=a.innov, graph=a.graph, k=a.k, D=a.D, T=T_, obs_noise=a.obs_noise,
             k_white=0, lags=a.lags, nl=a.nl, steps=steps, prior=a.prior, **score(z, zh, e, a.E), split_loss=inf["split_loss"], time=round(inf["time"], 1))
    if a.ivae:
        zi, ii = KD.fit("iVAE", x, e, a.k, seed=0, epochs=2 if a.quick else 200)
        r.update({f"iVAE_{kk}": v for kk, v in score(z, zi, e, a.E).items()})
    rows.append(r)
else:
    x, z, e, meta = gen_panel(seed=a.seed, E=a.E, T=T_, varmode=a.varmode, innov=a.innov, spurious=True, graph=a.graph)
    w = np.column_stack([z, meta["conf"]])
    for k, kw in ((5, 0), (7, 0), (7, 2)):
        zh, inf = KD.fit("CPT2", x, e, k, seed=0, steps=steps, k_white=kw, mech_lags=a.lags, mech_nl=a.nl)
        dyn = zh[:, :k - kw]
        r = dict(mode=a.mode, seed=a.seed, E=a.E, varmode=a.varmode, innov=a.innov, graph=a.graph, k=k, k_white=kw, steps=steps,
                 split_loss=inf["split_loss"], time=round(inf["time"], 1))
        p = score(w, dyn, e, a.E)
        r.update({f"dynamic_{kk}": v for kk, v in p.items()})
        if kw:   # how much of w5 does the white block carry? (should be low)
            r["white_r2_w5"] = round(rotation_diagnostics(w[200:], zh[200:, k - kw:])[0], 4)
        rows.append(r); print(r, flush=True)
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "outputs", f"cpt2_{a.mode}{a.tag}_s{a.seed}.csv"), index=False)
print(rows)
