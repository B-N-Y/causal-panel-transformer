"""R15: a third data family, the NCTRL generator (Song et al., NeurIPS 2023; github.com/xiangchensong/nctrl, MIT, d2540bb;
tools/generate_data.py::arhmm_pnl_change_gaussian_ts with the authors' __main__ setting: 8 latents, 2 lags, length 4, 5 HIDDEN Markov
regimes (stickiness 0.9), regime-dependent permuted nonlinear transition, 3-layer leaky-ReLU mixing).
Our model requires an OBSERVED regime, so the true regime labels are supplied (also to TDRL). CPT-2s: linear 1-lag versus 2-lag + NL
transition, selected by the held-out loss (as in R13). Evaluation: affine R^2, affine MCC and raw MCC at the last step (as in R7).
Output: outputs/cpt_on_nctrl_generator_{method}{tag}_s{seed}.csv
"""
import os, sys, argparse, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))

sys.path.insert(0, os.path.join(HERE, "third_party", "nctrl", "tools"))
import tdrl_generator as R7
import generate_data as NG

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n", type=int, default=40000)
ap.add_argument("--method", choices=["CPT2s", "TDRL"], required=True); ap.add_argument("--quick", action="store_true")
ap.add_argument("--tag", default=""); ap.add_argument("--window", type=int, default=6, help="number of last steps used by CPT2s (for a like-for-like comparison with TDRL: 3)")
ap.add_argument("--tdrl_epochs", type=int, default=18, help="number of TDRL epochs")
ap.add_argument("--steps", type=int, default=20000, help="number of CPT training updates")
ap.add_argument("--only", default="", help="restrict to these candidates: 'NL' (2 lags + NL64), 'FLOW' (flow candidate) or 'NL-FLOW'")
ap.add_argument("--flow", type=int, default=0, help=">0: the CPT-2 flow candidate (2 lags + NL64 + conditional monotone-transform prior of this width) also competes")
a = ap.parse_args()
np.random.seed(a.seed)                                   # the generator partly uses the global np.random state
D = NG.arhmm_pnl_change_gaussian_ts(NClass=5, lags=2, Nlayer=3, length=4, negSlope=0.2, latent_size=8, noise_scale=0.1,
                                    batch_size=500 if a.quick else a.n, state_stickiness=0.9, random_seed=a.seed)["data"]
X = D["X"].astype(np.float32); Y = D["Z"].astype(np.float32); C = D["C"].astype(np.int64)
X = (X - X.reshape(-1, X.shape[-1]).mean(0)) / X.reshape(-1, X.shape[-1]).std(0)
rows = []
if a.method == "CPT2s":
    X, Y, C = (np.ascontiguousarray(v[:, -a.window:]) for v in (X, Y, C))
    _cand = [(1, 0, 0), (2, 64, 0)] + ([(2, 64, a.flow)] if a.flow else [])
    if a.only:
        _wanted = a.only.split("-"); _cand = [t for t in _cand if ("FLOW" if t[2] else ("NL" if t[1] else "LIN")) in _wanted]
    for lags, nl, fl in _cand:
        zh, sec, npar = R7.run_cpt(X, Y, C, a.seed, steps=5 if a.quick else a.steps, bs=512, lr=1e-3, NClass=5, enc_regime=False,
                                   mech_lags=lags, mech_nl=nl, holdout=0.2, mech_flow=fl)
        rows.append(dict(method=f"CPT2_L{lags}NL{nl}" + (f"F{fl}" if fl else ""), seed=a.seed, window=a.window, steps=a.steps, split_loss=R7.run_cpt.split_loss,
                         split_elbo=getattr(R7.run_cpt, "split_elbo", float("nan")), sec=round(sec, 1),
                         **R7.eval_last(Y[:, -1], zh)))
    criterion = "split_elbo" if a.flow else "split_loss"   # with the flow candidate: held-out ELBO without free bits; otherwise held-out loss
    best = min(rows, key=lambda r: r[criterion]); rows.append(dict(best, method="CPT2s (selected)", selection_criterion=criterion))
else:
    # TDRL assumes a FIXED regime within the window; in NCTRL the regime changes within the window -> the last 3 steps (2 lags + 1)
    # and the regime of the last step are supplied (same window length as in R7).
    zh, sec, npar = R7.run_tdrl(np.ascontiguousarray(X[:, -3:]), np.ascontiguousarray(Y[:, -3:]), C[:, -1], a.seed,
                                epochs=1 if a.quick else a.tdrl_epochs, NClass=5, length=3)
    rows.append(dict(method="TDRL", seed=a.seed, tdrl_epochs=a.tdrl_epochs, sec=round(sec, 1), **R7.eval_last(Y[:, -1], zh)))
for r in rows:
    print(r, flush=True)
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "outputs", f"cpt_on_nctrl_generator_{a.method}{a.tag}_s{a.seed}.csv"), index=False)
