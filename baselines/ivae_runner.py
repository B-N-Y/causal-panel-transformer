"""Other literature baselines from the tdrl repository on our synthetic panel: iVAE, TCL, BetaVAE, SlowVAE.
Code: third_party/tdrl/LiLY/baselines (MIT, 655ea52). Model classes are used unchanged; the training loop is ours (same stubs
as tdrl_runner, same windowing [x_{t-1}, x_t], c = e_t, same evaluation, last-epoch selection).
  iVAE   : auxiliary variable u = regime index (Khemakhem et al. 2020; posterior mean g)
  TCL    : classes = regimes (Hyvarinen & Morioka 2016; features |h|); undefined for E=1 -> skipped
  BetaVAE: no auxiliary variable (beta=8; no identifiability claim, lower reference)
  SlowVAE: consecutive pairs (Klindt et al. 2021; Laplace transition prior)
Output: outputs/baseline_{method}{tag}.csv ; with --savez, latents are saved to outputs/z/.
"""
import os, sys, time, argparse
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))

import tdrl_runner as T                                  # stubs + path set-up + make_windows / evaluate
from simulation import gen_panel, dim_g_S
from LiLY.baselines.iVAE.model import iVAE            # noqa: E402
from LiLY.baselines.TCL.model import TCL              # noqa: E402
from LiLY.baselines.BetaVAE.model import BetaVAE      # noqa: E402
from LiLY.baselines.SlowVAE.model import SlowVAE      # noqa: E402
import LiLY.baselines.iVAE.net as _ivnet             # noqa: E402  (module-level constant device='cuda:1')

HP = {  # from the toy_nonlinear_ts_* configurations in the tdrl repository
    "iVAE": dict(lr=1e-3), "TCL": dict(lr=1e-3), "BetaVAE": dict(lr=1e-4, beta=8),
    "SlowVAE": dict(lr=1e-4, beta=1, gamma=10, rate_prior=6)}


def build(method, D, k, E, dev, hidden=128):
    h = HP[method]
    if method == "iVAE":
        _ivnet.device = dev        # the third-party code hard-codes 'cuda:1', which fails on a single GPU -> override at run time
        return iVAE(input_dim=D, z_dim=k, hidden_dim=hidden, lr=h["lr"], correlation="Pearson")
    if method == "TCL":
        return TCL(input_dim=D, z_dim=k, nclass=E, hidden_dim=hidden, lr=h["lr"], correlation="Pearson")
    if method == "BetaVAE":
        return BetaVAE(input_dim=D, z_dim=k, hidden_dim=hidden, beta=h["beta"], beta1=0.9, beta2=0.999,
                       lr=h["lr"], correlation="Pearson")
    return SlowVAE(input_dim=D, z_dim=k, hidden_dim=hidden, beta=h["beta"], gamma=h["gamma"], lr=h["lr"],
                   beta1=0.9, beta2=0.999, rate_prior=h["rate_prior"], correlation="Pearson")


def encode(method, model, x, c, dev, bs=4096):
    """Single-step (T, D) observations -> (T, kmodel) representation (the quantity each method uses in its own validation)."""
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.tensor(x[i:i + bs], device=dev)
            if method == "iVAE":
                u = torch.tensor(c[i:i + bs], device=dev, dtype=torch.float32).unsqueeze(-1)
                _, (g, _), _, _ = model.model.forward(xb, u); out.append(g)
            elif method == "TCL":
                _, feats = model.model(xb); out.append(feats)
            else:
                _, mu, _ = model.net(xb); out.append(mu)
    model.train()
    return torch.cat(out).cpu().numpy()


def run(method, seed=0, E=4, T_=6000, k=4, D=12, varmode=None, bestcase=True, innov="t5", epochs=200, bs=64,
        graph="fixed", spurious=True, obs_noise=0.05, kmodel=None, conf_mode="ar1", savez=None):
    x, z, e, meta = gen_panel(k=k, E=E, T=T_, D=D, seed=seed, nonlinear=True, bestcase=bestcase, innov=innov,
                              varmode=varmode, graph=graph, spurious=spurious, obs_noise=obs_noise, conf_mode=conf_mode)
    kmodel = kmodel or k
    xt, yt, ct = T.make_windows(x, z, e)
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build(method, D, kmodel, E, dev).to(dev)
    opt = model.configure_optimizers()
    opt = opt[0][0] if isinstance(opt, tuple) else (opt[0] if isinstance(opt, list) else opt)
    X, Y, C = (torch.tensor(a, device=dev) for a in (xt, yt, ct.astype(np.float32)))
    n = len(X); t0 = time.time(); g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            loss = model.training_step({"s1": {"xt": X[idx], "yt": Y[idx], "ct": C[idx]}}, i)
            loss = loss.mean() if loss.dim() else loss
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        if (ep + 1) % 50 == 0:
            print(f"  {method} seed{seed} ep{ep+1} loss={tot/n:.4f}", flush=True)
    zh = encode(method, model, x[1:], e[1:], dev)
    zt_true, e_t = z[1:], e[1:]
    res = T.evaluate(zt_true, zh, e_t, E)
    if spurious and kmodel > k:
        z5 = np.column_stack([zt_true, meta["conf"][1:]])
        res.update({f"z5_{a_}": b_ for a_, b_ in T.evaluate(z5, zh, e_t, E).items()})
    if savez:
        os.makedirs(os.path.join(HERE, "outputs", "z"), exist_ok=True)
        np.savez_compressed(os.path.join(HERE, "outputs", "z", f"{savez}_s{seed}.npz"), zt=zt_true.astype(np.float32),
                            zh=zh.astype(np.float32), e=e_t.astype(np.int64), A=np.stack(meta["A_list"]).astype(np.float32),
                            conf=meta["conf"][1:].astype(np.float32))
    return dict(method=method, seed=seed, E=E, T=T_, k=k, D=D, varmode=varmode, bestcase=bestcase, innov=innov,
                graph=graph, spurious=spurious, obs_noise=obs_noise, kmodel=kmodel, conf_mode=conf_mode,
                dim_gS=dim_g_S(meta["A_list"]), epochs=epochs, bs=bs, selection="last_epoch", **res,
                final_loss=round(tot / n, 4), sec=round(time.time() - t0, 1),
                params=sum(p.numel() for p in model.parameters()), tdrl_commit="655ea52", **{f"hp_{a}": b for a, b in HP[method].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", type=str, default="iVAE-TCL-BetaVAE-SlowVAE", help="separated by hyphens")
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--varmode", type=str, default=None, choices=[None, "none", "scale", "mean", "both"])
    ap.add_argument("--nobest", action="store_true"); ap.add_argument("--innov", type=str, default="t5")
    ap.add_argument("--graph", type=str, default="fixed", choices=["fixed", "random"])
    ap.add_argument("--nospurious", action="store_true"); ap.add_argument("--obs_noise", type=float, default=0.05)
    ap.add_argument("--kmodel", type=int, default=None)
    ap.add_argument("--conf_mode", type=str, default="ar1", choices=["ar1", "iid"])
    ap.add_argument("--savez", type=str, default=None); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.T = 1, 2, 800
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    for meth in a.methods.split("-"):
        if meth == "TCL" and a.E < 2:
            print("TCL is undefined for E=1 (single class): skipped"); continue
        rows = []
        for s in range(a.seed0, a.seed0 + a.seeds):
            r = run(meth, seed=s, E=a.E, T_=a.T, varmode=a.varmode, bestcase=not a.nobest, innov=a.innov, epochs=a.epochs,
                    graph=a.graph, spurious=not a.nospurious, obs_noise=a.obs_noise, kmodel=a.kmodel, conf_mode=a.conf_mode,
                    savez=(a.savez and f"{a.savez}_{meth}"))
            rows.append(r); print({kk: r[kk] for kk in ("method", "seed", "MCC", "r2_affine", "mcc_affine", "mcc_jd", "mcc_ica")}, flush=True)
            out = os.path.join(HERE, "outputs", f"baseline_{meth}{a.tag}.csv")
            pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Wrote {out}")
