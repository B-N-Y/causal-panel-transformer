"""Literature baseline TDRL (Yao et al. 2022, "changing causal dynamics" module) on our synthetic panel.
Code: third_party/tdrl (weirayao/tdrl, MIT, commit 655ea52). Only the module classes are imported; the training loop is ours:
  - a stub replaces the old pytorch_lightning (1.2.7) dependency (LightningModule = nn.Module, log = no-op)
  - MODEL SELECTION: last epoch (main result). TDRL's own script selects the checkpoint by a val_mcc computed with the TRUE
    latents (oracle selection, optimistic leakage); for comparison, 'oracle_*' columns are reported as well.
Data mapping (first-order model, lag=1): sample i = window [z_{t-1}, z_t], [x_{t-1}, x_t], c = e_t
  (A_{e_t} and S_{e_t} govern the transition at time t). TDRL's NP transition prior conditions the innovation on a regime embedding.
Evaluation as for our estimators: raw MCC, affine R^2, affine MCC (rotation_diagnostics), Stage B (resolve_rotation: JD, ICA).
Output: outputs/tdrl{tag}.csv (one row per seed; all flags as columns).
"""
import os, sys, time, types, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))

ROOT = os.path.dirname(HERE)
for p in [os.path.join(HERE, "third_party", "tdrl")]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
from simulation import gen_panel, mcc, rotation_diagnostics, resolve_rotation, dim_g_S, tdrl_condition

# --- stubs for TDRL's import-time dependencies (third-party code unchanged) ---
sys.modules.setdefault("ipdb", types.ModuleType("ipdb"))
_pl = types.ModuleType("pytorch_lightning")
class _LM(nn.Module):
    def log(self, *a, **k):
        pass
_pl.LightningModule = _LM
sys.modules["pytorch_lightning"] = _pl
from LiLY.modules.change import TimeVaryingProcess   # noqa: E402


def make_windows(x, z, e):
    """(T, .) series -> (T-1, 2, .) windows; c = e_t (last step of the window)."""
    xt = np.stack([x[:-1], x[1:]], axis=1).astype(np.float32)
    yt = np.stack([z[:-1], z[1:]], axis=1).astype(np.float32)
    ct = e[1:].astype(np.int64)
    return xt, yt, ct


def encode(model, xt, ct, dev, bs=2048):
    """Posterior mean per window; the last step (z_t) is returned."""
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(xt), bs):
            x = torch.tensor(xt[i:i + bs], device=dev)
            _, mus, _ = model.forward({"xt": x, "yt": x, "ct": torch.tensor(ct[i:i + bs], device=dev)})
            mus = mus.reshape(x.shape[0], x.shape[1], -1)
            out.append(mus[:, -1].cpu().numpy())
    model.train()
    return np.concatenate(out)


def evaluate(zt, zh, e, E):
    """Standard metrics; the first 200 steps are discarded (burn-in)."""
    m, _ = mcc(zt[200:], zh[200:])
    r2a, maff = rotation_diagnostics(zt[200:], zh[200:])
    rr = resolve_rotation(zt[200:], zh[200:], e[200:], E)
    return dict(MCC=round(float(m), 4), r2_affine=round(r2a, 4), mcc_affine=round(maff, 4),
                mcc_jd=round(rr["mcc_jd"], 4), mcc_ica=round(rr["mcc_ica"], 4))


def run(seed=0, E=4, T=6000, k=4, D=12, varmode=None, bestcase=True, innov="t5", epochs=200, bs=64,
        lr=5e-4, beta=2e-3, gamma=2e-2, hidden=128, embed=2, eval_every=10, graph="fixed", spurious=True, obs_noise=0.05, kmodel=None, conf_mode="ar1", savez=None):
    x, z, e, meta = gen_panel(k=k, E=E, T=T, D=D, seed=seed, nonlinear=True, bestcase=bestcase,
                              innov=innov, varmode=varmode, graph=graph, spurious=spurious, obs_noise=obs_noise,
                              conf_mode=conf_mode)
    kmodel = kmodel or k
    xt, yt, ct = make_windows(x, z, e)
    zt_true, e_t = z[1:], e[1:]
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = TimeVaryingProcess(input_dim=D, length=2, z_dim=kmodel, lag=1, nclass=E, hidden_dim=hidden,
                               embedding_dim=embed, trans_prior="NP", lr=lr, infer_mode="F", beta=beta,
                               gamma=gamma, decoder_dist="gaussian", correlation="Pearson").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    X, Y, C = (torch.tensor(a, device=dev) for a in (xt, yt, ct))
    n = len(X); t0 = time.time()
    best = dict(mcc=-1.0, ep=-1, res=None)          # oracle selection (TDRL's own rule), for comparison only
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            loss = model.training_step({"xt": X[idx], "yt": Y[idx], "ct": C[idx]}, i)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            zh = encode(model, xt, ct, dev)
            m_now, _ = mcc(zt_true[200:], zh[200:])
            if m_now > best["mcc"]:
                best = dict(mcc=float(m_now), ep=ep + 1, res=evaluate(zt_true, zh, e_t, E))
            print(f"  seed{seed} ep{ep+1:4d} loss={tot/n:.4f} mcc={m_now:.3f}", flush=True)
    zh = encode(model, xt, ct, dev)
    res = evaluate(zt_true, zh, e_t, E)
    _v = zh.var(0); res.update(active_units=int((_v > 0.01).sum()), au_var=";".join(f"{v:.3g}" for v in np.sort(_v)[::-1]))
    if savez:                                   # save latents for Stage B and edge analyses
        os.makedirs(os.path.join(HERE, "outputs", "z"), exist_ok=True)
        np.savez_compressed(os.path.join(HERE, "outputs", "z", f"{savez}_s{seed}.npz"), zt=zt_true.astype(np.float32),
                            zh=zh.astype(np.float32), e=e_t.astype(np.int64), A=np.stack(meta["A_list"]).astype(np.float32),
                            conf=meta["conf"][1:].astype(np.float32))
    # 5-dimensional scoring: the confounder (conf) also counts as a true latent (only if model dimension > k and spurious columns are on)
    if spurious and kmodel > k:
        z5 = np.column_stack([zt_true, meta["conf"][1:]])
        res.update({f"z5_{a_}": b_ for a_, b_ in evaluate(z5, zh, e_t, E).items()})
    row = dict(method="TDRL", seed=seed, E=E, T=T, k=k, D=D, varmode=varmode, bestcase=bestcase, innov=innov,
               graph=graph, spurious=spurious, obs_noise=obs_noise, kmodel=kmodel, conf_mode=conf_mode, dim_gS=dim_g_S(meta["A_list"]),
               tdrl_rank_ok=tdrl_condition(meta)[0], min_det_A=round(tdrl_condition(meta)[1], 4),
               epochs=epochs, bs=bs, lr=lr, beta=beta, gamma=gamma, hidden=hidden, embed=embed,
               selection="last_epoch", **res, final_loss=round(tot / n, 4), sec=round(time.time() - t0, 1),
               params=sum(p.numel() for p in model.parameters()),
               oracle_ep=best["ep"], **{f"oracle_{a}": b for a, b in (best["res"] or {}).items()},
               tdrl_commit="655ea52")
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--varmode", type=str, default=None, choices=[None, "none", "scale", "mean", "both"])
    ap.add_argument("--nobest", action="store_true", help="bestcase=False")
    ap.add_argument("--innov", type=str, default="t5")
    ap.add_argument("--graph", type=str, default="fixed", choices=["fixed", "random"])
    ap.add_argument("--obs_noise", type=float, default=0.05)
    ap.add_argument("--kmodel", type=int, default=None, help="latent dimension of the estimator (default: k)")
    ap.add_argument("--conf_mode", type=str, default="ar1", choices=["ar1", "iid"])
    ap.add_argument("--savez", type=str, default=None, help="file prefix for saved latents (outputs/z/)")
    ap.add_argument("--nospurious", action="store_true", help="disable the spurious (confounder) columns: x = g(z) + noise")
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 seed, 3 epochs, T=800")
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.T = 1, 3, 800
    rows = []
    for s in range(a.seed0, a.seed0 + a.seeds):
        r = run(seed=s, E=a.E, T=a.T, varmode=a.varmode, bestcase=not a.nobest, innov=a.innov,
                epochs=a.epochs, eval_every=1 if a.quick else 10, graph=a.graph, spurious=not a.nospurious, obs_noise=a.obs_noise,
                kmodel=a.kmodel, conf_mode=a.conf_mode, savez=a.savez)
        rows.append(r); print(r, flush=True)
        df = pd.DataFrame(rows)
        os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
        out = os.path.join(HERE, "outputs", f"tdrl{a.tag}.csv")
        df.to_csv(out, index=False)          # write after every seed so that a timeout loses nothing
    print(f"Wrote {out}")
    print(df[["seed", "MCC", "r2_affine", "mcc_affine", "mcc_jd", "mcc_ica", "oracle_MCC"]].to_string(index=False))
