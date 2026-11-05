"""R7, reverse test: CPT and TDRL on TDRL's OWN generator (Yao et al. 2022, "pnl_change"; control for home advantage).
Generator: the logic of third_party/tdrl/LiLY/tools/gen_dataset.py :: pnl_change_gaussian_ts (MIT, 655ea52), reproduced exactly
in memory without writing files; the paper's setting (S2.1.1): 8 latents, 2 lags, 20 domains x 7500 windows, leaky-ReLU (NOT linear)
transition, only 2 edge weights change across domains, fixed noise scale 0.1, mixing by 2 leaky-ReLU layers + orthogonal matrices.
Each sample is an independent 3-step window (z_{t-2}, z_{t-1} ~ N(0,1) standardised, z_t from the transition).
Evaluation: Stage A only (raw MCC, affine R^2, affine MCC) at the last step (z_t); because the transition is nonlinear, our linear
Stage B steps (JD/ICA on AR residuals) are not defined here.
CPT is MISSPECIFIED here (it assumes linear latent dynamics with one lag); this is the purpose of the test.
Output: outputs/tdrl_generator_{method}{tag}.csv
"""
import os, sys, time, argparse
import numpy as np, pandas as pd, torch
from scipy.stats import ortho_group

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))

import tdrl_runner as TD                                 # stubs + TimeVaryingProcess + path setup
from simulation import mcc, rotation_diagnostics
from cpt_model import CPT
from cpt_losses import cpt_loss


def _lrelu(x, s):
    return np.maximum(s * x, x)


def gen_tdrl_change(seed, NClass=20, per_class=7500, latent=8, lags=2, Nlayer=3, noise_scale=0.1, slope=0.2):
    """The 'source' branch of pnl_change_gaussian_ts, with a seeded RNG (original: global np.random)."""
    rng = np.random.default_rng(seed)
    conds = []
    for _ in range(10000):
        A = rng.uniform(1, 2, (latent, latent)); A /= np.sqrt((A ** 2).sum(0, keepdims=True)); conds.append(np.linalg.cond(A))
    thr = np.percentile(conds, 25)

    def uniform_mat():                       # generateUniformMat: redraw until the condition number is below the threshold
        while True:
            A = rng.uniform(-1, 1, (latent, latent)); A /= np.sqrt((A ** 2).sum(0, keepdims=True))
            if np.linalg.cond(A) < thr:
                return A
    trans = [uniform_mat() for _ in range(lags)][::-1]
    mixing = [ortho_group.rvs(latent, random_state=int(rng.integers(1 << 31))) for _ in range(Nlayer - 1)]
    pairs = [(1, 2), (3, 4)]
    w = rng.uniform(-1, 1, (NClass, len(pairs)))

    def mix(y):
        m = y.copy()
        for M in mixing:
            m = _lrelu(m, slope) @ M
        return m
    Y, X, C = [], [], []
    for j in range(NClass):
        yl = rng.normal(0, 1, (per_class, lags, latent))
        yl = (yl - yl.mean(0, keepdims=True)) / yl.std(0, keepdims=True)
        for p, (a, b) in enumerate(pairs):
            trans[0][a, b] = w[j, p]
        yt = rng.normal(0, noise_scale, (per_class, latent))
        for l in range(lags):
            yt += _lrelu(yl[:, l, :] @ trans[l], slope)
        yt = _lrelu(yt, slope)
        y = np.concatenate([yl, yt[:, None, :]], 1)
        Y.append(y); X.append(mix(y)); C.append(np.full(per_class, j))
    return (np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32), np.concatenate(C).astype(np.int64))


def eval_last(zt, zh):
    m, _ = mcc(zt, zh); r2, ma = rotation_diagnostics(zt, zh)
    return dict(MCC=round(float(m), 4), r2_affine=round(r2, 4), mcc_affine=round(ma, 4))


def run_cpt(X, Y, C, seed, steps=3000, bs=4096, lr=2e-3, NClass=20, enc_regime=True, mech_lags=1, mech_nl=0, holdout=0.0, all_steps=False, mech_flow=0):
    """holdout > 0: this fraction of the windows is held out of training; the held-out ELBO loss is stored (for model
    selection) in run_cpt.split_loss, and the held-out ELBO without free bits in run_cpt.split_elbo.
    all_steps=True: returns the posterior means of all steps (N, L, k) (default: last step only)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(seed)
    k, D = Y.shape[-1], X.shape[-1]
    m = CPT(d_in=D, k=k, d_model=64, n_heads=4, depth=4, n_regimes=NClass, recursive=True, enc_regime=enc_regime,
            mech_lags=mech_lags, mech_nl=mech_nl, mech_flow=mech_flow).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.tensor(X, device=dev)
    Et = torch.tensor(C if C.ndim == 2 else np.repeat(C[:, None], X.shape[1], 1), device=dev)   # 2-D C: time-varying regime
    g = torch.Generator().manual_seed(seed); t0 = time.time()
    perm = np.random.default_rng(seed + 7).permutation(len(X)); n_te = int(len(X) * holdout)
    tr_ix = torch.tensor(perm[n_te:] if n_te else np.arange(len(X)), device=dev); te_ix = perm[:n_te]
    for st in range(steps):
        idx = tr_ix[torch.randint(0, len(tr_ix), (bs,), generator=g).to(dev)] if n_te else torch.randint(0, len(X), (bs,), generator=g).to(dev)
        out = m(Xt[idx], Et[idx]); loss, _ = cpt_loss(out, Xt[idx], Et[idx], m.mech, obs_logvar=m.obs_logvar)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    zh = []; split_loss = np.nan
    if n_te:
        m.eval()
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            mu = m(Xt[i:i + 8192], Et[i:i + 8192], sample=False)["mu"]
            zh.append((mu if all_steps else mu[:, -1]).cpu().numpy())
        if n_te:
            ls = [float(cpt_loss(m(Xt[j], Et[j]), Xt[j], Et[j], m.mech, obs_logvar=m.obs_logvar)[0]) * len(j)
                  for j in (torch.tensor(te_ix[i:i + 8192], device=dev) for i in range(0, n_te, 8192))]
            split_loss = sum(ls) / n_te
            torch.manual_seed(seed + 99)                  # held-out ELBO without free bits (fair comparison with the flow candidate)
            le = [float(cpt_loss(m(Xt[j], Et[j]), Xt[j], Et[j], m.mech, obs_logvar=m.obs_logvar, free_bits=0.0)[0]) * len(j)
                  for j in (torch.tensor(te_ix[i:i + 8192], device=dev) for i in range(0, n_te, 8192))]
            run_cpt.split_elbo = sum(le) / n_te
    run_cpt.split_loss = split_loss
    return np.concatenate(zh), time.time() - t0, sum(p.numel() for p in m.parameters())


def run_tdrl(X, Y, C, seed, epochs=50, bs=512, NClass=20, on_checkpoint=None, length=3, ckpt=None, budget_sec=None):
    """ckpt: None (default) or a path. With a path, the model, the optimizer and the RNG states (data shuffling, torch CPU and CUDA) are
    written to this file after every epoch, and training resumes from the file if it exists. budget_sec: wall-clock budget of this
    call; if the next epoch would exceed it, training stops and (None, sec, npar) is returned, and the next job of the chain resumes.
    The state is in the dictionary run_tdrl.state (epochs_done, n_parts, total_sec)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(seed)
    k, D = Y.shape[-1], X.shape[-1]
    m = TD.TimeVaryingProcess(input_dim=D, length=length, z_dim=k, lag=2, nclass=NClass, hidden_dim=128, embedding_dim=2,
                              trans_prior="NP", lr=5e-4, infer_mode="F", beta=2e-3, gamma=2e-2,
                              decoder_dist="gaussian", correlation="Pearson").to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    Xt, Yt, Ct = (torch.tensor(a, device=dev) for a in (X, Y, C))
    g = torch.Generator().manual_seed(seed); t0 = time.time()
    ep0, sec0, n_parts = 0, 0.0, 1
    if ckpt is not None and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        m.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); g.set_state(st["g"].cpu()); torch.set_rng_state(st["cpu_rng"].cpu())
        if torch.cuda.is_available() and st.get("cuda_rng") is not None:
            try:
                torch.cuda.set_rng_state_all([r.cpu() for r in st["cuda_rng"]])
            except Exception as ex:                                            # noqa: BLE001
                print("  WARNING: could not restore the CUDA RNG state:", ex, flush=True)
        ep0, sec0, n_parts = st["ep"], st["sec"], st["n_parts"] + 1
        print(f"  RESUME: {ckpt} epoch {ep0}/{epochs}, previous training time {sec0:.0f} s, part {n_parts}", flush=True)
    m.train(); ep_times = []
    for ep in range(ep0, epochs):
        te = time.time()
        perm = torch.randperm(len(X), generator=g).to(dev)
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            loss = m.training_step({"xt": Xt[idx], "yt": Yt[idx], "ct": Ct[idx]}, i)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  TDRL seed{seed} ep{ep+1} loss={float(loss):.4f}", flush=True)
        ep_times.append(time.time() - te)
        if ckpt is not None:                              # atomic write: temporary file, then rename
            st = dict(model=m.state_dict(), opt=opt.state_dict(), g=g.get_state(), cpu_rng=torch.get_rng_state(),
                      cuda_rng=(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                      ep=ep + 1, sec=sec0 + time.time() - t0, n_parts=n_parts, epochs_target=epochs, seed=seed, bs=bs)
            torch.save(st, ckpt + ".tmp"); os.replace(ckpt + ".tmp", ckpt)
        if on_checkpoint is not None and (ep + 1) % 5 == 0 and ep + 1 < epochs:
            on_checkpoint(_encode_tdrl(m, Xt, Yt, Ct, k, length), ep + 1, sec0 + time.time() - t0); m.train()
        if budget_sec is not None and ep + 1 < epochs and (time.time() - t0) + 1.15 * max(ep_times) > budget_sec:
            run_tdrl.state = dict(epochs_done=ep + 1, n_parts=n_parts, total_sec=sec0 + time.time() - t0)
            print(f"  BUDGET: stopped after epoch {ep+1}/{epochs}; the next job of the chain resumes", flush=True)
            return None, sec0 + time.time() - t0, sum(p.numel() for p in m.parameters())
    run_tdrl.state = dict(epochs_done=max(epochs, ep0), n_parts=n_parts, total_sec=sec0 + time.time() - t0)
    return _encode_tdrl(m, Xt, Yt, Ct, k, length), sec0 + time.time() - t0, sum(p.numel() for p in m.parameters())


def _encode_tdrl(m, Xt, Yt, Ct, k, length=3):
    m.eval(); zh = []
    with torch.no_grad():
        for i in range(0, len(Xt), 8192):
            _, mus, _ = m.forward({"xt": Xt[i:i + 8192], "yt": Yt[i:i + 8192], "ct": Ct[i:i + 8192]})
            zh.append(mus.reshape(-1, length, k)[:, -1].cpu().numpy())
    return np.concatenate(zh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["CPT", "CPT2", "TDRL"], required=True)
    ap.add_argument("--holdout", type=float, default=0.0)
    ap.add_argument("--lags", type=int, default=1); ap.add_argument("--nl", type=int, default=0, help="hidden width of the CPT2 nonlinear transition (0 = none)")
    ap.add_argument("--steps", type=int, default=20000, help="number of CPT2 updates (batch 512, lr 1e-3; regime enters only the prior)")
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--per_class", type=int, default=7500); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=50, help="number of TDRL epochs (paper: 50)")
    ap.add_argument("--tag", type=str, default="")
    # split training: without these flags the behavior is unchanged
    ap.add_argument("--ckpt", action="store_true", help="TDRL: resume from outputs/ckpt/tdrl{tag}_s{seed}.pt")
    ap.add_argument("--budget_hours", type=float, default=None, help="TDRL: train for at most this many hours in this job (below the time limit of the cluster)")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    rows = []
    for s in range(a.seed0, a.seed0 + a.seeds):
        pc = 300 if a.quick else a.per_class
        X, Y, C = gen_tdrl_change(s, per_class=pc)
        if a.method == "CPT":
            zh, sec, npar = run_cpt(X, Y, C, s, steps=5 if a.quick else 3000)
        elif a.method == "CPT2":     # R11: R9 recipe (mini-batches, many updates, no regime input to the encoder)
            zh, sec, npar = run_cpt(X, Y, C, s, steps=5 if a.quick else a.steps, bs=512, lr=1e-3, enc_regime=False,
                                    mech_lags=a.lags, mech_nl=a.nl, holdout=a.holdout)
        else:
            def _ck(zh_, ep_, sec_):    # intermediate result, kept in case of a time-out
                rr = dict(method=a.method, seed=s, epochs=ep_, intermediate=True, generator="tdrl_pnl_change", NClass=20, per_class=pc,
                          latent=8, lags=2, **eval_last(Y[:, -1], zh_), sec=round(sec_, 1))
                pd.DataFrame(rows + [rr]).to_csv(os.path.join(HERE, "outputs", f"tdrl_generator_{a.method}{a.tag}.csv"), index=False)
            ck = None
            if a.ckpt:
                os.makedirs(os.path.join(HERE, "outputs", "ckpt"), exist_ok=True)
                ck = os.path.join(HERE, "outputs", "ckpt", f"tdrl{a.tag}_s{s}.pt")
            zh, sec, npar = run_tdrl(X, Y, C, s, epochs=(2 if a.ckpt else 1) if a.quick else a.epochs, on_checkpoint=_ck, ckpt=ck,
                                     budget_sec=None if a.budget_hours is None else a.budget_hours * 3600)
            if zh is None:                                # budget used up: the state is written, the next job of the chain resumes
                d_ = run_tdrl.state
                pd.DataFrame([dict(method=a.method, seed=s, epochs=a.epochs, epochs_done=d_["epochs_done"], n_parts=d_["n_parts"],
                                   total_sec=round(d_["total_sec"], 1), budget_hours=a.budget_hours, completed=False)]).to_csv(
                    os.path.join(HERE, "outputs", f"tdrl_generator_{a.method}{a.tag}_state_s{s}.csv"), index=False)
                print(f"PARTIAL: seed {s} epoch {d_['epochs_done']}/{a.epochs}", flush=True)
                continue
        r = dict(method=a.method, seed=s, epochs=(a.epochs if a.method == "TDRL" else None), steps=(a.steps if a.method == "CPT2" else None), mech_lags=a.lags, mech_nl=a.nl, holdout=a.holdout,
                 split_loss=(getattr(run_cpt, "split_loss", np.nan) if a.method == "CPT2" else np.nan), generator="tdrl_pnl_change", NClass=20, per_class=pc, latent=8, lags=2,
                 **eval_last(Y[:, -1], zh), sec=round(sec, 1), params=npar)
        if a.method == "TDRL" and a.ckpt:                 # extra columns only with split training
            r.update(ckpt=True, budget_hours=a.budget_hours, n_parts=run_tdrl.state["n_parts"], epochs_done=run_tdrl.state["epochs_done"], completed=True)
        rows.append(r); print(r, flush=True)
        out = os.path.join(HERE, "outputs", f"tdrl_generator_{a.method}{a.tag}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
    if rows:
        print(f"WROTE {out}")
