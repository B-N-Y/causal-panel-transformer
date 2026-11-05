"""Stage A encoders (common interface; the same code is used on synthetic and real data).
fit(method, x, e, k, seed, ...) -> (zh: T x k, info: {"train_loss", "split_loss", "time"})
  x : (T, D) observations (time ordered; in a panel, unit windows may be pooled: see the block / blocks arguments)
  e : (T,) observed regime (0..E-1)
Methods: "iVAE" (Khemakhem et al. 2020; u = regime; single time step), "TDRL" (Yao et al. 2022; [x_{t-1}, x_t] windows),
"CPT2" and "CPT2s" (Causal Panel Transformer).
Held-out loss: the LAST 20% of the time axis is not used for training; dimension selection (dimension.py) uses this loss.
"""
import os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))

import tdrl_runner as _T                                    # stubs, TimeVaryingProcess, path set-up
from LiLY.baselines.iVAE.model import iVAE as _iVAE      # noqa: E402
import LiLY.baselines.iVAE.net as _ivnet                 # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _split(T, frac=0.2):
    cut = int(T * (1 - frac)); return np.arange(cut), np.arange(cut, T)


def fit_ivae(x, e, k, seed=0, epochs=200, bs=64, lr=1e-3, hidden=128, holdout=0.2):
    torch.manual_seed(seed); np.random.seed(seed); _ivnet.device = DEV
    m = _iVAE(input_dim=x.shape[1], z_dim=k, hidden_dim=hidden, lr=lr, correlation="Pearson").to(DEV)
    opt = torch.optim.AdamW(m.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    tr, te = _split(len(x), holdout)
    X = torch.tensor(x, dtype=torch.float32, device=DEV); U = torch.tensor(e, dtype=torch.float32, device=DEV).unsqueeze(-1)
    g = torch.Generator().manual_seed(seed); t0 = time.time(); trt = torch.tensor(tr, device=DEV)
    for ep in range(epochs):
        perm = trt[torch.randperm(len(tr), generator=g).to(DEV)]
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            elbo, _ = m.model.elbo(X[idx], U[idx]); loss = -elbo
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        ltr = -float(m.model.elbo(X[tr], U[tr])[0]); lte = -float(m.model.elbo(X[te], U[te])[0])
        _, (gm, _), _, _ = m.model.forward(X, U)
    return gm.cpu().numpy(), dict(train_loss=ltr, split_loss=lte, time=time.time() - t0)


def fit_tdrl(x, e, k, seed=0, epochs=200, bs=64, lr=5e-4, holdout=0.2, block=None, blocks=None, max_steps=None, loss_max=None):
    """block=n (short panel): every n consecutive rows form one unit; [x_{t-1}, x_t] pairs crossing a unit boundary are
    excluded from training and from the loss. The following options default to None (no effect):
    blocks=[n_1, n_2, ...]: UNBALANCED panel (unit lengths, sum = T); pairs crossing a unit boundary are excluded.
    max_steps=n: training stops after n updates (epochs becomes an upper bound).
    loss_max=n: the training / held-out loss is computed on the first n valid pairs only (the Jacobian of the NP prior is
    quadratic in the batch size, so the full loss on very many pairs is slow)."""
    torch.manual_seed(seed); np.random.seed(seed)
    E = int(e.max()) + 1
    xt = np.stack([x[:-1], x[1:]], 1).astype(np.float32); ct = e[1:].astype(np.int64)
    m = _T.TimeVaryingProcess(input_dim=x.shape[1], length=2, z_dim=k, lag=1, nclass=E, hidden_dim=128, embedding_dim=2,
                              trans_prior="NP", lr=lr, infer_mode="F", beta=2e-3, gamma=2e-2, decoder_dist="gaussian",
                              correlation="Pearson").to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    tr, te = _split(len(xt), holdout)
    if block is not None:                                  # pair i = (t=i, t=i+1); invalid if i+1 is the first row of a unit
        valid = (np.arange(1, len(x)) % block) != 0
        tr, te = tr[valid[tr]], te[valid[te]]
    if blocks is not None:                                 # unbalanced panel: same rule, unit starts taken from blocks
        bl = np.asarray(blocks, int); assert bl.sum() == len(x)
        valid = np.ones(len(x) - 1, bool); valid[np.cumsum(bl)[:-1] - 1] = False
        tr, te = tr[valid[tr]], te[valid[te]]
    X = torch.tensor(xt, device=DEV); C = torch.tensor(ct, device=DEV)
    g = torch.Generator().manual_seed(seed); t0 = time.time(); trt = torch.tensor(tr, device=DEV)
    n_steps = 0
    for ep in range(epochs):
        perm = trt[torch.randperm(len(tr), generator=g).to(DEV)]
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            loss = m.training_step({"xt": X[idx], "yt": X[idx], "ct": C[idx]}, i)
            opt.zero_grad(); loss.backward(); opt.step()
            n_steps += 1
            if max_steps is not None and n_steps >= max_steps:
                break
        if max_steps is not None and n_steps >= max_steps:
            break
    if loss_max is not None:
        tr, te = tr[:loss_max], te[:loss_max]
    def _loss(ix):
        tot = 0.0
        for i in range(0, len(ix), 256):                  # chunks of 256 bound the Jacobian memory for large k, D; the value is unchanged
            j = torch.tensor(ix[i:i + 256], device=DEV)
            tot += float(m.training_step({"xt": X[j], "yt": X[j], "ct": C[j]}, 0)) * len(j)
        return tot / len(ix)
    ltr, lte = _loss(tr), _loss(te)          # the NP prior needs gradients for its Jacobian; do NOT use no_grad
    zh = _T.encode(m, xt, ct, DEV)                         # z_t per window (last step) -> T-1 rows
    zh = np.vstack([zh[:1], zh])                           # repeat the first row for t=0 (keeps the time axis)
    info = dict(train_loss=ltr, split_loss=lte, time=time.time() - t0)
    if max_steps is not None or blocks is not None or loss_max is not None:   # extra keys only with these options
        info.update(steps=n_steps, loss_max=loss_max)
    return zh, info


def fit(method, x, e, k, seed=0, **kw):
    return (fit_ivae if method == "iVAE" else fit_tdrl)(x, e, k, seed=seed, **kw)


# =====================================================================================================
# CPT-2: Transformer encoder with a dynamic prior, plus an optional white block
#   - the encoder is the CPT Transformer; the REGIME IS NOT GIVEN TO THE ENCODER (the regime enters only the prior;
#     feeding it to the encoder was found to be harmful)
#   - MINI-BATCH window training with many updates (insufficient training is the main failure mode)
#   - white block: the transition rows/columns of the last k_white components are fixed at ZERO by a mask -> no link to
#     the past or to the other components (a component without temporal structure; for spurious-column noise / an iid
#     confounder)
# =====================================================================================================
def fit_cpt2(x, e, k, seed=0, k_white=0, steps=20000, L=128, bs=64, lr=1e-3, holdout=0.2, depth=4, d_model=64,
             epochs=None, block=None, mech_lags=1, mech_nl=0, blocks=None, enc_bidir=False, enc_type="attn", mech_flow=0,
             dec="mlp", record_windows=False, prior_family="gauss", w_rc=0.0):
    """w_rc: weight of the regime-contrast term of cpt_loss in TRAINING (default 0: unchanged; Gaussian one-lag prior only).
    prior_family: "gauss" (default) | "student" (Student-t transition prior with nu = 5; KL by Monte Carlo); passed to the
    model only when it differs from the default, so the default call is unchanged.
    record_windows: if True, "split_elbo_windows" (held-out ELBO without free bits, one value per window; its mean equals
    split_elbo) is added to the info dict. Same call, same RNG consumption; with the default (False) nothing changes.
    dec: "mlp" (default, k->64->D) | "linear" (single affine layer k->D, linear measurement family).
    If dec != "mlp", the key "dec" is added to the info dict.
    block=None: a single long series (default). block=n (panel): rows are unit-major, every n consecutive rows form one
    unit; windows do NOT cross unit boundaries, L = min(L, n); the held-out loss uses the last holdout fraction of UNITS.
    blocks=[n_1, n_2, ...] (UNBALANCED panel): unit lengths (sum = T); units shorter than L are not used for training,
    window starts are sampled in proportion to (n_f - L + 1); encoding is done in chunks within each unit."""
    from cpt_model import CPT
    from cpt_losses import cpt_loss
    if w_rc > 0 and (mech_lags > 1 or mech_nl or mech_flow or prior_family != "gauss"):
        raise ValueError("w_rc: Gaussian one-lag prior only")
    torch.manual_seed(seed); np.random.seed(seed)
    E = int(e.max()) + 1; T = len(x)
    m = CPT(d_in=x.shape[1], k=k, d_model=d_model, n_heads=4, depth=depth, n_regimes=E, recursive=True,
            cross_unit=False, enc_regime=False, mech_lags=mech_lags, mech_nl=mech_nl,
            enc_bidir=enc_bidir, enc_type=enc_type, mech_flow=mech_flow, **({"dec": dec} if dec != "mlp" else {}),
            **({"prior_family": prior_family} if prior_family != "gauss" else {})).to(DEV)
    if k_white > 0:
        mask = np.ones((E, k, k), np.float32); mask[:, k - k_white:, :] = 0; mask[:, :, k - k_white:] = 0
        m.mech.mask = torch.tensor(mask, device=DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    cut = int(T * (1 - holdout))
    X = torch.tensor(x, dtype=torch.float32, device=DEV); Ei = torch.tensor(e, dtype=torch.long, device=DEV)
    g = torch.Generator().manual_seed(seed); t0 = time.time()
    if block is not None:
        L = min(L, block); nF = T // block; cutF = max(1, int(nF * (1 - holdout))); cut = cutF * block
    if blocks is not None:
        bl = np.asarray(blocks, int); assert bl.sum() == T
        off = np.concatenate([[0], np.cumsum(bl)[:-1]]); nF = len(bl); cutF = max(1, int(nF * (1 - holdout)))
        cut = int(off[cutF]) if cutF < nF else T
        eligible = np.where(bl[:cutF] >= L)[0]; wts = (bl[eligible] - L + 1).astype(float); wts /= wts.sum()
        rs = np.random.default_rng(seed)
    starts_max = cut - L
    for st in range(steps):
        if blocks is not None:
            f = rs.choice(eligible, size=bs, p=wts)
            s = torch.tensor(off[f] + (rs.random(bs) * (bl[f] - L + 1)).astype(int))
        elif block is None:
            s = torch.randint(0, starts_max, (bs,), generator=g)
        else:
            s = torch.randint(0, cutF, (bs,), generator=g) * block + torch.randint(0, block - L + 1, (bs,), generator=g)
        idx = (s[:, None] + torch.arange(L)[None, :]).to(DEV)
        xb, eb = X[idx], Ei[idx]
        out = m(xb, eb); loss, _ = cpt_loss(out, xb, eb, m.mech, obs_logvar=m.obs_logvar, w_sparse_base=0.01,
                                           w_sparse_shift=0.05, free_bits=0.02, beta=1.0, **({"w_rc": w_rc} if w_rc > 0 else {}))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()

    def _loss(a, b, fb=0.02, as_list=False):
        ls = []
        with torch.no_grad():
            if blocks is not None:
                starts = [int(off[f] + o) for f in range(nF) if a <= off[f] < b for o in range(0, bl[f] - L + 1, L)]
            else:
                starts = (range(a, b - L + 1, L) if block is None else [f * block + o for f in range(a // block, b // block)
                                                                        for o in range(0, block - L + 1, L)])
            for s0 in starts:
                xb, eb = X[s0:s0 + L][None], Ei[s0:s0 + L][None]
                ls.append(float(cpt_loss(m(xb, eb), xb, eb, m.mech, obs_logvar=m.obs_logvar, w_sparse_base=0.01,
                                         w_sparse_shift=0.05, free_bits=fb, beta=1.0)[0]))
        return (float(np.mean(ls)), ls) if as_list else float(np.mean(ls))
    m.eval()
    ltr, lte = _loss(0, cut), _loss(cut, T)
    torch.manual_seed(seed + 99)                                     # held-out ELBO without free bits: fair comparison of candidates with different KL estimators
    if record_windows:                                               # same call, also returns the per-window values
        lte_elbo, _windows = _loss(cut, T, fb=0.0, as_list=True)
    else:
        lte_elbo = _loss(cut, T, fb=0.0)
    zh = np.zeros((T, k), np.float32)
    if blocks is not None:
        chunks = [(int(off[f] + o), int(off[f] + min(o + L, bl[f]))) for f in range(nF) for o in range(0, bl[f], L)]
    else:
        chunks = ([(s0, min(s0 + L, T)) for s0 in range(0, T, L)] if block is None else
                  [(f * block + o, f * block + min(o + L, block)) for f in range(T // block) for o in range(0, block, L)])
    with torch.no_grad():
        for s0, s1 in chunks:
            lo = (0 if block is None else (s0 // block) * block) if blocks is None else int(off[np.searchsorted(off, s0, 'right') - 1])
            s0_ = max(lo, s1 - L)                                  # the last chunk is a full window (within the unit)
            zh[s0_:s1] = m(X[s0_:s1][None], Ei[s0_:s1][None], sample=False)["mu"][0].cpu().numpy()
    info = dict(train_loss=ltr, split_loss=lte, time=time.time() - t0, k_white=k_white, steps=steps, mech_lags=mech_lags,
                mech_nl=mech_nl, mech_flow=mech_flow, split_elbo=lte_elbo)
    if dec != "mlp":
        info["dec"] = dec
    if w_rc > 0:
        info["w_rc"] = w_rc
    if record_windows:
        info["split_elbo_windows"] = [float(v) for v in _windows]
    return zh, info


def fit_cpt2s(x, e, k, seed=0, candidates=((1, 0), (2, 64)), **kw):
    """CPT-2s: selects the transition family by HELD-OUT LOSS, e.g. linear with 1 lag vs 2 lags + a nonlinear term.
    candidates: (lags, nl), (lags, nl, flow) with flow > 0 for a flow candidate (e.g. (2, 64, 32)), or
    (lags, nl, flow, dec) with dec in {"mlp", "linear"} (measurement family).
    With two-element candidates only, selection uses the held-out loss. If any candidate has a flow or a decoder field,
    selection uses the held-out ELBO without free bits (so that different families are compared on the same kind of
    bound); with a decoder field, the selected decoder and the held-out ELBO / loss of ALL candidates are added to the info
    dict."""
    best = None
    has_dec = any(len(tuple(a_)) > 3 for a_ in candidates)
    criterion = "split_elbo" if has_dec or any(len(tuple(a_)) > 2 and tuple(a_)[2] for a_ in candidates) else "split_loss"   # no flow / decoder field: held-out loss
    all_cands = {}
    for cand in candidates:
        cand = tuple(cand); lags, nl, fl, dc = (cand + (0, "mlp")[len(cand) - 2:])[:4]      # missing fields: flow=0, dec="mlp"
        zh, inf = fit_cpt2(x, e, k, seed=seed, mech_lags=lags, mech_nl=nl, mech_flow=fl, **({"dec": dc} if dc != "mlp" else {}), **kw)
        if has_dec:
            all_cands[f"L{lags}NL{nl}" + (f"F{fl}" if fl else "") + f"_{dc}"] = dict(split_elbo=inf["split_elbo"], split_loss=inf["split_loss"])
        if best is None or inf[criterion] < best[1][criterion]:
            best = (zh, inf)
    best[1]["selected_transition"] = f"L{best[1]['mech_lags']}NL{best[1]['mech_nl']}" + (f"F{best[1]['mech_flow']}" if best[1].get("mech_flow") else "")
    best[1]["selection_criterion"] = criterion
    if has_dec:
        best[1]["selected_decoder"] = best[1].get("dec", "mlp")
        best[1]["candidates_split"] = all_cands
    return best


_FIT = {"iVAE": fit_ivae, "TDRL": fit_tdrl, "CPT2": fit_cpt2, "CPT2s": fit_cpt2s}


def fit(method, x, e, k, seed=0, **kw):      # (redefinition: adds CPT2 and CPT2s)
    if method in ("CPT2", "CPT2s"):
        kw.pop("epochs", None)
    return _FIT[method](x, e, k, seed=seed, **kw)
