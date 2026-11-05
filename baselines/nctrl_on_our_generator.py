"""R22: NCTRL itself on OUR generator (R10 cells; the reverse-direction fair comparison). Authors' configuration (hidden 128,
embedding 2, lr 5e-4, beta 2e-3, gamma 2e-2, mle_scaled:H, batch 64, windows of 2 lags + length 4), ~120k updates (same order
as R15c). NCTRL treats the regime as LATENT. Data: gen_panel (k=4, D=12, T=6000, no spurious columns, fixed graph) -> sliding
windows of length 6.
Primary: the model at the end of training (oracle-free; NO oracle early stopping); secondary: the model with the best
validation MCC (computed with the true z).
Evaluation as in R10 (t >= 200, true z): raw MCC, affine R^2, affine MCC; in addition our Stage B applied to the NCTRL output.
Stage B columns from stage_b_scoring (mcc_jd, mcc_ica, lever_design, MCC_design [primary, oracle-free], B [oracle upper
bound]); the pipeline's automatic choice (MCC_stage_b, resolve_v2) is reported as well. Validation set n_val = min(1024, 20%)
so that short series (T=1000: 995 windows) do not give an empty training set; steps_per_epoch >= 1. Time-budget stopping is
recorded in the columns epochs_done, epochs_planned, stopped_by_budget. --savez: latents are saved to outputs/z/ (for later
re-scoring). At T=6000, n_val = 1024.
Output: outputs/nctrl_on_our_generator{tag}_s{seed}.csv
"""
import os, sys, time, types, copy, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))

NC = os.path.join(HERE, "third_party", "nctrl")
sys.path.insert(0, NC); sys.path.insert(0, os.path.join(NC, "tools"))

_pl = types.ModuleType("pytorch_lightning")                    # --- lightning stub (model code unchanged) ---
class _LM(nn.Module):
    def log(self, *a, **k): pass
    def log_dict(self, *a, **k): pass
    def save_hyperparameters(self, *a, **k): pass
    def manual_backward(self, loss): loss.backward()
    def optimizers(self): return self._opts
_pl.LightningModule = _LM
sys.modules["pytorch_lightning"] = _pl
import generate_data as NG                                      # noqa: E402
from models.simulation import NCTRL                             # noqa: E402
from models.metrics.correlation import compute_mcc, compute_acc  # noqa: E402
from simulation import mcc, rotation_diagnostics                      # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--E", type=int, default=4)
ap.add_argument("--varmode", default="scale"); ap.add_argument("--innov", default="gauss"); ap.add_argument("--T", type=int, default=6000)
ap.add_argument("--k", type=int, default=4); ap.add_argument("--D", type=int, default=12); ap.add_argument("--graph", default="fixed")
ap.add_argument("--obs_noise", type=float, default=0.05)   # (K1) hard settings; defaults as in R22
ap.add_argument("--lr", type=float, default=5e-4, help="(K4) learning-rate sensitivity; the authors' value is 5e-4")
ap.add_argument("--Tb", type=int, default=0, help="(K2) >0: short panel; the series is split into units of length Tb, windows stay within units")
ap.add_argument("--updates", type=int, default=120000, help="total number of updates (~122k in R15c)"); ap.add_argument("--hours", type=float, default=4.5)
ap.add_argument("--savez", action="store_true", help="save latents to outputs/z/nctrl_on_our_generator{tag}_s{seed}.npz")
ap.add_argument("--quick", action="store_true"); ap.add_argument("--tag", default=""); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
import levers as KL   # noqa: E402
from simulation import gen_panel                                                  # noqa: E402
from stage_b_scoring import stage_b_columns                                    # noqa: E402
T_ = 1500 if a.quick else a.T; k = a.k; LAG = 2; LEN = 4; W = LAG + LEN
x, z, e, meta = gen_panel(seed=a.seed, E=a.E, T=T_, varmode=a.varmode, innov=a.innov, spurious=False, graph=a.graph,
                          k=a.k, D=a.D, obs_noise=a.obs_noise)
x = ((x - x.mean(0)) / x.std(0)).astype(np.float32)
if a.Tb:                                                         # (K2) short panel: same split as R17
    T_ = T_ - T_ % a.Tb; x, z, e = x[:T_], z[:T_], e[:T_]
    starts = np.array([b for b in range(0, T_ - W + 1) if b % a.Tb <= a.Tb - W])
else:
    starts = np.arange(0, T_ - W + 1)                            # sliding windows (stride 1)
X = torch.tensor(np.stack([x[b:b + W] for b in starts])); Y = torch.tensor(np.stack([z[b:b + W] for b in starts]), dtype=torch.float32)
C = torch.tensor(np.stack([e[b:b + W] for b in starts])).long()
torch.manual_seed(a.seed)
n_val = min(100 if a.quick else 1024, max(1, int(0.2 * len(X))))   # T=1000: 995 windows -> 199; T=6000: 1024
perm = torch.randperm(len(X)); tr, va = perm[n_val:], perm[:n_val]
m = NCTRL(x_dim=X.shape[-1], z_dim=k, lags=LAG, n_class=a.E, hidden_dim=128, embedding_dim=2, lr=a.lr, beta=2e-3, gamma=2e-2,
          correlation="Pearson", hmm_mode="mle_scaled:H").to(dev)
# NCTRL's HMM operates on the OBSERVATION x, but its constructor is given x_dim=z_dim (this goes unnoticed in the authors'
# setting, where D = k = 8). Here D=12, k=4 -> the HMM is rebuilt with the observation dimension (D); the rest of the model
# code is unchanged.
from models.hmm import HMM                                       # noqa: E402
m.hmm = HMM(n_class=a.E, lags=LAG, x_dim=X.shape[-1], hidden_dim=128, mode="mle_scaled:H").to(dev)
m._opts = m.configure_optimizers()
Xd, Yd, Cd = X.to(dev), Y.to(dev), C.to(dev)


def encode_windows(idx, bs=4096):
    m.eval(); zs, cs = [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]; _, mus, _, _ = m.net(Xd[j]); _, c_est = m.hmm(Xd[j])
            zs.append(mus.cpu()); cs.append(c_est.cpu())
    m.train(); return torch.cat(zs), torch.cat(cs)


if len(tr) < 1:
    raise SystemExit(f"empty training set (windows {len(X)}, n_val {n_val})")
t0 = time.time(); bs = 64; steps_per_epoch = max(1, (len(tr) + bs - 1) // bs)
ep_n = 1 if a.quick else max(1, a.updates // steps_per_epoch); best = (-np.inf, None, 0); ep_done = 0; stopped_by_budget = False
for ep in range(ep_n):
    for b, i in enumerate(range(0, len(tr), bs)):               # shuffle=False (the authors' DataLoader)
        j = tr[i:i + bs]; m.training_step((Xd[j], Yd[j], Cd[j]), b)
    ep_done = ep + 1
    over_budget = time.time() - t0 > a.hours * 3600
    if ep % 20 == 19 or ep == ep_n - 1 or over_budget:            # the authors' validation MCC (oracle selection is SECONDARY only); also at budget stop
        zv, _ = encode_windows(va); vm = compute_mcc(zv.reshape(-1, k).T.numpy(), Y[va].reshape(-1, k).T.numpy(), "Pearson")
        if vm > best[0]:
            best = (vm, copy.deepcopy(m.state_dict()), ep + 1)
        print(f"  NCTRL(our generator) seed{a.seed} ep{ep + 1}/{ep_n} val_mcc={vm:.4f} {time.time() - t0:.0f}s", flush=True)
    if over_budget:
        stopped_by_budget = ep_done < ep_n
        break
elapsed = time.time() - t0
all_idx = torch.arange(len(X))
budget = dict(epochs_done=ep_done, epochs_planned=ep_n, stopped_by_budget=bool(stopped_by_budget), n_val=n_val,
              updates_done=ep_done * steps_per_epoch, version="r22v2")
_saved = {}


def evaluate_short(label):                                       # (K2) same scoring as R17: window-end rows, panel-mode Stage B
    zz, c = encode_windows(all_idx); last_idx = starts + W - 1
    zr = zz[:, -1].numpy().astype(np.float64); zt = z[last_idx]; segment = last_idx // a.Tb; er = e[last_idx]
    mm, _ = mcc(zt, zr); r2, ma = rotation_diagnostics(zt, zr)
    r = dict(method=label, seed=a.seed, T_per_unit=a.Tb, N_units=T_ // a.Tb, E=a.E, varmode=a.varmode, innov=a.innov, k=a.k,
             MCC_raw=round(float(mm), 4), r2_affine=round(r2, 4), mcc_affine=round(ma, 4), epoch=ep_done, time=round(elapsed, 1))
    try:
        zb, info = KL.resolve_v2(zr, er, a.E, segment=segment)
        r.update(selection=info["selection"], MCC_resolved=round(float(mcc(zt, zb)[0]), 4) if zb is not None else np.nan)
    except Exception as ex:                                       # noqa: BLE001
        r.update(error=str(ex)[:120])
    r.update(stage_b_columns(zt, zr, er, a.E, a.varmode, a.innov, t_index=last_idx), **budget)   # panel mode, consecutive pairs only
    _saved[label] = dict(zh=zr.astype(np.float32), t=last_idx)
    return r


def evaluate(label):
    if a.Tb:
        return evaluate_short(label)
    zz, c = encode_windows(all_idx)
    zh = np.zeros((T_, k), np.float32); zh[W - 1:] = zz[:, -1].numpy(); zh[:W - 1] = zz[0, :W - 1].numpy()   # window end = t
    zt = z[200:]; zhh = zh[200:]
    mm, _ = mcc(zt, zhh); r2, ma = rotation_diagnostics(zt, zhh)
    acc, _ = compute_acc(C[:, LAG:].flatten().numpy(), c.flatten().numpy(), C=a.E)
    abk = stage_b_columns(zt, zhh, e[200:], a.E, a.varmode, a.innov)   # standard Stage B scoring; the design lever is primary
    r = dict(method=label, seed=a.seed, E=a.E, varmode=a.varmode, innov=a.innov, graph=a.graph, k=a.k, D=a.D, T=T_, obs_noise=a.obs_noise,
             lr=a.lr, **abk, MCC=round(float(mm), 4),
             r2_affine=round(r2, 4), mcc_affine=round(ma, 4), regime_accuracy=round(float(acc), 4), epoch=ep_done, time=round(elapsed, 1),
             nctrl_commit="d2540bb", **budget)
    _saved[label] = dict(zh=zh.astype(np.float32))
    try:                                                          # our Stage B (with the observed regime) on the NCTRL output
        zb, info = KL.resolve_v2(zh.astype(np.float64), e, a.E)
        r.update(stage_b_choice=info["selection"], MCC_stage_b=round(float(mcc(zt, zb[200:])[0]), 4) if zb is not None else np.nan)
    except Exception as ex:                                       # noqa: BLE001
        r.update(stage_b_error=str(ex)[:120])
    return r


rows = [evaluate("NCTRL (end of training)")]
m.load_state_dict(best[1]); rows.append(dict(evaluate("NCTRL (oracle selection)"), best_epoch=best[2], best_val_mcc=round(float(best[0]), 4)))
for r in rows:
    print(r, flush=True)
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
if a.savez:                                                       # save latents for later re-scoring
    os.makedirs(os.path.join(HERE, "outputs", "z"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "outputs", "z", f"nctrl_on_our_generator{a.tag}_s{a.seed}.npz"), z=z.astype(np.float32), e=e.astype(np.int64),
                        **{("last_" if "end of training" in k_ else "oracle_") + kk: vv for k_, d_ in _saved.items() for kk, vv in d_.items()})
pd.DataFrame(rows).to_csv(os.path.join(HERE, "outputs", f"nctrl_on_our_generator{a.tag}_s{a.seed}.csv"), index=False)
