"""R15c: the NCTRL model itself on its own generator (Song et al., NeurIPS 2023; github.com/xiangchensong/nctrl d2540bb, MIT).
Same data as R15 (same generator call and seed), with the authors' configs/simulation/simulation_nctrl.yaml settings:
hidden 128, embedding 2, lr 5e-4, beta 2e-3, gamma 2e-2, hmm_mode 'mle_scaled:H', batch 64, 200 epochs, 1024 validation
sequences, random_split + shuffle=False as in train_simulation.py. NCTRL treats the regime as LATENT (it infers it with an
HMM); in R15, CPT-2s and TDRL were given the true regime.
The authors' script selects the best model by VALIDATION MCC (computed with the true z!) and stops at MCC >= .99 (oracle
selection). Here:
  * PRIMARY: the model at the end of training (without looking at the true z), the same condition as for CPT-2s/TDRL in R15;
  * SECONDARY ("authors' selection", oracle): the model at the epoch with the highest validation MCC, for transparency only.
Evaluation as in R15: ALL sequences, last step, affine R^2, affine MCC and raw MCC against the true z; also regime accuracy.
A thin stub replaces pytorch-lightning (as in tdrl_runner.py; model code unchanged). Time budget: --hours.
The authors' rule "stop if validation MCC >= .99" is an oracle rule; it is available only through the --oracle_stop flag and
is OFF by default, so the primary model is truly the end-of-training model. Columns: epochs_done, epochs_planned,
stopped_by_budget, oracle_stopped.
Output: outputs/nctrl_on_nctrl_generator{tag}_s{seed}.csv
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

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n", type=int, default=40000)
ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--hours", type=float, default=4.5)
ap.add_argument("--oracle_stop", action="store_true", help="the authors' oracle stopping at val MCC >= .99; OFF by default")
ap.add_argument("--quick", action="store_true"); ap.add_argument("--tag", default=""); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(a.seed)                                           # as in R15 (the generator partly uses the global np.random)
D = NG.arhmm_pnl_change_gaussian_ts(NClass=5, lags=2, Nlayer=3, length=4, negSlope=0.2, latent_size=8, noise_scale=0.1,
                                    batch_size=500 if a.quick else a.n, state_stickiness=0.9, random_seed=a.seed)["data"]
X = torch.tensor(D["X"], dtype=torch.float32); Y = torch.tensor(D["Z"], dtype=torch.float32); C = torch.tensor(D["C"]).long()
torch.manual_seed(a.seed)
n_val = 100 if a.quick else 1024
perm = torch.randperm(len(X)); tr, va = perm[n_val:], perm[:n_val]
m = NCTRL(x_dim=X.shape[-1], z_dim=8, lags=2, n_class=5, hidden_dim=128, embedding_dim=2, lr=5e-4, beta=2e-3, gamma=2e-2,
          correlation="Pearson", hmm_mode="mle_scaled:H").to(dev)
m._opts = m.configure_optimizers()
Xd, Yd, Cd = X.to(dev), Y.to(dev), C.to(dev)


def encode_windows(idx, bs=4096):
    m.eval(); zs, cs = [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]; _, mus, _, _ = m.net(Xd[j]); _, c_est = m.hmm(Xd[j])
            zs.append(mus.cpu()); cs.append(c_est.cpu())
    m.train(); return torch.cat(zs), torch.cat(cs)


t0 = time.time(); ep_n = 1 if a.quick else a.epochs; bs = 64; best = (-np.inf, None, 0); ep_done = 0
stopped_by_budget = False; oracle_stopped = False
for ep in range(ep_n):
    for b, i in enumerate(range(0, len(tr), bs)):               # shuffle=False (the authors' DataLoader)
        j = tr[i:i + bs]; m.training_step((Xd[j], Yd[j], Cd[j]), b)
    ep_done = ep + 1
    zv, _ = encode_windows(va)                                   # the authors' validation MCC: all steps, true z
    vm = compute_mcc(zv.reshape(-1, 8).T.numpy(), Y[va].reshape(-1, 8).T.numpy(), "Pearson")
    if vm > best[0]:
        best = (vm, copy.deepcopy(m.state_dict()), ep + 1)
    print(f"  NCTRL seed{a.seed} ep{ep + 1} val_mcc={vm:.4f} {time.time() - t0:.0f}s", flush=True)
    if a.oracle_stop and vm >= 0.99:                            # only with the flag; off in the primary run
        oracle_stopped = ep_done < ep_n
        break
    if time.time() - t0 > a.hours * 3600:
        stopped_by_budget = ep_done < ep_n
        break
elapsed = time.time() - t0
all_idx = torch.arange(len(X))


def evaluate(label):
    z, c = encode_windows(all_idx)
    zt, zh = Y[:, -1].numpy(), z[:, -1].numpy()
    mm, _ = mcc(zt, zh); r2, ma = rotation_diagnostics(zt, zh)
    acc, _ = compute_acc(C[:, 2:].flatten().numpy(), c.flatten().numpy(), C=5)
    return dict(method=label, seed=a.seed, MCC=round(float(mm), 4), r2_affine=round(r2, 4), mcc_affine=round(ma, 4),
                regime_accuracy=round(float(acc), 4), epoch=ep_done, time=round(elapsed, 1), nctrl_commit="d2540bb",
                epochs_done=ep_done, epochs_planned=ep_n, stopped_by_budget=bool(stopped_by_budget), oracle_stopped=bool(oracle_stopped),
                oracle_stop_flag=bool(a.oracle_stop), version="r15cv2")


rows = [evaluate("NCTRL (end of training)")]
m.load_state_dict(best[1]); rows.append(dict(evaluate("NCTRL (authors' selection, oracle)"), best_epoch=best[2], best_val_mcc=round(float(best[0]), 4)))
for r in rows:
    print(r, flush=True)
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "outputs", f"nctrl_on_nctrl_generator{a.tag}_s{a.seed}.csv"), index=False)
