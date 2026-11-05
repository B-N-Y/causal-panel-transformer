"""iSDS (Balsells-Rodas, Wang and Li, ICML 2024; github.com/charlio23/identifiable-SDS, MIT, b6c11a5), one of the closest
competitors, on our generator (same seeds and cells as R10). iSDS treats the regime as LATENT (Markov) and models the
dynamics with a per-regime MLP.
The authors' train_snlds.py recipe is followed: PCA -> NeuralMSM (pre-training of regimes and transitions; here 3 restarts,
the authors use 10) -> VariationalSNLDS (bidirectional LSTM encoder, alpha inference; beta schedule 0 -> 1; Q/pi unfrozen
later), 100 epochs in the authors' schedule.
Data: one long series (T=6000) -> windows of length 200, stride 50. Measured: affine R^2 / MCC (true z), MCC after OUR
Stage B (with the observed regime, for fairness), and how well iSDS recovers the regime (accuracy under the best permutation).
Update rule: if --epochs is not given, the number of epochs = round(--updates / number of batches), default --updates 9600
(T=6000, stride 10 -> 581 windows, 12 batches x 800 epochs = 9,600, close to the authors' ~10,000). The number of updates is
thus ~9,600 for every T (T=1000: 81 windows, 2 batches -> 4,800 epochs; units of length 10/25/100 -> 800/1,920/4,800 epochs).
The authors' schedule (KL warm-up, unfreezing of Q/pi, learning-rate decay) is scaled in proportion to the number of epochs
(scale = epochs // 100). If --epochs is given, it is used as is.
Stage B columns from stage_b_scoring (lever_design, MCC_design primary; B is an oracle upper bound), also in short-panel mode
(consecutive pairs only). --savez: latents are saved to outputs/z/.
Output: outputs/isds{tag}_s{seed}.csv
"""
import os, sys, time, argparse, itertools, numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))

sys.path.insert(0, os.path.join(HERE, "third_party", "isds")); import encoders as KD                                         # (imported for its path set-up)
from simulation import gen_panel, rotation_diagnostics, resolve_rotation, mcc
from stage_b_scoring import stage_b_columns
from models.VariationalSNLDS import VariationalSNLDS
from models.NeuralMSM import NeuralMSM
from sklearn.decomposition import PCA

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--E", type=int, default=4)
ap.add_argument("--varmode", default="scale"); ap.add_argument("--innov", default="gauss"); ap.add_argument("--graph", default="fixed")
ap.add_argument("--epochs", type=int, default=None, help="if not given, derived from the --updates rule"); ap.add_argument("--msm_restart", type=int, default=3)
ap.add_argument("--updates", type=int, default=9600, help="target number of updates; epochs = round(updates / number of batches)")
ap.add_argument("--savez", action="store_true", help="save latents to outputs/z/isds{tag}_s{seed}.npz")
ap.add_argument("--stride", type=int, default=50, help="window stride (10 -> ~580 windows, close to the authors' ~10k updates)")
ap.add_argument("--k", type=int, default=4); ap.add_argument("--D", type=int, default=12); ap.add_argument("--T", type=int, default=6000)
ap.add_argument("--obs_noise", type=float, default=0.05)   # (K1) hard settings; defaults as in R10
ap.add_argument("--Tb", type=int, default=0, help="(K2) >0: short panel with units of this length; window = min(200, Tb), within units")
ap.add_argument("--quick", action="store_true"); ap.add_argument("--tag", default=""); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed); np.random.seed(a.seed)
T_ = 1500 if a.quick else a.T; k = a.k
x, z, e, meta = gen_panel(seed=a.seed, E=a.E, T=T_, varmode=a.varmode, innov=a.innov, spurious=False, graph=a.graph,
                          k=a.k, D=a.D, obs_noise=a.obs_noise)
x = ((x - x.mean(0)) / x.std(0)).astype(np.float32); D = x.shape[1]; W, S = 200, a.stride
if a.Tb:                                                          # (K2) short panel: same split as R17
    T_ = T_ - T_ % a.Tb; x, z, e = x[:T_], z[:T_], e[:T_]; W = min(200, a.Tb)
    starts = np.array([u * a.Tb + o for u in range(T_ // a.Tb) for o in range(0, a.Tb - W + 1, max(1, min(S, a.Tb - W + 1)))])
else:
    starts = np.arange(0, T_ - W + 1, S)
seq = np.stack([x[b:b + W] for b in starts])
n_batches = int(np.ceil(len(seq) / 50))                          # batch size 50 (DataLoader below)
if a.epochs is None:                                             # update rule
    a.epochs = max(1, int(round(a.updates / n_batches)))
t0 = time.time()
# --- MSM pre-training (authors' recipe) ---
obs_new = PCA(n_components=k).fit_transform(seq.reshape(-1, D)).reshape(-1, W, k)
best_ll, best_msm = -np.inf, None
for i in range(1 if a.quick else a.msm_restart):
    m = NeuralMSM(a.E, k, hid_dim=16, device=dev, lr=7e-3, causal=False, l1_penalty=0, l2_penalty=0, activation="cos")
    ll, _, _ = m.fit(torch.from_numpy(obs_new), 2 if a.quick else 1000, batch_size=100, early_stopping=2, max_scheduling_steps=2)
    m.to("cpu")
    if ll > best_ll:
        best_ll, best_msm = ll, m
model = VariationalSNLDS(D, k, 64, a.E, encoder_type="recurent", device=dev, annealing=False, inference="alpha", beta=0)
model.transitions = best_msm.transitions.to(dev)
model.Q = torch.nn.Parameter(best_msm.Q.log()).to(dev); model.pi = torch.nn.Parameter(best_msm.pi.log().to(dev))
model.init_cov = torch.nn.Parameter(best_msm.init_cov.to(dev)); model.init_mean = torch.nn.Parameter(best_msm.init_mean.to(dev))
model.temperature = 1; model.beta = 0
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
sched_scale = max(1, a.epochs // 100)                           # the authors' schedule (100 epochs) scaled to the number of epochs
sch = torch.optim.lr_scheduler.StepLR(opt, step_size=40 * sched_scale, gamma=0.5)
model.Q.requires_grad_(False); model.pi.requires_grad_(False)
dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(seq)), batch_size=50, shuffle=True)
ep_n = 2 if a.quick else a.epochs
for ep in range(ep_n):
    if 5 * sched_scale <= ep < 10 * sched_scale:
        model.beta = 1
    elif ep >= 10 * sched_scale:
        model.beta = 1; model.Q.requires_grad_(True); model.pi.requires_grad_(True)
    for (sample,) in dl:
        xb = sample.float().to(dev); opt.zero_grad(); _, _, _, losses = model(xb)
        losses["loss"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    sch.step()
# --- encoding: non-overlapping windows of length 200 ---
model.eval(); zh = np.zeros((T_, k), np.float32); gam = np.zeros((T_, a.E), np.float32)
with torch.no_grad():
    for b in (list(range(0, T_ - W + 1, W)) + ([T_ - W] if T_ % W else [])) if not a.Tb else \
             [u * a.Tb + o for u in range(T_ // a.Tb) for o in range(0, a.Tb, W)]:
        xb = torch.from_numpy(x[b:b + W][None]).to(dev)
        _, z_mu, _ = model._encode_obs(xb); zh[b:b + W] = z_mu.reshape(W, k).cpu().numpy()
        g, _ = model._compute_posteriors(model._compute_local_evidence(z_mu.reshape(1, W, k))); gam[b:b + W] = g.reshape(W, a.E).cpu().numpy()
eh = gam.argmax(1); acc = max(np.mean(np.array(p)[eh] == e) for p in itertools.permutations(range(a.E)))
if a.Tb:                                                          # (K2) same scoring as R17 (first row of each unit excluded; panel-mode Stage B)
    import levers as KL
    segment = np.repeat(np.arange(T_ // a.Tb), a.Tb); msk = np.r_[False, segment[1:] == segment[:-1]]
    m_, _ = mcc(z[msk], zh[msk]); r2, ma = rotation_diagnostics(z[msk], zh[msk]); lever_choice = np.nan
    abk = stage_b_columns(z[msk], zh[msk], e[msk], a.E, a.varmode, a.innov, t_index=np.arange(T_)[msk])   # panel mode
    try:
        zb, info = KL.resolve_v2(zh.astype(np.float64), e, a.E, segment=segment); mres = float(mcc(z[msk], zb[msk])[0]) if zb is not None else np.nan
        lever_choice = info.get("selection")
    except Exception:                                             # noqa: BLE001
        mres = np.nan
else:
    m_, _ = mcc(z[200:], zh[200:]); r2, ma = rotation_diagnostics(z[200:], zh[200:]); mres = np.nan; lever_choice = np.nan
    abk = stage_b_columns(z[200:], zh[200:], e[200:], a.E, a.varmode, a.innov)   # same as resolve_rotation
row = dict(method="iSDS", version="isdsv2", seed=a.seed, k=a.k, D=a.D, T=T_, obs_noise=a.obs_noise, stride=a.stride, n_windows=len(seq), updates=ep_n * len(dl), E=a.E, varmode=a.varmode, innov=a.innov, graph=a.graph, epochs=ep_n, MCC=m_, r2_affine=r2,
           mcc_affine=ma, **abk, regime_accuracy=acc,
           time=round(time.time() - t0, 1), isds_commit="b6c11a5", T_per_unit=a.Tb, MCC_resolved=mres, stage_b_choice_auto=lever_choice)
if a.savez:                                                       # save latents for later re-scoring
    os.makedirs(os.path.join(HERE, "outputs", "z"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "outputs", "z", f"isds{a.tag}_s{a.seed}.npz"), z=z.astype(np.float32), zh=zh, e=e.astype(np.int64))
print(row, flush=True)
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
pd.DataFrame([row]).to_csv(os.path.join(HERE, "outputs", f"isds{a.tag}_s{a.seed}.csv"), index=False)
