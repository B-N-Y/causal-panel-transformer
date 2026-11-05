"""Out-of-sample evaluation on the Causal Chambers light tunnel with the protocol of the CITRIS benchmark (Gamella, Bing and Runge
2025): the representation is estimated on the sequence citris_1 only, and evaluated on the separate experiment `uniform`, whose
latent factors are drawn independently and uniformly (rows 0-999 for matching and read-out fitting, rows 1000+ for measurement).
Pipeline, all on citris_1 without ground truth: first 256 principal components of the pixels (all components divided by the
standard deviation of the first one), Stage A (regime contrast with one direction per intervention regime, or the first five
principal components), Stage B (lever selection of Section 7, B2 on a single regime). Variant RC_noB: the regime-contrast
coordinates themselves, without Stage B. The resulting fixed affine map is applied
to the `uniform` images.
Metrics on `uniform` rows 1000+: (i) strict, the matched single component itself (Pearson MCC and Spearman); (ii) as in the
benchmark, an MLP from the matched component to its factor fitted on rows 0-999 (diagonal R^2 and Spearman).
Usage: python citris_uniform_eval.py <lt_crl_benchmark_v1.zip> <citris_1_x64.npy> <out_dir>"""
import io, os, sys, zipfile
import numpy as np, pandas as pd, torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.neural_network import MLPRegressor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "cpt"))
import levers as KL
from contrast import regime_contrast

ZIP, NPY, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
TRUTH = ["red", "green", "blue", "pol_1", "pol_2"]
zf = zipfile.ZipFile(ZIP); K = "lt_crl_benchmark_v1/"
d = pd.read_csv(zf.open(K + "citris_1/citris_1.csv")); u = pd.read_csv(zf.open(K + "uniform/uniform.csv"))
Xu = np.stack([np.asarray(Image.open(io.BytesIO(zf.read(K + "uniform/images_64/" + f))).convert("RGB")) for f in u.image_file])
dev = "cuda" if torch.cuda.is_available() else "cpu"

# principal components as in experiments/real_data_citris.py (scale="common"), with the projection kept for the uniform images
X = torch.tensor(np.load(NPY).reshape(len(d), -1), dtype=torch.float32, device=dev) / 255.
mu = X.mean(0); X -= mu
ev, V = torch.linalg.eigh(X.T @ X / len(X)); V = V[:, torch.argsort(ev, descending=True)[:256]]
F = (X @ V).cpu().numpy().astype(np.float64); del X
Fm, Fs = F.mean(0), F[:, 0].std(); F = (F - Fm) / Fs
Fu = ((torch.tensor(Xu.reshape(len(u), -1), dtype=torch.float32, device=dev) / 255. - mu) @ V).cpu().numpy().astype(np.float64)
Fu = (Fu - Fm) / Fs
e = d.flag.to_numpy(int); E = int(e.max()) + 1; Z = d[TRUTH].to_numpy(float); Zu = u[TRUTH].to_numpy(float)
cut = int(len(F) * 0.8)


def stage_a(name):
    if name == "RC_noB":
        return stage_a("RC")
    if name == "PCA5":
        return F[:, :5].copy(), Fu[:, :5].copy()
    _, info = regime_contrast(F, e, 1); W = info["W"]; P = F @ W
    m, s = P[:cut].mean(0), P[:cut].std(0)
    return (P - m) / s, (Fu @ W - m) / s


def mcc_in(zt, zh):
    C = np.abs(np.corrcoef(zt.T, zh.T)[:5, 5:]); r, c = linear_sum_assignment(-C); return float(C[r, c].mean())


rows = []
va, te = np.arange(1000), np.arange(1000, len(u))
for name in ("RC", "PCA5", "RC_noB"):
    zA, zAu = stage_a(name)
    if name == "RC_noB":
        z, info = zA.copy(), dict(certificate=None, selection="none (no Stage B)")
    else:
        z, info = KL.resolve_v2(zA, e, E, b2_filter="single")
    A1 = np.c_[np.ones(len(zA)), zA]; B = np.linalg.lstsq(A1, z, rcond=None)[0]
    lin_r2 = float(1 - ((A1 @ B - z) ** 2).sum() / ((z - z.mean(0)) ** 2).sum())      # Stage B is linear: should be ~1
    zu = np.c_[np.ones(len(zAu)), zAu] @ B
    C = np.abs(np.corrcoef(Zu[va].T, zu[va].T)[:5, 5:]); r, c = linear_sum_assignment(-C)
    row = dict(stage_a=name, certificate=info.get("certificate"), selection=info.get("selection"), stageB_linear_r2=lin_r2,
               mcc_citris1_insample=mcc_in(Z, z), n_val=len(va), n_test=len(te))
    pe, sp, r2m, spm = [], [], [], []
    for i, j in zip(r, c):
        x_va, x_te, y_va, y_te = zu[va, j:j + 1], zu[te, j:j + 1], Zu[va, i], Zu[te, i]
        pe.append(abs(np.corrcoef(x_te[:, 0], y_te)[0, 1])); sp.append(abs(spearmanr(x_te[:, 0], y_te)[0]))
        xm, xs, ym, ys = x_va.mean(), x_va.std(), y_va.mean(), y_va.std()
        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=0).fit((x_va - xm) / xs, (y_va - ym) / ys)
        yh = mlp.predict((x_te - xm) / xs) * ys + ym
        r2m.append(1 - ((y_te - yh) ** 2).mean() / y_te.var()); spm.append(spearmanr(yh, y_te)[0])
        row.update({f"pearson_{TRUTH[i]}": pe[-1], f"spearman_{TRUTH[i]}": sp[-1], f"mlp_r2_{TRUTH[i]}": r2m[-1],
                    f"mlp_spearman_{TRUTH[i]}": spm[-1]})
    row.update(mcc_uniform=float(np.mean(pe)), spearman_uniform=float(np.mean(sp)), mlp_r2_diag=float(np.mean(r2m)),
               mlp_spearman_diag=float(np.mean(spm)))
    rows.append(row); print(row, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "citris_uniform_eval.csv"), index=False)
