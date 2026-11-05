"""Common Stage B scoring.
Same measure as simulation.resolve_rotation (regime-wise AR residuals -> JD and ICA; the true z is used ONLY for scoring).
For each method the following columns are written:
  mcc_jd, mcc_ica     : result of each of the two levers separately
  lever_design        : lever fixed IN ADVANCE by the data design (scale variability -> JD; no scale variability with t5
                        innovations -> ICA; Gaussian without a lever -> JD)
  MCC_design          : primary (oracle-free) MCC after Stage B = the column of the design lever
  B                   : max(JD, ICA), ONLY a labelled oracle upper bound
Panel (short-unit) mode: if t_index is given, only consecutive (t, t+1) pairs are used (pairs crossing a unit boundary are
dropped); with t_index None the result is IDENTICAL to simulation.resolve_rotation.
"""
import numpy as np
from simulation import resolve_rotation, mcc, _joint_diag


def design_lever(varmode, innov):
    if varmode in ("scale", "both"):
        return "JD"
    return "ICA" if innov == "t5" else "JD"


def _resolve_panel(z_true, z_hat, e, E, t_index):
    """Panel version of resolve_rotation: only pairs with t_index[i+1] == t_index[i] + 1."""
    from sklearn.decomposition import FastICA
    zh = z_hat - z_hat.mean(0); k = zh.shape[1]
    t_index = np.asarray(t_index); valid = t_index[1:] == t_index[:-1] + 1
    res = np.zeros_like(zh[1:]); covs = []
    for r in range(E):
        idx = np.where((e[1:] == r) & valid)[0]
        if len(idx) < 5 * k:
            continue
        Xp = np.column_stack([np.ones(len(idx)), zh[idx]]); Y = zh[idx + 1]
        Bc, *_ = np.linalg.lstsq(Xp, Y, rcond=None); res[idx] = Y - Xp @ Bc; covs.append(np.cov((Y - Xp @ Bc).T))
    res = res[valid]
    Cp = np.cov(res.T); w, U = np.linalg.eigh(Cp); Wh = U @ np.diag(1 / np.sqrt(np.maximum(w, 1e-10))) @ U.T
    out = {}
    out["mcc_jd"] = float(mcc(z_true, zh @ Wh @ _joint_diag([Wh @ c @ Wh for c in covs]))[0]) if len(covs) >= 2 else np.nan
    try:
        ica = FastICA(n_components=k, whiten="unit-variance", max_iter=2000, random_state=0).fit(res)
        out["mcc_ica"] = float(mcc(z_true, zh @ ica.components_.T)[0])
    except Exception:                                              # noqa: BLE001
        out["mcc_ica"] = np.nan
    return out


def stage_b_columns(z_true, z_hat, e, E, varmode, innov, t_index=None):
    rr = resolve_rotation(z_true, z_hat, e, E) if t_index is None else _resolve_panel(z_true, z_hat, e, E, t_index)
    lev = design_lever(varmode, innov)
    jd, ica = rr["mcc_jd"], rr["mcc_ica"]
    r4 = lambda v: round(float(v), 4) if np.isfinite(v) else np.nan   # noqa: E731
    return dict(mcc_jd=r4(jd), mcc_ica=r4(ica), lever_design=lev, MCC_design=r4(jd if lev == "JD" else ica),
                B=r4(np.nanmax([jd, ica])) if np.isfinite([jd, ica]).any() else np.nan)
