"""Synthetic evaluation of the end-to-end pipeline (fresh seeds; the ground truth is used only for SCORING).
Usage: python experiments/evaluate_pipeline.py --seed 100 --E 4 --varmode scale --innov gauss [--nospurious] [--graph random]
Output: outputs/pipeline{tag}_s{seed}.csv (one row) + outputs/z/pipeline{tag}_s{seed}.npz
(with --save_latents also outputs/z/pipeline_stage_a{tag}_s{seed}.npz for every dataset)
With --b3_pattern the zero pattern is passed to the pipeline and the B3 solutions are scored; --b3_scan adds the uniqueness check
of Section 7 (columns b3uniq_*, selection_rule). --generator bs uses Counterexample BS (cpt/simulation_bs.py).
Scores: whether the selected k is correct (truth: 4 without spurious columns, 5 with them), certificate, MCC (against z or
[z, conf] of the true dimension), edge-type accuracy (support F1, FIXED/REGIME_DEPENDENT accuracy), correct identification of the
spurious latent, relative error of the weights.
"""
import os, sys, argparse, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))

import pipeline as BH
from simulation import gen_panel, mcc, rotation_diagnostics

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=100); ap.add_argument("--E", type=int, default=4)
ap.add_argument("--varmode", default="scale"); ap.add_argument("--innov", default="gauss")
ap.add_argument("--nospurious", action="store_true"); ap.add_argument("--graph", default="fixed")
ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--quick", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--encoders", default="iVAE-TDRL", help="main-second encoder (dash-separated, since sbatch --export splits at commas)")
ap.add_argument("--dim_c", type=float, default=0.07, help="elbow threshold (0.07; confirmed on fresh seeds)")
ap.add_argument("--dim_rule", default="v2", help="v2 (default) | elbow (elbow rule)")
# optional; without them the behavior and the columns are unchanged
ap.add_argument("--b2_variant", default=None, help="B2 variant: default (not given) = residuals pooled across regimes; "
                "'single' = the regime with the most observations")
ap.add_argument("--b2_extra", default=None, help="second B2 variant on the same Stage A ('pooled' | 'single'); written to *_extra columns")
ap.add_argument("--save_latents", action="store_true", help="save the Stage-A latents for every dataset (not only with the correct dimension)")
ap.add_argument("--generator", default="panel", choices=["panel", "bs"], help="bs: Counterexample BS (E = 2, Gaussian, no spurious block)")
ap.add_argument("--delta", type=float, default=None, help="strength of B1 for varmode scale_delta (s_je = 0.3 (1 + delta q_je))")
ap.add_argument("--b3_pattern", default=None, choices=["correct", "missing_edge", "extra_free"], help="pattern passed to the pipeline (B3)")
ap.add_argument("--b3_extra", default=None, help="further pattern versions on the same Stage A (dash-separated, e.g. missing_edge-extra_free)")
ap.add_argument("--b3_scan", type=int, default=0, help=">0: number of random starts of the B3 scan with the correct pattern")
ap.add_argument("--rc_m", type=int, default=None, help="regime-contrast pre-projection, directions per regime (pipeline rc_m)")
a = ap.parse_args()
T_ = 800 if a.quick else 6000; ep = 2 if a.quick else a.epochs
if a.generator == "bs":
    from simulation_bs import gen_bs
    if a.E != 2 or not a.nospurious or a.innov != "gauss":
        raise SystemExit("--generator bs needs --E 2 --innov gauss --nospurious (definition of BS)")
    x, z, e, meta = gen_bs(seed=a.seed, T=T_)
else:
    x, z, e, meta = gen_panel(seed=a.seed, E=a.E, T=T_, varmode=a.varmode, innov=a.innov, spurious=not a.nospurious, graph=a.graph,
                              **({"delta": a.delta} if a.delta is not None else {}))
k_true = 4 if a.nospurious else 5
patterns, masks_main = None, None
if a.b3_pattern:
    import b3_pattern as B3P
    if not a.nospurious:
        raise SystemExit("--b3_pattern needs --nospurious (the pattern is defined on the true latent dimension k = 4)")
    base = meta["masks"] if a.generator == "bs" else (np.abs(np.stack(meta["A_list"])) > 1e-9).astype(float)
    patterns = B3P.pattern_versions(base, a.seed)
    masks_main = patterns[a.b3_pattern]
out = BH.run_pipeline(x, e, ks=(2, 3, 4, 5, 6, 7) if not a.quick else (3, 4, 5), dim_seeds=(0, 1) if not a.quick else (0,),
                  seed=0, epochs=ep, B=200 if not a.quick else 20, encoders=tuple(a.encoders.split("-")),
                  cpt2_steps=30 if a.quick else 20000, dim_c=a.dim_c, dim_rule=a.dim_rule,
                  **({"b2_filter": a.b2_variant} if a.b2_variant and a.b2_variant != "pooled" else {}), b2_extra=a.b2_extra,
                  **({"masks": masks_main} if masks_main is not None else {}), **({"rc_m": a.rc_m} if a.rc_m is not None else {}))
w = z if a.nospurious else np.column_stack([z, meta["conf"]])
k_sel = out["dimension"]["k_selected"]
row = dict(seed=a.seed, E=a.E, varmode=a.varmode, innov=a.innov, spurious=not a.nospurious, graph=a.graph, k_true=k_true,
           encoders=a.encoders, dim_c=a.dim_c, dim_rule=a.dim_rule, k_selected=k_sel, k_correct=k_sel == k_true, selection=out["stage_b"]["selection"], certificate=out["stage_b"]["certificate"],
           cross_rho=out["stage_b"]["cross_rho"], **{kk: v for kk, v in out["stage_b"].items() if kk.startswith("stability")},
           curve=json.dumps({str(kk): round(v[0], 4) for kk, v in out["dimension"].get("curve", {}).items()}), time=round(out["time"], 1))
zA = out["zA"]; kA = a.encoders.split("-")[0]           # main encoder
if a.rc_m is not None:
    row.update(rc_m=a.rc_m, rc_d=out["rc"]["d"])
row[f"r2_aff_{kA}"], row[f"mcc_aff_{kA}"] = rotation_diagnostics(w[200:], zA[200:])
row[f"mcc_raw_{kA}"] = mcc(w[200:], zA[200:])[0]
if a.b2_variant or a.b2_extra or a.save_latents:           # additional columns only with the optional flags
    from levers import cross_encoder_rho
    row.update(b2_variant=a.b2_variant or "pooled")
    if zA.shape[1] == w.shape[1]:                          # true rho_min of each of the two encoders
        row["rho_min_A"] = cross_encoder_rho(w[200:], zA[200:]); row["rho_min_B"] = cross_encoder_rho(w[200:], out["zB"][200:])
    if a.b2_extra:
        ie = out["stage_b_extra"]; row.update(b2_extra=a.b2_extra, selection_extra=ie["selection"], certificate_extra=ie["certificate"],
                                             **{kk + "_extra": v for kk, v in ie.items() if kk.startswith("stability")})
        if out["z_extra"] is not None:
            row["mcc_resolved_extra"] = mcc(w[200:], out["z_extra"][200:])[0]
    if a.save_latents:
        os.makedirs(os.path.join(HERE, "outputs", "z"), exist_ok=True)
        np.savez_compressed(os.path.join(HERE, "outputs", "z", f"pipeline_stage_a{a.tag}_s{a.seed}.npz"), zA=zA, zB=out["zB"], e=e, w=w)
if a.generator != "panel" or a.delta is not None or a.b3_pattern:   # additional columns only with these flags
    from itertools import combinations
    row.update(generator=a.generator, delta=a.delta, b3_pattern=a.b3_pattern or "", b3_extra=a.b3_extra or "", b3_scan=a.b3_scan, T=T_,
               b2_variant=a.b2_variant or "pooled")
    if a.delta is not None:                                # separation of the profiles: 0 = proportional profiles
        P_ = meta["noise_scale"] ** 2; P_ = P_ / np.linalg.norm(P_, axis=0, keepdims=True)
        row["b1_margin"] = float(min(1 - abs(P_[:, i] @ P_[:, j]) for i, j in combinations(range(P_.shape[1]), 2)))
    if a.generator == "bs":
        row.update(bs_spurious_perm_distance=json.dumps([round(v, 4) for v in meta["bs_spurious_perm_distance"]]),
                   bs_spurious_energy=json.dumps([float(v) for v in meta["bs_spurious_energy"]]))
    if a.b3_pattern:
        import levers as KL
        E_ = int(e.max()) + 1
        row["b3_reached"] = out["stage_b"]["selection"] == "B3"     # no stable B1 or B2: the pipeline moved on to B3
        row["pattern_mismatch"] = bool(out.get("pattern_mismatch", False))
        usable = zA.shape[1] == w.shape[1] and not row["pattern_mismatch"]
        # estimator-free reference: affine copy of the true z (condition number 5, noise 0.05, rng 1000 + seed)
        _g = np.random.default_rng(1000 + a.seed).standard_normal((4, 4)); _U, _, _Vt = np.linalg.svd(_g)
        _rng = np.random.default_rng(1000 + a.seed); _rng.standard_normal((4, 4))
        z_copy = z @ (_U @ np.diag(np.geomspace(1.0, 5.0, 4)) @ _Vt).T + 0.05 * _rng.standard_normal(z.shape)
        versions = [a.b3_pattern] + [s_ for s_ in (a.b3_extra.split("-") if a.b3_extra else []) if s_ != a.b3_pattern]
        for s_ in versions:
            if s_ not in patterns:
                row[f"pattern_missing_{s_}"] = True; continue
            m_ = patterns[s_]; row[f"n_zeros_{s_}"] = int((m_ == 0).sum())
            for source, zz, ok in (("pipeline", zA, usable), ("estimator_free", z_copy, True)):
                if not ok:
                    continue
                z3 = out["z"] if (source == "pipeline" and s_ == a.b3_pattern and row["b3_reached"]) else KL.apply_b3(zz, e, E_, m_)
                on = f"{source}_{s_}"
                row[f"f_{on}"] = B3P.off_pattern_energy(z3, e, E_, m_)
                row[f"mcc_b3_{on}"] = mcc(w[200:], z3[200:])[0]
                row[f"perm_distance_b3_{on}"] = B3P.perm_distance(w[200:], z3[200:])
                row[f"solved_b3_{on}"] = bool(row[f"mcc_b3_{on}"] >= 0.90)
                if a.generator == "bs":
                    cl, d_ = B3P.solution_class(w[200:], z3[200:], meta["M_spurious"])
                    row[f"class_{on}"] = cl; row.update({f"distance_{on}_{kk}": round(v, 4) for kk, v in d_.items()})
        if a.b3_scan > 0:                                      # distinct solutions that fit the correct pattern
            for source, zz, ok in (("pipeline", zA, usable), ("estimator_free", z_copy, True)):
                if not ok:
                    continue
                sols = B3P.b3_scan(zz, e, E_, patterns["correct"], n_starts=a.b3_scan, seed=a.seed)
                classes = [(B3P.solution_class(w[200:], s3[200:], meta["M_spurious"])[0] if a.generator == "bs"
                            else ("correct" if B3P.perm_distance(w[200:], s3[200:]) <= 0.25 else "other")) for _, s3 in sols]
                row[f"scan_{source}_n_matching"] = len(sols)
                row[f"scan_{source}_classes"] = json.dumps({c_: classes.count(c_) for c_ in sorted(set(classes))})
                row[f"scan_{source}_f_class"] = json.dumps([[round(fv, 6), c_] for (fv, _), c_ in zip(sols, classes)])
                n_s, n_y = B3P.b3_uniqueness(sols)                 # uniqueness check: B3 is accepted only with exactly one class
                row[f"b3uniq_{source}_n_classes"] = n_s; row[f"b3uniq_{source}_n_near"] = n_y
                if source == "pipeline":
                    row["selection_rule"] = "NOT_IDENTIFIABLE" if (row["selection"] == "B3" and n_s != 1) else row["selection"]
if out["z"] is not None:
    zr = out["z"]; m_, perm = mcc(w[200:], zr[200:]); row["mcc_resolved"] = m_
    if "edges" in out and len(perm) == w.shape[1] and zr.shape[1] == w.shape[1]:
        K = out["edges"]; kk = w.shape[1]
        # true A^aug and edge types (for scoring); conf is AR(1) with coefficient 0.8 when spurious columns are on
        At = [np.block([[Ae, np.zeros((4, 1))], [np.zeros((1, 4)), np.array([[0.8]])]]) if kk == 5 else Ae for Ae in meta["A_list"]]
        At = np.array(At); nz = np.abs(At) > 1e-9
        true_type = np.where(~nz.any(0), "none", np.where(nz.all(0) & np.all(np.isclose(At, At[0]), 0), "FIXED", "REGIME_DEPENDENT"))
        # align estimated components with the truth: estimated column perm[g] <-> true component g
        inv = {int(p): g for g, p in enumerate(perm)}
        type_h = np.empty((kk, kk), dtype=object)
        for i in range(kk):
            for j in range(kk):
                type_h[inv[i], inv[j]] = K["type"][i, j]
        present_h = np.zeros_like(nz)
        for r in range(nz.shape[0]):
            for i in range(kk):
                for j in range(kk):
                    present_h[r, inv[i], inv[j]] = K["present"][r, i, j]
        tp = (present_h & nz).sum(); fp = (present_h & ~nz).sum(); fn = (~present_h & nz).sum()
        row["support_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
        mask = true_type != "none"
        row["type_accuracy_on_edges"] = float((type_h[mask] == true_type[mask]).mean())
        row["none_accuracy"] = float((type_h[~mask] == "none").mean())
        if kk == 5:
            row["spurious_correct"] = (4 in [inv[i] for i in K["spurious_latent"]]) and len(K["spurious_latent"]) == 1
        os.makedirs(os.path.join(HERE, "outputs", "z"), exist_ok=True)
        np.savez_compressed(os.path.join(HERE, "outputs", "z", f"pipeline{a.tag}_s{a.seed}.npz"), w=w, zA=zA, zB=out["zB"], z=zr, e=e,
                            A_hat=K["A"], type=K["type"].astype(str), present=K["present"])
print(row, flush=True)
os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
pd.DataFrame([row]).to_csv(os.path.join(HERE, "outputs", f"pipeline{a.tag}_s{a.seed}.csv"), index=False)
