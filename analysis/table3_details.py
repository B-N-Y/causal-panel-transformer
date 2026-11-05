"""Details behind Table 3 and Section 8.5 of the paper (Appendix D).

1. Mean and standard deviation over seeds of the Stage-A error 1 - R^2_aff, per cell and method.
2. All tests of Table 3 (Stage A: exact sign test over the pooled seed-cell pairs of a row, or the exact Wilcoxon test where the
   table reports it; end to end: exact McNemar test) with Holm's correction over the whole table.
Solved counts (main criterion and item 3) use only cells with a lever; in the cell E4_none_g a "solved" run is a chance event.
3. Solved counts under the stricter, dataset-specific criterion MCC >= max(0.90, 99th percentile of the random-rotation MCC of the
   same dataset), using results/ROTATION_THRESHOLD/rotation_thresholds.csv (experiments/rotation_thresholds.py).

Usage: python analysis/table3_details.py [--out_dir results/TABLE3_DETAILS]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "analysis"))
from summarize_runs import mcnemar_exact, wilcoxon_exact  # noqa: E402

RES = os.path.join(HERE, "results")
SOLVED = 0.90
KEY = ["seed", "E", "k", "T", "varmode", "innov", "graph", "spurious"]

MAIN = [  # cell, CPT stem (R10), TDRL (run, stem), lever, iVAE file, iSDS stem, NCTRL stem
    ("E2-scale-G", "E2_scale_g", ("R1T", "tdrl_R1_E2_vscale_gauss"), "JD", None, "isds_v2_E2_scale_g", "nctrl_on_our_generator_v2_E2_scale_g"),
    ("E4-scale-G", "E4_scale_g", ("R1T", "tdrl_R1_E4_vscale_gauss"), "JD", "baseline_iVAE_R6_E4_vscale_gauss.csv",
     "isds_v2_E4_scale_g", "nctrl_on_our_generator_v2_E4_scale_g"),
    ("E4-none-G", "E4_none_g", ("R1T", "tdrl_R1_E4_vnone_gauss"), "JD", None, None, None),
    ("E4-none-t5", "E4_none_t5", ("R1T", "tdrl_R1_E4_vnone_t5"), "ICA", "baseline_iVAE_R6_E4_vnone_t5.csv",
     "isds_v2_E4_none_t5", "nctrl_on_our_generator_v2_E4_none_t5"),
    ("E4-scale-G-random", "E4_scale_g_rand", ("R5T", "tdrl_R5_rand_E4_vscale_gauss"), "JD", None, None, None),
]
HARD = [("k6", "JD"), ("k8D12", "JD"), ("T1000", "JD"), ("t5rand", "ICA"), ("noise", "JD")]
FRESH = [("E4_scale_g", "JD"), ("E4_none_t5", "ICA"), ("E4_none_g", "JD")]
NO_LEVER = {"E4_none_g"}   # cell without a lever: kept in the Stage-A comparisons, excluded from the solved counts


def load(run, pattern):
    fs = sorted(glob.glob(os.path.join(RES, run, pattern)))
    if not fs:
        raise FileNotFoundError(f"{run}/{pattern}")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).sort_values("seed").reset_index(drop=True)


def score(d, lever, design=False):
    if design and "MCC_design" in d:
        return d["MCC_design"]
    return d["mcc_jd"] if lever == "JD" else d["mcc_ica"]


def sign_test(a, b):
    """Two-sided exact sign test that a > b and a < b are equally likely (ties dropped)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n_pos, n_neg = int((a > b).sum()), int((a < b).sum())
    n = n_pos + n_neg
    return 1.0 if n == 0 else min(1.0, 2 * binom.cdf(min(n_pos, n_neg), n, 0.5))


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
    return adj


class Thresholds:
    def __init__(self):
        t = pd.read_csv(os.path.join(RES, "ROTATION_THRESHOLD", "rotation_thresholds.csv"))
        t["spurious"] = t.spurious.astype(str).str.lower() == "true"
        self.q = {tuple(r[c] for c in KEY): float(r["rot_q99"]) for _, r in t.iterrows()}

    def of(self, seed, E, k, T, varmode, innov, graph, spurious=False):
        return max(SOLVED, self.q[(seed, E, k, T, varmode, innov, graph, bool(spurious))])

    def strict(self, d, s, T=6000):
        """Boolean array: MCC s of each row of d at least the dataset-specific threshold (data given by the columns of d)."""
        th = [self.of(int(x.seed), int(x.E), int(x.k), int(getattr(x, "T", T) if pd.notna(getattr(x, "T", np.nan)) else T),
                      x.varmode, x.innov, getattr(x, "graph", "fixed")) for x in d.itertuples()]
        return np.asarray(s) >= np.asarray(th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(RES, "TABLE3_DETAILS"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    TH = Thresholds()
    err, tests, strict = [], [], []

    def add_err(block, cell, method, d):
        e_ = 1 - d.r2_affine
        err.append(dict(block=block, cell=cell, method=method, n=len(d), mean=e_.mean(), sd=e_.std(ddof=1)))

    # ---- main cells (seeds 0-9)
    pool = {nm: dict(c_r2=[], k_r2=[], cs=[], ks=[], cs_st=[], ks_st=[]) for nm in ("TDRL", "iVAE", "iSDS", "NCTRL")}
    for cell, stem, (trun, tstem), lever, ivf, isf, ncf in MAIN:
        c = load("R10", f"cpt2_stage_a_{stem}_s*.csv"); add_err("main", cell, "CPT", c)
        comps = {"TDRL": load(trun, f"{tstem}_s*.csv")}
        if ivf:
            comps["iVAE"] = load("R6B", ivf)
        if isf:
            comps["iSDS"] = load("ISDS2", f"{isf}_s*.csv")
        if ncf:
            n = load("R22v2", f"{ncf}_s*.csv"); comps["NCTRL"] = n[n.method == "NCTRL (end of training)"].reset_index(drop=True)
        cs, cst = score(c, lever) >= SOLVED, TH.strict(c, score(c, lever))
        for nm, d in comps.items():
            assert list(d.seed) == list(c.seed) == list(range(10)), (cell, nm)
            add_err("main", cell, nm, d)
            sk = score(d, lever, design=(nm == "NCTRL"))
            P = pool[nm]; P["c_r2"] += list(c.r2_affine); P["k_r2"] += list(d.r2_affine)
            if stem in NO_LEVER:
                continue
            P["cs"] += list(cs); P["ks"] += list(sk >= SOLVED); P["cs_st"] += list(cst); P["ks_st"] += list(TH.strict(c, sk))
    for nm, P in pool.items():
        tests.append(dict(row=f"main cells vs {nm}", column="Stage A", test="sign", p=sign_test(P["c_r2"], P["k_r2"])))
        tests.append(dict(row=f"main cells vs {nm}", column="end to end", test="McNemar", p=mcnemar_exact(P["cs"], P["ks"])[0]))
        strict.append(dict(row=f"main cells vs {nm}", cpt=sum(P["cs"]), comp=sum(P["ks"]), cpt_strict=sum(P["cs_st"]),
                           comp_strict=sum(P["ks_st"]), p_strict=mcnemar_exact(P["cs_st"], P["ks_st"])[0]))

    # ---- fresh seeds (200-209)
    pool = {nm: dict(c_r2=[], k_r2=[], cs=[], ks=[], cs_st=[], ks_st=[]) for nm in ("TDRL", "iSDS", "NCTRL", "PCA")}
    for tag, lever in FRESH:
        c = load("R10F", f"cpt2_stage_a_{tag}_R10F_s*.csv").set_index("seed", drop=False); add_err("fresh", tag, "CPT", c)
        n = load("NCTRL_R10F", f"nctrl_on_our_generator_R10F_{tag}_s*.csv")
        comps = {"TDRL": load("R10F", f"tdrl_hard_settings_{tag}_R10F_s*.csv"), "iSDS": load("ISDS_R10F", f"isds_R10F_{tag}_s*.csv"),
                 "NCTRL": n[n.method == "NCTRL (end of training)"].reset_index(drop=True),
                 "PCA": load("PCA_BASELINE", f"pca_baseline_{tag}_s200-209.csv")}
        for nm, d in comps.items():
            cc = c.loc[d.seed]; add_err("fresh", tag, nm, d)
            sc_, sk_ = score(cc, lever).values, d.MCC_design.values
            P = pool[nm]; P["c_r2"] += list(cc.r2_affine); P["k_r2"] += list(d.r2_affine)
            if tag in NO_LEVER:
                continue
            P["cs"] += list(sc_ >= SOLVED); P["ks"] += list(sk_ >= SOLVED)
            P["cs_st"] += list(TH.strict(cc, sc_)); P["ks_st"] += list(TH.strict(cc, sk_))
    for nm, P in pool.items():
        lab = "fresh seeds vs " + nm
        tests.append(dict(row=lab, column="Stage A", test="sign", p=sign_test(P["c_r2"], P["k_r2"])))
        tests.append(dict(row=lab, column="end to end", test="McNemar", p=mcnemar_exact(P["cs"], P["ks"])[0]))
        strict.append(dict(row=lab, cpt=sum(P["cs"]), comp=sum(P["ks"]), cpt_strict=sum(P["cs_st"]), comp_strict=sum(P["ks_st"]),
                           p_strict=mcnemar_exact(P["cs_st"], P["ks_st"])[0]))

    # ---- hard settings (seeds 0-9)
    pool = {nm: dict(c_r2=[], k_r2=[], cs=[], ks=[], cs_st=[], ks_st=[]) for nm in ("TDRL", "iSDS", "NCTRL")}
    for tag, lever in HARD:
        c = load("R14", f"cpt2_stage_a_3c_{tag}_s*.csv"); add_err("hard", tag, "CPT", c)
        n = load("K1v2_nctrl", f"nctrl_on_our_generator_k1v2_{tag}_s*.csv")
        comps = {"TDRL": load("K1v2_tdrl", f"tdrl_hard_settings_k1v2_{tag}_s*.csv"),
                 "NCTRL": n[n.method == "NCTRL (end of training)"].reset_index(drop=True),
                 "iSDS": load("K1v2_isds_T1000", "isds_k1v2_T1000_s*.csv") if tag == "T1000" else load("K1_isds", f"isds_k1_{tag}_s*.csv")}
        cs, cst = score(c, lever) >= SOLVED, TH.strict(c, score(c, lever))
        for nm, d in comps.items():
            assert list(d.seed) == list(c.seed) == list(range(10)), (tag, nm)
            add_err("hard", tag, nm, d)
            sk = score(d, lever, design=True)
            P = pool[nm]; P["c_r2"] += list(c.r2_affine); P["k_r2"] += list(d.r2_affine)
            P["cs"] += list(cs); P["ks"] += list(sk >= SOLVED); P["cs_st"] += list(cst); P["ks_st"] += list(TH.strict(c, sk))
    for nm, P in pool.items():
        lab = "hard settings vs " + nm
        tests.append(dict(row=lab, column="Stage A", test="sign", p=sign_test(P["c_r2"], P["k_r2"])))
        tests.append(dict(row=lab, column="end to end", test="McNemar", p=mcnemar_exact(P["cs"], P["ks"])[0]))
        strict.append(dict(row=lab, cpt=sum(P["cs"]), comp=sum(P["ks"]), cpt_strict=sum(P["cs_st"]), comp_strict=sum(P["ks_st"]),
                           p_strict=mcnemar_exact(P["cs_st"], P["ks_st"])[0]))

    # ---- generators of TDRL and NCTRL (Stage A only)
    t = load("R7T_FULL", "tdrl_generator_TDRL_R7TFULL_s*.csv")
    s_all = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, "R13", "tdrl_generator_CPT2_R13_*_s*.csv")))],
                      ignore_index=True)
    lin = s_all[(s_all.mech_lags == 1) & (s_all.mech_nl == 0)].sort_values("seed")
    sel = s_all.loc[s_all.groupby("seed").split_loss.idxmin()].sort_values("seed")
    for nm, d in (("TDRL", t), ("CPT, linear prior", lin), ("CPT, transition selection", sel)):
        add_err("TDRL generator", "-", nm, d)
    tests.append(dict(row="TDRL generator, linear prior", column="Stage A", test="Wilcoxon", p=wilcoxon_exact(lin.r2_affine, t.r2_affine)[0]))
    tests.append(dict(row="TDRL generator, selection", column="Stage A", test="Wilcoxon", p=wilcoxon_exact(sel.r2_affine, t.r2_affine)[0]))
    old = load("R15c", "nctrl_on_nctrl_generator_s*.csv"); new = load("R15c_s7_v2", "nctrl_on_nctrl_generator_v2_s7.csv")
    prim = old[old.method == "NCTRL (end of training)"].reset_index(drop=True)
    s7 = new[new.method == "NCTRL (end of training)"].iloc[0]
    for col in ("MCC", "r2_affine", "mcc_affine"):
        prim.loc[prim.seed == 7, col] = s7[col]
    cs_ = load("R15", "cpt_on_nctrl_generator_CPT2s_s*.csv").query("method == 'CPT2s (selected)'").reset_index(drop=True)
    add_err("NCTRL generator", "-", "NCTRL", prim); add_err("NCTRL generator", "-", "CPT", cs_)
    tests.append(dict(row="NCTRL generator", column="Stage A", test="Wilcoxon", p=wilcoxon_exact(cs_.r2_affine, prim.r2_affine)[0]))

    E_ = pd.DataFrame(err); T_ = pd.DataFrame(tests); S_ = pd.DataFrame(strict)
    T_["p_holm"] = holm(T_.p)
    E_.to_csv(os.path.join(a.out_dir, "stage_a_error_mean_sd.csv"), index=False)
    T_.to_csv(os.path.join(a.out_dir, "table3_tests_holm.csv"), index=False)
    S_.to_csv(os.path.join(a.out_dir, "solved_strict_table3.csv"), index=False)

    # ---- Section 8.5: reported solutions under the strict criterion
    rows = []
    for run, extra in (("PIPE150", True), ("PIPE160", False)):
        d = load(run, "pipeline_*_s*.csv"); sfx = "_extra" if extra else ""
        d = d.assign(cert=d["certificate" + sfx], m=d["mcc_resolved" + sfx], k=d.k_true - d.spurious.astype(str).str.lower().eq("true"),
                     T=6000)
        d["spurious"] = d.spurious.astype(str).str.lower() == "true"
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["th"] = [TH.of(int(x.seed), int(x.E), int(x.k), 6000, x.varmode, x.innov, x.graph, bool(x.spurious)) for x in d.itertuples()]
    lv = d[~((d.varmode == "none") & (d.innov == "gauss"))]
    rep = lv[lv.cert != "NOT_IDENTIFIABLE"]
    dl = load("DELTA_PIPE", "pipeline_delta_d*_s*.csv").assign(T=6000)
    dl["th"] = [TH.of(int(x.seed), int(x.E), int(x.k_true), 6000, x.varmode, x.innov, x.graph) for x in dl.itertuples()]
    rd = dl[dl.certificate != "NOT_IDENTIFIABLE"]
    out = pd.DataFrame([dict(evaluation="Section 8.5, 100 datasets with a lever", reported=len(rep), correct=int((rep.m >= SOLVED).sum()),
                             correct_strict=int((rep.m >= rep.th).sum()), threshold_max=rep.th.max()),
                        dict(evaluation="Section 8.5, nearly proportional profiles", reported=len(rd),
                             correct=int((rd.mcc_resolved >= SOLVED).sum()), correct_strict=int((rd.mcc_resolved >= rd.th).sum()),
                             threshold_max=rd.th.max())])
    out.to_csv(os.path.join(a.out_dir, "solved_strict_pipeline.csv"), index=False)

    pd.set_option("display.width", 200)
    print(E_.round(4).to_string(index=False)); print(); print(T_.to_string(index=False)); print()
    print(S_.to_string(index=False)); print(); print(out.to_string(index=False))


if __name__ == "__main__":
    main()
