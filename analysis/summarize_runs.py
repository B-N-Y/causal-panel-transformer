# -*- coding: utf-8 -*-
"""Summary tables of the comparison runs (CPT versus the competing methods).

Usage:  python analysis/summarize_runs.py [--results DIR] [section ...]
        sections: r10f k1v2 k4v2 r22v2 r15c (default: all)
Reads the result CSV files from DIR and its subfolders (default: outputs/; for the shipped results: --results results) and
prints Markdown tables; writes no files.

Rules:
- primary score after Stage B = the lever implied by the design (scale variation -> JD, no scale variation with t5 innovations
  -> ICA, no lever with Gaussian innovations -> JD); resolved = this score >= .90; max(JD, ICA) is reported only as an
  'oracle lever choice (upper bound)'.
- paired test: two-sided EXACT Wilcoxon signed-rank test; zero differences are dropped, tied |differences| receive average
  ranks, and the exact distribution is obtained by complete enumeration over the average ranks.
- counts of resolved runs: exact McNemar test on the discordant pairs (two-sided binomial).
- W/T/L: CPT better with p <= .05 -> W; competitor better with p <= .05 -> L; otherwise T.
"""
import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import binom

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "outputs")          # replaced by --results in __main__
RESOLVED = 0.90

pd.set_option("display.width", 250)


# ---------------------------------------------------------------- helpers
def load(pattern, folder=None):
    fs = sorted(glob.glob(os.path.join(folder or RES, "**", pattern), recursive=True))
    if not fs:
        return pd.DataFrame()
    out = []
    for f in fs:
        d = pd.read_csv(f)
        d["_file"] = os.path.basename(f)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def wilcoxon_exact(a, b):
    """Two-sided exact Wilcoxon test (zeros dropped, average ranks, complete enumeration). Returns (p, n_effective)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    d = d[np.abs(d) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0, 0
    r = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    wplus = r[d > 0].sum()
    mu = r.sum() / 2.0
    obs = abs(wplus - mu)
    if n <= 20:
        cnt = 0
        tot = 0
        # all sign assignments
        for signs in itertools.product((0, 1), repeat=n):
            s = np.dot(signs, r)
            tot += 1
            if abs(s - mu) >= obs - 1e-9:
                cnt += 1
        return cnt / tot, n
    raise ValueError("n > 20: exact enumeration is not implemented")


def mcnemar_exact(sa, sb):
    sa = np.asarray(sa, bool)
    sb = np.asarray(sb, bool)
    b = int((sa & ~sb).sum())
    c = int((~sa & sb).sum())
    if b + c == 0:
        return 1.0, b, c
    p = min(1.0, 2 * binom.cdf(min(b, c), b + c, 0.5))
    return p, b, c


def wtl(n_cpt_better, n_comp_better, p, larger_better=True):
    if p <= 0.05:
        return "W" if n_cpt_better > n_comp_better else "L"
    return "T"


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def f4(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def fp(p):
    if p is None or np.isnan(p):
        return "n/a"
    return f"{p:.3f}" if p >= 0.001 else "<.001"


def compare(cpt, comp, larger_better=True):
    """cpt, comp: seed-aligned arrays. Returns (cpt_better, comp_better, p, W/T/L)."""
    a = np.asarray(cpt, float)
    b = np.asarray(comp, float)
    if not larger_better:
        a, b = -a, -b
    ci = int((a - b > 1e-12).sum())
    ri = int((b - a > 1e-12).sum())
    p, _ = wilcoxon_exact(a, b)
    return ci, ri, p, wtl(ci, ri, p)


def md(df):
    cols = list(df.columns)
    s = "| " + " | ".join(str(c) for c in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        s += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return s


def heading(t):
    print("\n\n## " + t + "\n")


def design(d, lever):
    return d["mcc_jd"] if lever == "JD" else d["mcc_ica"]


# ============================================================== R10F
def r10f():
    heading("R10F: three E=4 cells on fresh seeds 200-209 (CPT vs TDRL)")
    cells = [("E4_scale_g", "E4-scale (B1)", "JD"), ("E4_none_t5", "E4-t5 (B2)", "ICA"),
             ("E4_none_g", "E4-no-lever (JD)", "JD")]
    rows = []
    tests = []
    for tag, name, lever in cells:
        c = load(f"cpt2_stage_a_{tag}_R10F_s*.csv").sort_values("seed").reset_index(drop=True)
        t = load(f"tdrl_hard_settings_{tag}_R10F_s*.csv").sort_values("seed").reset_index(drop=True)
        assert list(c.seed) == list(t.seed) == list(range(200, 210)), tag
        cB = design(c, lever)
        tB = t["MCC_design"]
        assert (t.lever_design == lever).all()
        r2 = compare(c.r2_affine, t.r2_affine)
        mb = compare(cB, tB)
        mr = compare(c.MCC, t.MCC)
        ma = compare(c.mcc_affine, t.mcc_affine)
        sc, st = (cB >= RESOLVED), (tB >= RESOLVED)
        pm, b_, c_ = mcnemar_exact(sc, st)
        rows.append({
            "Cell": name,
            "Raw MCC CPT / TDRL": f"{c.MCC.mean():.3f} / {t.MCC.mean():.3f}",
            "Affine MCC": f"{c.mcc_affine.mean():.3f} / {t.mcc_affine.mean():.3f}",
            "Affine R² (worst)": f"{c.r2_affine.mean():.3f} ({c.r2_affine.min():.3f}) / {t.r2_affine.mean():.3f} ({t.r2_affine.min():.3f})",
            "R²: CPT better; p": f"{r2[0]}/10; {fp(r2[2])}",
            "MCC_B CPT / TDRL": f"{cB.mean():.3f} / {tB.mean():.3f}",
            "MCC_B: CPT better; p": f"{mb[0]}/10; {fp(mb[2])}",
            "Resolved CPT / TDRL": f"{int(sc.sum())} / {int(st.sum())}",
        })
        tests.append({
            "Cell": name,
            "Raw MCC: CPT better; p": f"{mr[0]}/10; {fp(mr[2])}",
            "Affine MCC: CPT better; p": f"{ma[0]}/10; {fp(ma[2])}",
            "McNemar (b, c; p)": f"{b_}, {c_}; {fp(pm)}",
            "Oracle lever: MCC_B† CPT / TDRL": f"{np.maximum(c.mcc_jd, c.mcc_ica).mean():.3f} / {t.B.mean():.3f}",
            "Oracle resolved": f"{int((np.maximum(c.mcc_jd, c.mcc_ica) >= RESOLVED).sum())} / {int((t.B >= RESOLVED).sum())}",
            "TDRL oracle checkpoint: R² / MCC_design": f"{t.oracle_r2_affine.mean():.3f} / {t.oracle_MCC_design.mean():.3f}",
            "ρ_min CPT": f"{c.rho_min.mean():.3f}",
            "Time (s) CPT / TDRL": f"{c.time.mean():.0f} / {t.time.mean():.0f}",
        })
    print(md(pd.DataFrame(rows)))
    print()
    print(md(pd.DataFrame(tests)))


# ============================================================== K1v2
K1_SETTINGS = [("k6", "k=6, D=12, random graph", "JD", dict(k=6, D=12, T=6000, obs_noise=0.05, graph="random", E=4)),
               ("k8D12", "k=8, D=12", "JD", dict(k=8, D=12, T=6000, obs_noise=0.05, graph="random", E=4)),
               ("T1000", "T=1000", "JD", dict(k=4, D=12, T=1000, obs_noise=0.05, graph="fixed", E=4)),
               ("t5rand", "E=6, t5, random graph", "ICA", dict(k=4, D=12, T=6000, obs_noise=0.05, graph="random", E=6)),
               ("noise", "observation noise 0.3", "JD", dict(k=4, D=12, T=6000, obs_noise=0.3, graph="fixed", E=4))]


def k1_cpt(tag, lever):
    c = load(f"cpt2_stage_a_3c_{tag}_s*.csv").sort_values("seed").reset_index(drop=True)
    c["MCC_design"] = design(c, lever)
    c["iVAE_MCC_design"] = c["iVAE_mcc_jd"] if lever == "JD" else c["iVAE_mcc_ica"]
    return c


def k1v2():
    heading("K1v2: hard settings; TDRL, NCTRL and iSDS vs CPT (R14)")
    main = []
    extra = []
    quality = []
    pool = {"TDRL": ([], []), "NCTRL": ([], []), "iSDS": ([], [])}
    for tag, name, lever, setting in K1_SETTINGS:
        c = k1_cpt(tag, lever)
        assert list(c.seed) == list(range(10))
        t = load(f"tdrl_hard_settings_k1v2_{tag}_s*.csv").sort_values("seed").reset_index(drop=True)
        n_all = load(f"nctrl_on_our_generator_k1v2_{tag}_s*.csv")
        n = n_all[n_all.method == "NCTRL (end of training)"].sort_values("seed").reset_index(drop=True)
        no = n_all[n_all.method != "NCTRL (end of training)"].sort_values("seed").reset_index(drop=True)
        if tag == "T1000":
            i = load("isds_k1v2_T1000_s*.csv")
        else:
            i = load(f"isds_k1_{tag}_s*.csv")            # jobs/K1_isds.txt; valid for the T=6000 settings
            if len(i):
                i["MCC_design"] = design(i, lever)
        i = i.sort_values("seed").reset_index(drop=True) if len(i) else i
        # setting check
        for nm, d in [("TDRL", t), ("NCTRL", n)]:
            for kk, v in setting.items():
                if kk in d.columns:
                    assert (d[kk] == v).all(), (nm, tag, kk)
            assert (d.lever_design == lever).all(), (nm, tag)
        if len(i):
            for kk, v in setting.items():
                if kk in i.columns:
                    assert (i[kk] == v).all(), ("iSDS", tag, kk)
        row = {"Setting (lever)": f"{name} ({lever})",
               "CPT R² (worst); resolved": f"{c.r2_affine.mean():.3f} ({c.r2_affine.min():.3f}); {int((c.MCC_design >= RESOLVED).sum())}",
               "iVAE R²; resolved": f"{c.iVAE_r2_affine.mean():.3f}; {int((c.iVAE_MCC_design >= RESOLVED).sum())}"}
        for nm, d in [("TDRL", t), ("iSDS", i), ("NCTRL", n)]:
            if len(d) == 10:
                row[f"{nm} R² (worst); resolved"] = (f"{d.r2_affine.mean():.3f} ({d.r2_affine.min():.3f}); "
                                                     f"{int((d.MCC_design >= RESOLVED).sum())}")
            elif len(d):
                row[f"{nm} R² (worst); resolved"] = f"ONLY {len(d)} seeds ({list(d.seed)}): {d.r2_affine.mean():.3f}; {int((d.MCC_design >= RESOLVED).sum())}/{len(d)}"
            else:
                row[f"{nm} R² (worst); resolved"] = "no data"
        main.append(row)
        for nm, d in [("TDRL", t), ("iSDS", i), ("NCTRL", n)]:
            if len(d) != 10:
                continue
            assert list(d.seed) == list(range(10))
            r2 = compare(c.r2_affine, d.r2_affine)
            ma = compare(c.mcc_affine, d.mcc_affine)
            mr = compare(c.MCC, d.MCC)
            mb = compare(c.MCC_design, d.MCC_design)
            pm, b_, c_ = mcnemar_exact(c.MCC_design >= RESOLVED, d.MCC_design >= RESOLVED)
            pool[nm][0].extend(list(c.MCC_design >= RESOLVED))
            pool[nm][1].extend(list(d.MCC_design >= RESOLVED))
            extra.append({"Setting": name, "Competitor": nm,
                          "Affine R² CPT/competitor (CPT better; p; W/T/L)": f"{c.r2_affine.mean():.3f}/{d.r2_affine.mean():.3f} ({r2[0]}/10; {fp(r2[2])}; {r2[3]})",
                          "Affine MCC": f"{c.mcc_affine.mean():.3f}/{d.mcc_affine.mean():.3f} ({ma[0]}/10; {fp(ma[2])}; {ma[3]})",
                          "Raw MCC": f"{c.MCC.mean():.3f}/{d.MCC.mean():.3f} ({mr[0]}/10; {fp(mr[2])}; {mr[3]})",
                          "MCC_B design": f"{c.MCC_design.mean():.3f}/{d.MCC_design.mean():.3f} ({mb[0]}/10; {fp(mb[2])}; {mb[3]})",
                          "Resolved (McNemar b,c; p)": f"{int((c.MCC_design >= RESOLVED).sum())}/{int((d.MCC_design >= RESOLVED).sum())} ({b_},{c_}; {fp(pm)})",
                          "Oracle lever MCC_B†; resolved": f"{c.B.mean():.3f}/{d.B.mean():.3f}; {int((c.B >= RESOLVED).sum())}/{int((d.B >= RESOLVED).sum())}",
                          })
        # quality / additional information
        q_row = {"Setting": name,
                 "TDRL time (s)": f"{t.time.mean():.0f}",
                 "TDRL oracle checkpoint: R² / MCC_B": f"{t.oracle_r2_affine.mean():.3f} / {t.oracle_MCC_design.mean():.3f}",
                 "NCTRL regime accuracy": f"{n.regime_accuracy.mean():.2f}",
                 "NCTRL budget stop": f"{int(n.stopped_by_budget.sum())}/10",
                 "NCTRL epochs (done/planned)": f"{int(n.epochs_done.min())}–{int(n.epochs_done.max())} / {int(n.epochs_planned.min())}–{int(n.epochs_planned.max())}",
                 "NCTRL updates": f"{int(n.updates_done.min())}–{int(n.updates_done.max())}",
                 "NCTRL n_val": f"{int(n.n_val.iloc[0])}",
                 "NCTRL automatic B (MCC_stage_b; resolved)": f"{n.MCC_stage_b.mean():.3f}; {int((n.MCC_stage_b >= RESOLVED).sum())}",
                 "NCTRL oracle checkpoint R²; MCC_B": f"{no.r2_affine.mean():.3f}; {no.MCC_design.mean():.3f}",
                 "NCTRL time (s)": f"{n.time.mean():.0f}",
                 "CPT time (s)": f"{c.time.mean():.0f}"}
        if len(i):
            q_row["iSDS regime acc.; updates; epochs"] = f"{i.regime_accuracy.mean():.2f}; {int(i.updates.iloc[0])}; {int(i.epochs.iloc[0])}"
        quality.append(q_row)
    print(md(pd.DataFrame(main)))
    print("\n### Paired tests (CPT minus competitor; 10 seeds; two-sided exact Wilcoxon; resolved: exact McNemar)\n")
    print(md(pd.DataFrame(extra)))
    print("\n### Pooled number of resolved runs (over the complete settings available; exact McNemar)\n")
    rows = []
    for nm, (a, b) in pool.items():
        if not a:
            continue
        p, b_, c_ = mcnemar_exact(a, b)
        rows.append({"Competitor": nm, "settings×seeds": len(a), "CPT resolved": int(np.sum(a)), "Competitor resolved": int(np.sum(b)),
                     "b (CPT only)": b_, "c (competitor only)": c_, "p": fp(p)})
    print(md(pd.DataFrame(rows)))
    print("\n### Data quality and additional information\n")
    print(md(pd.DataFrame(quality)))
    print("\niSDS source: T=1000 from jobs/K1v2_isds_T1000.txt (`isds_k1v2_T1000_*`); the other four settings from "
          "jobs/K1_isds.txt (`isds_k1_*`).")


# ============================================================== K4v2
def k4v2():
    heading("K4v2: sensitivity to training settings (E=4, scale, Gaussian, seeds 0-4) vs the CPT default (R10, same seeds)")
    c = load("cpt2_stage_a_E4_scale_g_s*.csv").sort_values("seed")
    c = c[c.seed < 5].reset_index(drop=True)
    c["MCC_design"] = c.mcc_jd
    rows = [{"Method / setting": "CPT default (R10)", "Affine R² (worst)": f"{c.r2_affine.mean():.4f} ({c.r2_affine.min():.3f})",
             "Affine MCC": f"{c.mcc_affine.mean():.4f}", "Raw MCC": f"{c.MCC.mean():.3f}",
             "MCC_B (JD)": f"{c.MCC_design.mean():.3f}", "Resolved": f"{int((c.MCC_design >= RESOLVED).sum())}/5",
             "R²: CPT better; p": "n/a", "Budget stop": "n/a", "Time (s)": f"{c.time.mean():.0f}"}]
    best = {}
    for nm, pattern in [("TDRL", "tdrl_hard_settings_k4v2_lr*_ep*_s*.csv"), ("NCTRL", "nctrl_on_our_generator_k4v2_lr*_g*_s*.csv")]:
        d = load(pattern)
        if nm == "NCTRL":
            d = d[d.method == "NCTRL (end of training)"]
        d["setting"] = d._file.str.extract(r"k4v2_(lr[0-9.]+_(?:ep|g)[0-9k]+)_s")[0]
        for setting, g in d.groupby("setting"):
            g = g.sort_values("seed").reset_index(drop=True)
            assert list(g.seed) == list(range(5)), (nm, setting)
            assert (g.E == 4).all() and (g.varmode == "scale").all() and (g.innov == "gauss").all()
            r2 = compare(c.r2_affine, g.r2_affine)
            ma = compare(c.mcc_affine, g.mcc_affine)
            mb = compare(c.MCC_design, g.MCC_design)
            stop = (f"{int(g.stopped_by_budget.sum())}/5 (epochs {int(g.epochs_done.min())}–{int(g.epochs_done.max())}/{int(g.epochs_planned.max())}; "
                    f"updates {int(g.updates_done.min())}–{int(g.updates_done.max())})") if nm == "NCTRL" else f"n/a (epochs {int(g.epochs.iloc[0])}, lr {g.lr.iloc[0]})"
            rows.append({"Method / setting": f"{nm} {setting}", "Affine R² (worst)": f"{g.r2_affine.mean():.4f} ({g.r2_affine.min():.3f})",
                         "Affine MCC": f"{g.mcc_affine.mean():.4f}", "Raw MCC": f"{g.MCC.mean():.3f}",
                         "MCC_B (JD)": f"{g.MCC_design.mean():.3f}", "Resolved": f"{int((g.MCC_design >= RESOLVED).sum())}/5",
                         "R²: CPT better; p": f"{r2[0]}/5; {fp(r2[2])}; aff.MCC {ma[0]}/5; MCC_B {mb[0]}/5, p {fp(mb[2])}",
                         "Budget stop": stop, "Time (s)": f"{g.time.mean():.0f}"})
            for m in ("r2_affine", "mcc_affine", "MCC", "MCC_design"):
                if m not in best.get(nm, {}) or g[m].mean() > best[nm][m][1]:
                    best.setdefault(nm, {})[m] = (setting, g[m].mean(), g[m].to_numpy())
    print(md(pd.DataFrame(rows)))
    print("\n### Each competitor at its BEST setting per metric (chosen with the true z, in favour of the competitor) vs the CPT default\n")
    rr = []
    for nm, dd in best.items():
        for m, (setting, mu, arr) in dd.items():
            cc = c[m].to_numpy()
            ci, ri, p, w = compare(cc, arr)
            rr.append({"Competitor": nm, "Metric": m, "Best setting": setting, "Competitor mean": f"{mu:.4f}", "CPT mean": f"{cc.mean():.4f}",
                       "Difference (CPT − competitor)": f"{cc.mean() - mu:+.4f}", "CPT better / competitor better (5 seeds)": f"{ci}/{ri}",
                       "p (n=5; smallest possible .0625)": fp(p), "Competitor above CPT?": "YES" if mu > cc.mean() else "no"})
    print(md(pd.DataFrame(rr)))
    # consistency: the default TDRL setting (lr 5e-4, 200 epochs) must agree with the R1 TDRL runs
    t0 = load("tdrl_hard_settings_k4v2_lr0.0005_ep200_s*.csv").sort_values("seed").reset_index(drop=True)
    r1 = load("tdrl_R1_E4_vscale_gauss_s*.csv").sort_values("seed")
    r1 = r1[r1.seed < 5].reset_index(drop=True)
    if len(r1):
        print(f"\nConsistency: K4v2 TDRL default setting (lr 5e-4, 200 epochs) seeds 0-4 R² {t0.r2_affine.mean():.4f} vs R1 TDRL "
              f"{r1.r2_affine.mean():.4f}; largest per-seed |difference| {np.abs(t0.r2_affine.values - r1.r2_affine.values).max():.4f}")
    n0 = load("nctrl_on_our_generator_k4v2_lr0.0005_g120k_s*.csv")
    n0 = n0[n0.method == "NCTRL (end of training)"].sort_values("seed").reset_index(drop=True)
    r22 = load("nctrl_on_our_generator_E4_scale_g_s*.csv")
    r22 = r22[(r22.method == "NCTRL (end of training)") & (r22.seed < 5)].sort_values("seed").reset_index(drop=True)
    print(f"Consistency: K4v2 NCTRL default (lr 5e-4, 120k) seeds 0-4 R² {n0.r2_affine.mean():.4f} vs R22 same seeds {r22.r2_affine.mean():.4f}")


# ============================================================== R22v2
def r22v2():
    heading("R22v2: NCTRL on our generator, design-fixed lever (same seeds as the R10 CPT runs)")
    cells = [("E4_scale_g", "E=4, scale, Gaussian (JD)", "JD"), ("E4_none_t5", "E=4, none, t5 (ICA)", "ICA"),
             ("E2_scale_g", "E=2, scale, Gaussian (JD)", "JD")]
    rows = []
    rows2 = []
    hc, hn, hno = [], [], []
    for tag, name, lever in cells:
        c = load(f"cpt2_stage_a_{tag}_s*.csv").sort_values("seed").reset_index(drop=True)
        c["MCC_design"] = design(c, lever)
        na = load(f"nctrl_on_our_generator_v2_{tag}_s*.csv")
        n = na[na.method == "NCTRL (end of training)"].sort_values("seed").reset_index(drop=True)
        no = na[na.method != "NCTRL (end of training)"].sort_values("seed").reset_index(drop=True)
        oa = load(f"nctrl_on_our_generator_{tag}_s*.csv")
        o = oa[oa.method == "NCTRL (end of training)"].sort_values("seed").reset_index(drop=True)
        assert (n.lever_design == lever).all()
        r2 = compare(c.r2_affine, n.r2_affine)
        ma = compare(c.mcc_affine, n.mcc_affine)
        mr = compare(c.MCC, n.MCC)
        mb = compare(c.MCC_design, n.MCC_design)
        pm, b_, c_ = mcnemar_exact(c.MCC_design >= RESOLVED, n.MCC_design >= RESOLVED)
        hc += list(c.MCC_design >= RESOLVED)
        hn += list(n.MCC_design >= RESOLVED)
        hno += list(n.B >= RESOLVED)
        rows.append({"Cell": name,
                     "Affine R² CPT/NCTRL (CPT better; p)": f"{c.r2_affine.mean():.3f}/{n.r2_affine.mean():.3f} ({r2[0]}/10; {fp(r2[2])})",
                     "Affine MCC": f"{c.mcc_affine.mean():.3f}/{n.mcc_affine.mean():.3f} ({ma[0]}/10; {fp(ma[2])})",
                     "Raw MCC": f"{c.MCC.mean():.3f}/{n.MCC.mean():.3f} ({mr[0]}/10; {fp(mr[2])})",
                     "MCC_B design": f"{c.MCC_design.mean():.3f}/{n.MCC_design.mean():.3f} ({mb[0]}/10; {fp(mb[2])})",
                     "Resolved (b,c; p)": f"{int((c.MCC_design >= RESOLVED).sum())}/{int((n.MCC_design >= RESOLVED).sum())} ({b_},{c_}; {fp(pm)})",
                     "NCTRL oracle lever MCC_B†; resolved": f"{n.B.mean():.3f}; {int((n.B >= RESOLVED).sum())}",
                     "NCTRL regime": f"{n.regime_accuracy.mean():.2f}"})
        rows2.append({"Cell": name,
                      "v2 automatic (MCC_stage_b; resolved; B1 chosen)": f"{n.MCC_stage_b.mean():.3f}; {int((n.MCC_stage_b >= RESOLVED).sum())}; {int((n.stage_b_choice == 'B1').sum())}",
                      "R22 automatic (MCC_stage_b; resolved)": f"{o.MCC_stage_b.mean():.3f}; {int((o.MCC_stage_b >= RESOLVED).sum())}",
                      "v2 vs R22: mean R² / largest per-seed absolute difference": f"{n.r2_affine.mean():.3f} vs {o.r2_affine.mean():.3f} / {np.abs(n.r2_affine.values - o.r2_affine.values).max():.3f}",
                      "v2 vs R22: mean raw MCC": f"{n.MCC.mean():.3f} vs {o.MCC.mean():.3f}",
                      "oracle checkpoint R²; raw; MCC_B": f"{no.r2_affine.mean():.3f}; {no.MCC.mean():.3f}; {no.MCC_design.mean():.3f}",
                      "budget stop; epochs; updates": f"{int(n.stopped_by_budget.sum())}/10; {int(n.epochs_done.min())}/{int(n.epochs_planned.min())}; {int(n.updates_done.min())}–{int(n.updates_done.max())}",
                      "time (s) CPT / NCTRL": f"{c.time.mean():.0f} / {n.time.mean():.0f}"})
    print(md(pd.DataFrame(rows)))
    p, b_, c_ = mcnemar_exact(hc, hn)
    print(f"\nPooled (30 seeds): CPT {sum(hc)} / NCTRL {sum(hn)} resolved (design-fixed); exact McNemar b={b_}, c={c_}, p={fp(p)}. "
          f"NCTRL with the oracle lever: {sum(hno)}/30.\n")
    print(md(pd.DataFrame(rows2)))


# ============================================================== R15c
def r15c():
    heading("R15c: NCTRL on its own generator, oracle stopping OFF (seed 7 from jobs/R15c_s7_v2.txt); 10 seeds")
    old = load("nctrl_on_nctrl_generator_s*.csv")
    new = load("nctrl_on_nctrl_generator_v2_s7.csv")
    for lab, pre in [("primary", "NCTRL (end of training)"), ("oracle", "NCTRL (authors'")]:
        o = old[old.method.str.startswith(pre)].sort_values("seed").reset_index(drop=True)
        nw = new[new.method.str.startswith(pre)]
        m = o.copy()
        m.loc[m.seed == 7, ["MCC", "r2_affine", "mcc_affine", "regime_accuracy", "epoch", "time"]] = \
            nw[["MCC", "r2_affine", "mcc_affine", "regime_accuracy", "epoch", "time"]].values
        for nm, d in [("seed 7 from jobs/R15c.txt", o), ("seed 7 from jobs/R15c_s7_v2.txt", m)]:
            print(f"- {lab}, {nm}: affine R² {d.r2_affine.mean():.4f} (worst {d.r2_affine.min():.3f}); affine MCC {d.mcc_affine.mean():.4f}; "
                  f"raw MCC {d.MCC.mean():.4f} (range {d.MCC.min():.2f}–{d.MCC.max():.2f}); raw ≥ .90: {int((d.MCC >= .9).sum())}/10; "
                  f"regime {d.regime_accuracy.mean():.3f}; time {d.time.mean():.0f} s")
        if lab == "primary":
            prim = m
    s7o = old[(old.seed == 7) & old.method.str.startswith("NCTRL (end of training")].iloc[0]
    s7n = new[new.method.str.startswith("NCTRL (end of training")].iloc[0]
    print(f"\nSeed 7: R15c raw {s7o.MCC:.4f}, R² {s7o.r2_affine:.4f}, epoch {s7o.epoch}, {s7o.time:.0f} s  ->  v2 raw {s7n.MCC:.4f}, "
          f"R² {s7n.r2_affine:.4f}, epoch {s7n.epoch}/{s7n.epochs_planned}, {s7n.time:.0f} s, budget stop {s7n.stopped_by_budget}, "
          f"oracle stopped {s7n.oracle_stopped}")
    # paired tests
    cs = load("cpt_on_nctrl_generator_CPT2s_s*.csv")
    cs = cs[cs.method.str.contains("selected")].sort_values("seed").reset_index(drop=True)
    print("\nPaired tests (CPT minus NCTRL primary, 10 seeds; seed 7 from the v2 run):")
    for nm, c in [("CPT-sel 20k (R15)", cs)]:
        for mt in ("r2_affine", "mcc_affine", "MCC"):
            ci, ri, p, w = compare(c[mt], prim[mt])
            print(f"- {nm} vs NCTRL, {mt}: {c[mt].mean():.4f} vs {prim[mt].mean():.4f}; CPT better {ci}, NCTRL better {ri}, equal {10 - ci - ri}; p = {fp(p)} ({w})")
    for nm, f in [("TDRL 18 epochs", "cpt_on_nctrl_generator_TDRL_s*.csv")]:
        t = load(f)
        if not len(t):
            continue
        t = t.sort_values("seed").reset_index(drop=True)
        ci, ri, p, w = compare(prim.MCC, t.MCC)
        print(f"- NCTRL vs {nm}, raw MCC (information): {prim.MCC.mean():.3f} vs {t.MCC.mean():.3f}; NCTRL better {ci}/10; p = {fp(p)}")


SECTIONS = ["r10f", "k1v2", "k4v2", "r22v2", "r15c"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Summary tables of the comparison runs.")
    ap.add_argument("--results", default=os.path.join(HERE, "outputs"),
                    help="folder with the result CSV files, searched recursively (default: outputs/; shipped results: results/)")
    ap.add_argument("sections", nargs="*", help="sections to print (default: all): " + " ".join(SECTIONS))
    args = ap.parse_args()
    unknown = [s for s in args.sections if s not in SECTIONS]
    if unknown:
        ap.error(f"unknown section(s): {unknown}")
    RES = args.results
    for s in args.sections or SECTIONS:
        globals()[s]()
