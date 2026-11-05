"""Summary of end-to-end pipeline results (evaluate_pipeline.py).
Usage: python experiments/summarize_pipeline.py '<glob>' [--extra]
--extra: summarize the second B2 variant (*_extra columns, written with --b2_extra) instead of the primary one; the edge
columns refer to the primary variant and are then left out."""
import sys, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
extra = "--extra" in sys.argv[2:]
d = pd.concat([pd.read_csv(f).assign(file=f) for f in glob.glob(sys.argv[1])], ignore_index=True)
if extra:
    d["certificate"], d["mcc_resolved"], d["selection"] = d.certificate_extra, d.mcc_resolved_extra, d.selection_extra
    for c in ("support_F1", "type_accuracy_on_edges", "spurious_correct"):
        d[c] = np.nan
d["scenario"] = d.file.str.extract(r"pipeline_[^_]+_(.*)_s\d+\.csv")[0]; d["resolved"] = d.mcc_resolved >= 0.9
g = d.groupby("scenario")
t = pd.DataFrame({"k_correct": g.k_correct.sum(), "n_resolved": g.resolved.sum(),
                  "G": g.certificate.apply(lambda s: (s == "RELIABLE").sum()), "S": g.certificate.apply(lambda s: (s == "SUSPECT").sum()),
                  "T": g.certificate.apply(lambda s: (s == "NOT_IDENTIFIABLE").sum()), "mcc_res": g.mcc_resolved.mean().round(3),
                  "F1": g.support_F1.mean().round(3), "type": g.type_accuracy_on_edges.mean().round(3),
                  "spurious": g.spurious_correct.apply(lambda s: f"{int(s.dropna().astype(bool).sum())}/{s.notna().sum()}"),
                  "r2A": g[[c for c in d.columns if c.startswith("r2_aff_")][0]].mean().round(3)})
print(t.to_string())
G = d[d.certificate == "RELIABLE"]; Tn = d[d.certificate == "NOT_IDENTIFIABLE"]
lev = d[d.scenario != "E4_no_lever"]                     # "E4_no_lever": run tag of the scenario without any lever
rep = lev[lev.certificate != "NOT_IDENTIFIABLE"]         # a stable lever was reported
print(f"resolved (with lever) {lev.resolved.sum()}/{len(lev)} | lever reported {len(rep)} -> correct {rep.resolved.sum()} | "
      f"RELIABLE {len(G)} -> correct {G.resolved.sum()} | NOT_IDENTIFIABLE {len(Tn)} (of which resolved {Tn.resolved.sum()}) | "
      f"dimension {d.k_correct.sum()}/{len(d)}")
