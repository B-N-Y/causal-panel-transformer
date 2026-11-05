"""Edge scores of the single-regime B2 solution, recomputed from saved Stage-A latents.

A pipeline run with pooled B2 as the primary variant and single-regime B2 as the extra variant (run PIPE150: --b2_extra single
--save_latents) scores the edges of the primary (pooled) solution only. This script reloads the Stage-A latents saved by
experiments/evaluate_pipeline.py --save_latents, applies levers.resolve_v2 with single-regime B2 and edges.type_edges, and scores
the edges exactly as evaluate_pipeline.py does. It is light (minutes on a CPU) and trains nothing.

Usage: python experiments/rescore_edges_single_regime.py [--runs 'outputs/pipeline_v7b2_*_s*.csv'] [--zdir outputs/z]
                                                         [--out outputs/rescore_edges_single_regime.csv]
Columns selection_in_run / certificate_in_run repeat the single-regime verdict stored in the run (selection_extra,
certificate_extra), as a check that the rescoring reproduces it.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("cpt", "baselines", "experiments"):
    if os.path.join(HERE, _d) not in sys.path:
        sys.path.insert(0, os.path.join(HERE, _d))
import levers as KL  # noqa: E402
import edges as KN  # noqa: E402
from simulation import gen_panel, mcc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=os.path.join(HERE, "outputs", "pipeline_v7b2_*_s*.csv"), help="glob of the pipeline CSV files")
    ap.add_argument("--zdir", default=os.path.join(HERE, "outputs", "z"), help="folder with pipeline_stage_a{tag}_s{seed}.npz")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "rescore_edges_single_regime.csv"))
    a = ap.parse_args()
    files = sorted(glob.glob(a.runs))
    runs = pd.concat([pd.read_csv(f).assign(file=os.path.basename(f)) for f in files], ignore_index=True)
    rows = []
    for _, c in runs.iterrows():
        tag = re.match(r"pipeline(_.*)_s\d+\.csv", c.file).group(1)
        zz = np.load(os.path.join(a.zdir, f"pipeline_stage_a{tag}_s{c.seed}.npz"))
        zA, zB, e, w = zz["zA"], zz["zB"], zz["e"], zz["w"]
        E = int(e.max()) + 1
        z, info = KL.resolve_v2(zA, e, E, zh_other=zB, b2_filter="single")
        row = dict(seed=c.seed, E=c.E, varmode=c.varmode, innov=c.innov, spurious=c.spurious, graph=c.graph, k_selected=c.k_selected,
                   k_true=c.k_true, selection=info["selection"], certificate=info["certificate"],
                   selection_in_run=c.selection_extra, certificate_in_run=c.certificate_extra)
        if z is not None:
            m_, perm = mcc(w[200:], z[200:]); row["mcc_resolved"] = m_
            if info["certificate"] != "NOT_IDENTIFIABLE" and z.shape[1] == w.shape[1] and len(perm) == w.shape[1]:
                _, _, _, meta = gen_panel(seed=int(c.seed), E=int(c.E), T=6000, varmode=c.varmode, innov=c.innov,
                                          spurious=bool(c.spurious), graph=c.graph)
                K = KN.type_edges(z, e, E, B=200, seed=0); kk = w.shape[1]
                # true A^aug and edge types; the confounder is AR(1) with coefficient 0.8 when the spurious block is on
                At = [np.block([[Ae, np.zeros((4, 1))], [np.zeros((1, 4)), np.array([[0.8]])]]) if kk == 5 else Ae for Ae in meta["A_list"]]
                At = np.array(At); nz = np.abs(At) > 1e-9
                true_type = np.where(~nz.any(0), "none", np.where(nz.all(0) & np.all(np.isclose(At, At[0]), 0), "FIXED", "REGIME_DEPENDENT"))
                inv = {int(q): g for g, q in enumerate(perm)}          # estimated column perm[g] <-> true component g
                type_h = np.empty((kk, kk), dtype=object); present_h = np.zeros_like(nz)
                for i in range(kk):
                    for j in range(kk):
                        type_h[inv[i], inv[j]] = K["type"][i, j]
                        for r in range(nz.shape[0]):
                            present_h[r, inv[i], inv[j]] = K["present"][r, i, j]
                tp = (present_h & nz).sum(); fp = (present_h & ~nz).sum(); fn = (~present_h & nz).sum()
                row["support_F1"] = 2 * tp / max(2 * tp + fp + fn, 1)
                mask = true_type != "none"
                row["type_accuracy_on_edges"] = float((type_h[mask] == true_type[mask]).mean())
                row["none_accuracy"] = float((type_h[~mask] == "none").mean())
                if kk == 5:
                    row["spurious_correct"] = (4 in [inv[i] for i in K["spurious_latent"]]) and len(K["spurious_latent"]) == 1
        rows.append(row)
    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    d.to_csv(a.out, index=False)
    print("same selection as stored in the run:", (d.selection == d.selection_in_run).mean(),
          " same certificate:", (d.certificate == d.certificate_in_run).mean())


if __name__ == "__main__":
    main()
