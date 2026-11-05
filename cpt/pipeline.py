"""End-to-end two-stage causal solver (never accesses ground-truth values).
Steps: (1) latent dimension selection -> k*; (2) encoder fit with k* (a second fit is kept for a diagnostic only);
(3) Stage B: the lever with the highest split-half stability is used if its stability reaches t_stab (B1 JD / B2 ICA);
otherwise B3 if a zero pattern is given; otherwise the output is NOT_IDENTIFIABLE (abstention). The B3 uniqueness check of
the paper (Algorithm 1) is b3_pattern.b3_scan + b3_pattern.b3_uniqueness, applied by experiments/evaluate_pipeline.py;
(4) edge typing + normalized weights with classical OLS standard errors and Holm correction (edges.type_edges).
out["stage_b"]["certificate"]: "NOT_IDENTIFIABLE" means abstain; "RELIABLE" and "SUSPECT" both mean a reported solution and
differ only in the cross-encoder agreement, a diagnostic that is not part of the procedure in the paper.
Input: x (T x D), e (T,). Output: a dictionary (writing to disk is left to the caller).
"""
import time
import numpy as np
import encoders as KD, dimension as BY, levers as KL, edges as KN


def run_pipeline(x, e, ks=range(2, 8), dim_seeds=(0, 1), seed=0, epochs=200, masks=None, B=200, k_fixed=None,
             encoders=("iVAE", "TDRL"), cpt2_steps=20000, dim_c=0.07, dim_rule="v2", b2_filter=None,
             decoder="mlp", b2_extra=None, rc_m=None, cpt2_w_rc=None):
    """encoders: (primary, secondary) encoders. The latent of the primary encoder is resolved; the secondary encoder is used
    only for the cross-encoder agreement (diagnostic, info["cross_rho"]). If both are the same method, the second fit uses a
    different seed (the diagnostic is then the agreement across seeds).
    B: unused (accepted for backward compatibility; edge uncertainty uses classical standard errors).
    k_fixed: if given, dimension selection is skipped and this latent dimension is used.
    dim_rule: dimension-selection rule. "v2" (default: over-select, then prune; see dimension.select_dimension_v2) or
    any other value, conventionally "elbow" (elbow rule on the iVAE held-out loss with threshold dim_c;
    dimension.select_dimension). "v2" is defined only for a single long series.
    dim_seeds: seeds of the encoder fits used by the elbow rule.
    b2_filter: None (default: B2 on the residuals pooled across regimes) or an option passed to levers.resolve_v2 (argument
    b2_filter), e.g. "single" (B2 on the regime with the most observations).
    b2_extra: None (default) or a second B2 variant ("pooled", "single" or another b2_filter option) applied to the SAME
    Stage-A latent for reporting only; its result is returned in out["stage_b_extra"] and out["z_extra"].
    rc_m: None (default: no effect) or the number of regime-contrast directions per non-reference regime; the observations are
    then replaced by their regime-contrast projection (contrast.regime_contrast) before dimension selection and Stage A.
    cpt2_w_rc: None (default: no effect) or the weight of the regime-contrast training term of the CPT2 encoders (Stage A).
    decoder: "mlp" (default) | "linear" (CPT2 encoders with a linear decoder) | "select" (CPT2 encoders are fitted as CPT2s,
    which chooses between an MLP and a linear decoder with the default transition family, by the held-out ELBO without
    free bits). Affects only CPT2 encoders; dimension selection and iVAE/TDRL are unchanged."""
    t0 = time.time(); E = int(e.max()) + 1; out = {}
    if rc_m is not None:
        from contrast import regime_contrast
        x, rc = regime_contrast(x, e, rc_m); out["rc"] = dict(m=rc_m, ref=rc["ref"], d=x.shape[1])
    if k_fixed is None:
        if dim_rule == "v2":
            k, dim_info = BY.select_dimension_v2(x, e, kmax=max(8, max(ks)), fit_kw=dict(steps=cpt2_steps))   # ks<=7 -> kmax 8
        else:
            k, dim_info = BY.select_dimension(x, e, ks=ks, seeds=dim_seeds, epochs=epochs, c=dim_c)
        out["dimension"] = dict(k_selected=k, **{kk: v for kk, v in dim_info.items()})
    else:
        k = k_fixed; out["dimension"] = dict(k_selected=k, note="given by caller")
    kA, kB = encoders
    _kw = lambda m: dict(steps=cpt2_steps, **({"w_rc": cpt2_w_rc} if cpt2_w_rc else {})) if m in ("CPT2", "CPT2s") else dict(epochs=epochs)
    _method = lambda m: m                                 # encoder name -> KD.fit method (default: identical)
    if decoder != "mlp":
        _kw0 = _kw
        if decoder == "linear":
            _kw = lambda m: dict(_kw0(m), **({"dec": "linear"} if m == "CPT2" else {}))
        elif decoder == "select":
            _method = lambda m: "CPT2s" if m == "CPT2" else m
            _kw = lambda m: dict(_kw0(m), **({"candidates": ((1, 0, 0, "mlp"), (1, 0, 0, "linear"))} if m == "CPT2" else {}))
        else:
            raise ValueError(f"decoder={decoder!r}: expected 'mlp', 'linear' or 'select'")
    zA, iA = KD.fit(_method(kA), x, e, k, seed=seed, **_kw(kA))
    seedB = seed + 1 if kA == kB else seed               # same encoder twice: the second fit uses a different seed
    zB, iB = KD.fit(_method(kB), x, e, k, seed=seedB, **_kw(kB))
    out["encoders"] = {kA: iA, (kB if kB != kA else kB + "_2"): iB}
    if masks is not None and np.shape(masks)[-1] != k:   # the pattern does not fit the selected k: B3 cannot be used
        out["pattern_mismatch"] = True; masks = None
    z, info = KL.resolve_v2(zA, e, E, zh_other=zB, masks=masks, **({"b2_filter": b2_filter} if b2_filter is not None else {}))
    out["stage_b"] = info; out["zA"], out["zB"], out["z"] = zA, zB, z
    if b2_extra is not None:                             # second B2 variant on the same Stage A (reporting only)
        out["z_extra"], out["stage_b_extra"] = KL.resolve_v2(zA, e, E, zh_other=zB, masks=masks,
                                                             **({"b2_filter": b2_extra} if b2_extra != "pooled" else {}))
    if z is not None and info["certificate"] != "NOT_IDENTIFIABLE":
        out["edges"] = KN.type_edges(z, e, E, B=B, seed=seed)
    out["time"] = time.time() - t0
    return out
