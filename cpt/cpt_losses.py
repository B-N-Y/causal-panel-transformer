"""CPT loss function: ELBO + mechanism penalties."""
import math
import torch


def cpt_loss(out, x, e, mech, w_sparse_base=0.01, w_sparse_shift=0.05,
             w_pred=0.0, y=None, obs_logvar=None, free_bits=0.02, beta=1.0, w_jd=0.0, w_rc=0.0):
    """ELBO = rec_NLL + KL(q || p(z_t|z_{t-1},e)) + sparsity penalties.
    The prior is CONDITIONED ON THE PAST: in the panel SCM the latent components factorize only conditionally on the past."""
    z, mu, logvar, xh = out["z"], out["mu"], out["logvar"], out["x_hat"]
    N, T, k = z.shape
    olv = obs_logvar.clamp(-8, 4)
    rec = 0.5 * (((x - xh) ** 2) / olv.exp() + olv).sum(-1).mean()

    # history-conditioned prior: z_{t-1} is detached (the prior treats the past as data)
    zp = z[:, :-1].detach().reshape(-1, k)
    e1 = e[:, 1:].reshape(-1)
    if getattr(mech, "lags", 1) > 1 or getattr(mech, "nl", None) is not None or getattr(mech, "flow", None) is not None:   # CPT-2-NL / flow prior: z_{t-2}
        zp2 = torch.cat([z[:, :1], z[:, :-2]], 1).detach().reshape(-1, k)
        pm, plv = mech.prior(zp, e1, zp2)
    else:
        pm, plv = mech.prior(zp, e1)
    mu1 = mu[:, 1:].reshape(-1, k)
    lv1 = logvar[:, 1:].reshape(-1, k)
    if getattr(mech, "flow", None) is not None:
        # CPT-2 flow prior: conditional monotone-transformation prior; single-sample MC KL (as in the Student branch):
        # KL = E_q[log q] - E_q[log p],  E_q[log q] = -H(q) in closed form
        z1 = z[:, 1:].reshape(-1, k)
        zp2f = torch.cat([z[:, :1], z[:, :-2]], 1).detach().reshape(-1, k)
        logp = mech.flow_logp(z1, pm, plv, e1, zp, zp2f)
        neg_H = -0.5 * (lv1 + math.log(2 * math.pi * math.e))
        kl = neg_H - logp
    elif getattr(mech, "family", "gauss") == "student":
        # Student-t conditional prior: KL has no closed form -> single-sample MC:
        # KL = E_q[log q] - E_q[log p];  E_q[log q] = -H(q) (Gaussian entropy, closed form)
        nu = mech.nu
        z1 = z[:, 1:].reshape(-1, k)                      # sample from q (reparametrized)
        sig = (0.5 * plv).exp()
        u = (z1 - pm) / sig
        logp = (torch.lgamma(torch.tensor((nu + 1) / 2, device=z.device))
                - torch.lgamma(torch.tensor(nu / 2, device=z.device))
                - 0.5 * math.log(nu * math.pi) - 0.5 * plv
                - 0.5 * (nu + 1) * torch.log1p(u ** 2 / nu))
        neg_H = -0.5 * (lv1 + math.log(2 * math.pi * math.e))
        kl = neg_H - logp
    else:
        kl = 0.5 * (plv - lv1 + (lv1.exp() + (mu1 - pm) ** 2) / plv.exp() - 1)
    if getattr(mech, "flow", None) is not None:
        kl = kl.mean(0).clamp(min=free_bits).sum()        # flow prior: the MC estimate is noisy -> free bits on the BATCH mean
    else:
        kl = kl.clamp(min=free_bits).sum(-1).mean()      # free bits: prevents posterior collapse

    sb, ss = mech.penalties()
    loss = rec + beta * kl + w_sparse_base * sb + w_sparse_shift * ss   # beta: KL weight
    # --- JD PENALTY (differentiable version of the joint-diagonalization lever B1) ---
    # The log-likelihood of a diagonal Gaussian prior sees only the MARGINAL variance of the residuals; it is blind to
    # correlations between components. The information that resolves the rotation (joint diagonalizability of the
    # regime-wise innovation covariances) lies exactly in these correlations. Penalty: in each regime, off-diagonal
    # energy / diagonal energy of the covariance of the posterior-mean residuals (scale invariant).
    if w_jd > 0:
        r = mu1 - pm                                        # (B*T', k) residual (posterior mean)
        jd = torch.zeros((), device=r.device); n_e = 0
        for reg in torch.unique(e1):
            rr = r[e1 == reg]
            if rr.shape[0] < 4 * k: continue
            rr = rr - rr.mean(0, keepdim=True)
            C = rr.T @ rr / (rr.shape[0] - 1)
            off = C - torch.diag(torch.diagonal(C))
            jd = jd + (off ** 2).sum() / ((torch.diagonal(C) ** 2).sum() + 1e-8)
            n_e += 1
        if n_e > 0:
            jd = jd / n_e
            loss = loss + w_jd * jd
    # --- REGIME-CONTRAST TERM (optional; lever B1 used in Stage A) ---
    # Cross-entropy of the observed regime e_t given the transition (mu_{t-1} -> mu_t) of the posterior means, scored by the
    # regime-wise priors N(A_e mu_{t-1} + b_e, diag(exp(logvar_e))). The regime is not an input of the encoder, so the term can
    # only be lowered by latent components whose innovations differ across regimes. Gaussian one-lag prior only.
    rc_val = 0.0
    if w_rc > 0:
        k_ = mu.shape[-1]; m0 = mu[:, :-1].reshape(-1, k_); m1 = mu[:, 1:].reshape(-1, k_)
        E_ = mech.b.shape[0]; A_all = mech.A(torch.arange(E_, device=mu.device))            # (E, k, k)
        pred = torch.einsum("eij,nj->nei", A_all, m0) + mech.b.unsqueeze(0)               # (n, E, k)
        lv = mech.log_var.clamp(-8, 4).unsqueeze(0)
        ll = -0.5 * (((m1.unsqueeze(1) - pred) ** 2) / lv.exp() + lv).sum(-1)               # (n, E)
        rc = torch.nn.functional.cross_entropy(ll, e1)
        loss = loss + w_rc * rc; rc_val = float(rc.item())
    jd_val = float(jd.item()) if w_jd > 0 and n_e > 0 else 0.0
    parts = dict(rec=rec.item(), kl=kl.item(), sparse_base=sb.item(),
                 sparse_shift=ss.item(), jd=jd_val)
    if w_rc > 0:
        parts["rc"] = rc_val
    if w_pred > 0 and y is not None:
        pred = ((out["y_hat"] - y) ** 2).mean()
        loss = loss + w_pred * pred
        parts["pred"] = pred.item()
    parts["loss"] = loss.item()
    return loss, parts
