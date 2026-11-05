"""Causal Panel Transformer (CPT).
Design:
  * ELBO + HISTORY-CONDITIONED factorized prior (breaks rotation invariance)
  * temporal path is primary, regime gate secondary (measured by ablation)
  * mechanism MoE: regime-wise A_e + sparsity penalties (separates the three edge types)
  * weight-tied recursive core: architectural counterpart of the fixed-point iteration of an SCM
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PanelAttention(nn.Module):
    """Two-axis attention: (1) causal over time (t' <= t), (2) across units (cross-section)."""

    def __init__(self, d_model, n_heads, cross_unit=True, dropout=0.1):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.cross_unit = cross_unit
        if cross_unit:
            self.unit_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                   batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, h, unit_mask=None):
        """h: (N, T, D): N units, T time steps."""
        N, T, D = h.shape
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), 1)
        x = self.ln1(h)
        a, _ = self.time_attn(x, x, x, attn_mask=None if getattr(self, "bidir", False) else causal, need_weights=False)
        h = h + a
        if self.cross_unit and N > 1:
            # cross-sectional axis: across units for each t (T moved to the batch dimension)
            x = self.ln2(h).transpose(0, 1)               # (T, N, D)
            a, _ = self.unit_attn(x, x, x, attn_mask=unit_mask, need_weights=False)
            h = h + a.transpose(0, 1)
        h = h + self.ff(self.ln3(h))
        return h


class MechanismMoE(nn.Module):
    """Regime-wise transition matrix A_e (weighted latent graph).
    Fixed edge = identical in all e; regime-dependent = varies with e; the sparsity penalties
    SEPARATE the two (sparse mechanism shift)."""

    def __init__(self, k, n_regimes, family="gauss", nu=5.0):
        super().__init__()
        # family: "gauss" (closed-form KL) | "student" (Student-t conditional prior with fixed nu;
        # a Gaussian prior is blind to the non-Gaussianity (Comon) lever)
        self.family, self.nu = family, nu
        self.A_base = nn.Parameter(torch.randn(k, k) * 0.05)      # fixed backbone
        self.A_shift = nn.Parameter(torch.zeros(n_regimes, k, k))  # regime deviation
        # HARD SPARSITY CONSTRAINT (pattern lever B3). mask=None -> unconstrained (default).
        # mask: (n_regimes, k, k) binary tensor; A_e = (A_base + A_shift_e) * mask_e.
        # If the model class fixes both the diagonal innovation covariance and the zero pattern of A_e, the remaining
        # rotational freedom is skew-symmetric matrices ∩ the pattern-preserving algebra; if this is zero, the rotation
        # is resolved WITHOUT regime variance ratios AND without non-Gaussianity.
        self.register_buffer("mask", None)
        self.b = nn.Parameter(torch.zeros(n_regimes, k))
        self.log_var = nn.Parameter(torch.zeros(n_regimes, k))
        # CPT-2-NL options (added at the END of CPT.__init__ -> the default RNG order is preserved):
        # lags=2 -> + A2_e z_{t-2};  nl -> + h([z_{t-1}, z_{t-2}, onehot(e)]) (last layer zero: starts linear)
        self.lags, self.nl, self.n_regimes = 1, None, n_regimes
        self.flow = None                                    # CPT-2 flow prior: only if mech_flow > 0

    def A(self, e):
        A = self.A_base.unsqueeze(0) + self.A_shift[e]
        if self.mask is not None:
            A = A * self.mask[e]
        return A

    def prior(self, z_prev, e, z_prev2=None):
        """p(z_t | z_{t-1}, e) = N(A_e z_{t-1} + b_e, diag(exp(logvar_e)))
        (+ lags=2: A2_e z_{t-2};  + nl: h(z_{t-1}, z_{t-2}, e); only if selected)"""
        A = self.A(e)                                     # (B, k, k)
        mu = torch.einsum("bij,bj->bi", A, z_prev) + self.b[e]
        if self.lags > 1 or self.nl is not None:
            zp2 = z_prev if z_prev2 is None else z_prev2
            if self.lags > 1:
                mu = mu + torch.einsum("bij,bj->bi", self.A2[e], zp2)
            if self.nl is not None:
                oh = torch.nn.functional.one_hot(e, self.n_regimes).float()
                inp = torch.cat([z_prev] + ([zp2] if self.lags > 1 else []) + [oh], -1)
                mu = mu + self.nl(inp)
        return mu, self.log_var[e].clamp(-8, 4)

    def flow_logp(self, z1, mu, plv, e, z_prev, z_prev2):
        """CPT-2 flow prior: log p(z_t | past, e) = sum_i [log N(eps_i; 0, 1) + log d eps_i / d z_i] (returned per component).
        At initialization (last layer zero, w ~ e^-5) eps = (z - mu) / sigma_e, i.e. equal to the Gaussian prior."""
        B, k = z1.shape; K = self.flow_K
        oh = torch.nn.functional.one_hot(e, self.n_regimes).float()
        o = self.flow(torch.cat([z_prev, z_prev2, oh], -1)).view(B, k, 1 + 2 * K)
        log_a = -0.5 * plv + o[..., 0]                         # scale: regime log-variance + history-dependent deviation
        w = torch.exp(o[..., 1:1 + K] - 5.0)                   # mixture weights > 0 (about 0.007 at initialization)
        off = o[..., 1 + K:]
        v = torch.exp(self.flow_logv).unsqueeze(0)            # (1, k, K) > 0
        u = (z1 - mu).unsqueeze(-1)                            # (B, k, 1)
        th = torch.tanh(v * u + off)
        a = log_a.exp()
        eps = a * u.squeeze(-1) + (w * th).sum(-1)
        deriv = a + (w * v * (1 - th ** 2)).sum(-1)
        return -0.5 * eps ** 2 - 0.5 * math.log(2 * math.pi) + torch.log(deriv)

    def penalties(self):
        sparse_shift = self.A_shift.abs().mean()          # regime-dependent edges sparse
        sparse_base = self.A_base.abs().mean()            # backbone sparse
        return sparse_base, sparse_shift


class CPT(nn.Module):
    def __init__(self, d_in, k=4, d_model=64, n_heads=4, depth=4, n_regimes=1,
                 recursive=True, regime_gated_depth=False, cross_unit=True,
                 dropout=0.1, extra_depth=2, prior_family="gauss", prior_nu=5.0, mech_mask=None,
                 structured_post=False, enc_type="attn", use_pos=True, enc_regime=True, mech_lags=1, mech_nl=0, enc_bidir=False, mech_flow=0,
                 dec="mlp"):
        """recursive=True  -> a single block applied `depth` times (weight-tied)
           recursive=False -> `depth` separate blocks (vanilla)
           regime_gated_depth -> depth+extra_depth iterations in the crisis regime"""
        super().__init__()
        self.k, self.depth, self.recursive = k, depth, recursive
        # STRUCTURED POSTERIOR: q(z_t|.) = N_obs * N_prior (Gaussian product, Kalman update). A free posterior can move
        # z_hat to an arbitrary nonlinear h(z) (low affine R^2 that longer training does not fix); the structural tie
        # forces z_hat to follow LINEAR dynamics.
        self.structured_post = structured_post
        # Diagnostic options (defaults = standard model): enc_type 'attn' | 'mlp' (per time step, does not look at
        # neighbouring steps) | 'gru'; use_pos=False disables the positional embedding; enc_regime=False does not feed
        # the regime to the encoder.
        self.enc_type, self.use_pos, self.enc_regime = enc_type, use_pos, enc_regime
        self.regime_gated_depth, self.extra_depth = regime_gated_depth, extra_depth
        self.inp = nn.Linear(d_in, d_model)
        self.regime_emb = nn.Embedding(max(n_regimes, 1), d_model)
        self.pos = nn.Parameter(torch.randn(1, 4096, d_model) * 0.02)
        n_blocks = 1 if recursive else depth
        self.blocks = nn.ModuleList([PanelAttention(d_model, n_heads, cross_unit, dropout)
                                     for _ in range(n_blocks)])
        self.to_post = nn.Linear(d_model, 2 * k)          # q(z|.) mu, logvar
        if enc_type == "mlp":                              # created only if selected (the default RNG order is preserved)
            self.mlp_enc = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU())
        self.dec = nn.Sequential(nn.Linear(k, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_in))
        self.obs_logvar = nn.Parameter(torch.zeros(d_in))
        self.mech = MechanismMoE(k, max(n_regimes, 1), family=prior_family, nu=prior_nu)
        if mech_mask is not None:
            self.mech.mask = torch.as_tensor(mech_mask, dtype=torch.float32)
        # OUTCOME HEAD: y = f(z, d). z = confounder representation (backdoor set),
        # d = OBSERVED treatment variable. do(d=v): z is held FIXED and d is changed.
        # This is the neural-network counterpart of backdoor adjustment, NOT an input perturbation.
        self.head_y = nn.Sequential(nn.Linear(k + 1, d_model), nn.GELU(),
                                    nn.Linear(d_model, d_model // 2), nn.GELU(),
                                    nn.Linear(d_model // 2, 1))
        # enc_bidir: BIDIRECTIONAL time attention (smoother; sees future x). Valid for the do-adjustment only if there is
        # no feedback (D_t -> z_{t+1}); the default is the causal mask (filter).
        for blk in self.blocks:
            blk.bidir = enc_bidir
        # enc_type='gru': 2-layer UNIDIRECTIONAL GRU (a filter equivalent to causally masked attention);
        # created only if selected and LAST (the default model and RNG order are unchanged)
        if enc_type == "gru":
            self.gru_enc = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)
        # CPT-2-NL: created only if requested and LAST (the default model is unchanged)
        nR = max(n_regimes, 1)
        if mech_lags > 1:
            self.mech.lags = mech_lags
            self.mech.A2 = nn.Parameter(torch.zeros(nR, k, k))
        if mech_nl > 0:
            self.mech.nl = nn.Sequential(nn.Linear(k * min(mech_lags, 2) + nR, mech_nl), nn.GELU(),
                                         nn.Linear(mech_nl, mech_nl), nn.GELU(), nn.Linear(mech_nl, k))
            nn.init.zeros_(self.mech.nl[-1].weight); nn.init.zeros_(self.mech.nl[-1].bias)
        # CPT-2 flow prior: per-component CONDITIONAL MONOTONE transformation prior (instead of an NCTRL/TDRL-type
        # nonparametric transition; invertibility is GUARANTEED and the derivative is in closed form):
        # eps_i = a_i(c) u_i + sum_j w_ij(c) tanh(v_ij u_i + o_ij(c)), u_i = z_{t,i} - mu_i(c), a, w, v > 0
        # =>  d eps_i / d z_{t,i} > 0. c = [z_{t-1}, z_{t-2}, onehot(e)].
        # Given the past the components are INDEPENDENT (conditional diagonality is preserved); the scale and shape of the
        # shock may depend on the past (post-nonlinear structure with the noise inside the nonlinearity).
        # Created only if requested and LAST.
        if mech_flow > 0:
            K = 4
            self.mech.flow_K = K
            self.mech.flow = nn.Sequential(nn.Linear(k * 2 + nR, mech_flow), nn.GELU(), nn.Linear(mech_flow, mech_flow), nn.GELU(),
                                           nn.Linear(mech_flow, k * (1 + 2 * K)))
            nn.init.zeros_(self.mech.flow[-1].weight); nn.init.zeros_(self.mech.flow[-1].bias)
            self.mech.flow_logv = nn.Parameter(torch.zeros(k, K))            # v_ij = exp(.) > 0
        # LINEAR MEASUREMENT FAMILY: dec="linear" -> g(z) = W z + c (a single affine layer k -> D).
        # The observation-noise parametrization (obs_logvar, regime independent, D-dimensional) is the SAME as for the MLP.
        # The MLP decoder is still created above and replaced here (LAST): the initialization of all other parameters and
        # the RNG order are identical to the MLP model (with the same seed the two candidates differ only in the decoder).
        # Default dec="mlp" -> nothing changes.
        # Stage A holds for every INJECTIVE smooth g; x = Bz + c + nu with full-column-rank B lies in the model class.
        self.dec_type = dec
        if dec == "linear":
            self.dec = nn.Linear(k, d_in)
        elif dec != "mlp":
            raise ValueError(f"dec={dec!r}: expected 'mlp' or 'linear'")

    def encode(self, x, e, unit_mask=None):
        """x: (N,T,d_in), e: (N,T) regime index"""
        N, T, _ = x.shape
        h = self.inp(x)
        if self.use_pos:
            h = h + self.pos[:, :T]
        if self.enc_regime:
            h = h + self.regime_emb(e)
        if self.enc_type == "mlp":
            post = self.to_post(self.mlp_enc(h))
            return post[..., :self.k], post[..., self.k:].clamp(-8, 4)
        if self.enc_type == "gru":
            post = self.to_post(self.gru_enc(h)[0])
            return post[..., :self.k], post[..., self.k:].clamp(-8, 4)
        steps = self.depth
        if self.regime_gated_depth and e.numel():
            # iterate deeper if the share of the crisis regime (highest index) is high
            crisis_share = (e == (self.regime_emb.num_embeddings - 1)).float().mean()
            steps = self.depth + int(self.extra_depth * (crisis_share > 0.25))
        if self.recursive:
            for _ in range(steps):
                h = self.blocks[0](h, unit_mask)
        else:
            for blk in self.blocks:
                h = blk(h, unit_mask)
        post = self.to_post(h)
        mu, logvar = post[..., :self.k], post[..., self.k:].clamp(-8, 4)
        return mu, logvar

    def _fuse_with_prior(self, mu_o, logvar_o, e):
        """Gaussian product: observation term (encoder) x prior transition (mechanism).
        z_hat_t is tied to a LINEAR function of z_hat_{t-1}; the freedom of an arbitrary h(z) is removed.
        Forward pass (filter-like, single step; not a full smoother, a cheap approximation)."""
        N, T, k = mu_o.shape
        pv_o = torch.exp(-logvar_o.clamp(-8, 4))                 # observation precision
        mu_prev = mu_o[:, :-1].detach()                          # z_{t-1} (gradient stopped: no fixed point)
        pm, plv = self.mech.prior(mu_prev.reshape(-1, k), e[:, 1:].reshape(-1))
        pm = pm.view(N, T - 1, k); plv = plv.view(N, T - 1, k)
        pv_p = torch.exp(-plv.clamp(-8, 4))
        pv = pv_o[:, 1:] + pv_p
        mu_f = (mu_o[:, 1:] * pv_o[:, 1:] + pm * pv_p) / pv
        mu = torch.cat([mu_o[:, :1], mu_f], dim=1)
        logvar = torch.cat([logvar_o[:, :1], -torch.log(pv.clamp(min=1e-6))], dim=1)
        return mu, logvar.clamp(-8, 4)

    def forward(self, x, e, d=None, unit_mask=None, sample=True):
        """d: (N,T) observed treatment variable. Zero if None."""
        mu, logvar = self.encode(x, e, unit_mask)
        if self.structured_post:
            mu, logvar = self._fuse_with_prior(mu, logvar, e)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        xh = self.dec(z)
        dd = torch.zeros_like(z[..., :1]) if d is None else d.unsqueeze(-1)
        yh = self.head_y(torch.cat([z, dd], -1)).squeeze(-1)
        return dict(z=z, mu=mu, logvar=logvar, x_hat=xh, y_hat=yh)

    @torch.no_grad()
    def do(self, x, e, d_value, unit_mask=None):
        """E[Y | do(D = d_value)]: do-operator on the OBSERVED treatment.
        z (confounder representation) is held FIXED and only d is changed: backdoor adjustment.
        d_value: scalar or (N,T) tensor."""
        mu, _ = self.encode(x, e, unit_mask)
        dd = torch.full_like(mu[..., :1], float(d_value)) if np.isscalar(d_value) \
            else d_value.unsqueeze(-1)
        return self.head_y(torch.cat([mu, dd], -1)).squeeze(-1)

    @torch.no_grad()
    def do_latent(self, x, e, j, value, d=None, unit_mask=None):
        """E[Y | do(z_j = value)]: intervention on a LATENT mechanism (secondary analysis)."""
        mu, _ = self.encode(x, e, unit_mask)
        z = mu.clone(); z[..., j] = value
        dd = torch.zeros_like(z[..., :1]) if d is None else d.unsqueeze(-1)
        return self.head_y(torch.cat([z, dd], -1)).squeeze(-1)
