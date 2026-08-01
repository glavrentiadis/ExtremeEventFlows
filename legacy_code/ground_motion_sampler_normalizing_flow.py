#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

r"""
Created on Mon Mar 30 06:47:48 2026

@author: glavrent

Normalizing-flow rare-event sampler for ground-motion-earthquake event sequences.

This script implements the flow-based approach in described in Gibson et al. (2021) 
for rare-event simulation for seismic event chains over a finite horizon. 
The target distribution is

    q*(x) \propto p(x) rho(x),

where p(x) is the original event-sequence density and rho(x) is a smooth rare-
event penalty/surrogate. Training minimizes

    E_z[ log p_z(z) - log|det dT/dz| - log h(T(z)) ]

with h(x) = p(x) rho(x),

so that the learned flow directly generates samples concentrated in the rare-
event region. Importance weights p(x)/q_theta(x) can then be used to estimate
rare-event probabilities or conditional expectations.

Design choices:
- Fixed-length sequence representation with padding/masking.
- Each event is represented by (dt, mag, eps), where:
    dt  : inter-arrival time
    mag : magnitude
    eps : standardized intra-event residual used to generate lnSA.
- A RealNVP flow acts on the flattened vector.
- Physical constraints are enforced by deterministic transforms:
    dt  = softplus(raw_dt)
    mag = m_min + (m_max-m_min) * sigmoid(raw_mag)
- Ground-motion amplitudes are generated through a user-supplied GM mean/std
  function. A simple default surrogate is included.
"""

# libaries
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
#math
import math
import numpy as np
#import gmms
import pygmm
#pytorch modules
import torch
from torch import nn
from torch.distributions import Normal

Tensor = torch.Tensor

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def log1mexp(x: Tensor) -> Tensor:
    """Stable log(1-exp(-x)) for x > 0."""
    log2 = math.log(2.0)
    return torch.where(x < log2, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))


def standard_normal_survival(z: torch.Tensor) -> torch.Tensor:
    """
    Standard normal survival function: 1 - Phi(z)
    """
    return 0.5 * torch.erfc(z / math.sqrt(2.0))


def poisson_tail_prob(mu: float, k: int) -> float:
    p = math.exp(-mu)
    cdf = p
    for j in range(1, k + 1):
        p *= mu / j
        cdf += p
    return max(0.0, 1.0 - cdf)


# Density Functions
# ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---
def truncated_exponential_logpdf(x: Tensor, rate: Tensor, low: float, high: float) -> Tensor:
    """Log-pdf of truncated exponential on [low, high], shifted by low."""
    y = x - low
    width = high - low
    invalid = (x < low) | (x > high)
    norm = -log1mexp(rate * width)
    out = torch.log(rate) - rate * y + norm
    out = torch.where(invalid, torch.full_like(out, -torch.inf), out)
    return out


def exponential_logpdf(x: Tensor, rate: Tensor) -> Tensor:
    invalid = x <= 0
    out = torch.log(rate) - rate * x
    out = torch.where(invalid, torch.full_like(out, -torch.inf), out)
    return out


def normal_logpdf(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    var = std.square()
    return -0.5 * (math.log(2.0 * math.pi) + torch.log(var) + (x - mean).square() / var)

# -----------------------------------------------------------------------------
# Sequence representation
# -----------------------------------------------------------------------------

@dataclass
class FlowConfig:
    seq_len: int = 12
    hidden_dim: int = 256
    n_coupling_layers: int = 8
    mag_min: float = 4.0
    mag_max: float = 8.5
    lambda_rate: float = 4.0
    b_value: float = 1.0
    t_max: float = 50.0
    gm_threshold: float = 0.75
    penalty_alpha: float = 40.0
    site_vs30: float = 500.0
    distance_km: float = 20.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    @property
    def beta(self) -> float:
        return math.log(10.0) * self.b_value


@dataclass
class SequenceBatch:
    raw: Tensor              # [B, D], unconstrained flow output
    dt: Tensor               # [B, K]
    mag: Tensor              # [B, K]
    eps: Tensor              # [B, K]
    event_time: Tensor       # [B, K]
    mask: Tensor             # [B, K] ; 1 if event time <= t_max
    ln_sa: Tensor            # [B, K]
    sa: Tensor               # [B, K]
    
# -----------------------------------------------------------------------------
# Surrogate GM
# -----------------------------------------------------------------------------

class DefaultGroundMotionModel(nn.Module):
    """
    Simple differentiable surrogate for ln(SA) mean/std.

    Replace this with a pygmm-backed surrogate if needed. The interface is:
        mean, std = model(mag, distance_km, vs30)
    where mag has shape [...].
    """

    def forward(self, mag: Tensor, distance_km: Tensor, vs30: Tensor) -> Tuple[Tensor, Tensor]:
        # heuristic trend: stronger shaking for larger M, weaker for larger R,
        # mild site amplification for lower Vs30.
        mean = -4.25 + 1.15 * mag - 1.10 * torch.log(distance_km + 5.0) - 0.20 * torch.log(vs30 / 500.0)
        std = torch.full_like(mean, 0.55)
        return mean, std
    
    
# -----------------------------------------------------------------------------
# RealNVP blocks
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden: int, mask: Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale_net = MLP(dim, hidden, dim)
        self.shift_net = MLP(dim, hidden, dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_masked = x * self.mask
        log_s = self.scale_net(x_masked) * (1.0 - self.mask)
        t = self.shift_net(x_masked) * (1.0 - self.mask)
        log_s = 0.8 * torch.tanh(log_s)
        y = x_masked + (1.0 - self.mask) * (x * torch.exp(log_s) + t)
        log_det = log_s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        y_masked = y * self.mask
        log_s = self.scale_net(y_masked) * (1.0 - self.mask)
        t = self.shift_net(y_masked) * (1.0 - self.mask)
        log_s = 0.8 * torch.tanh(log_s)
        x = y_masked + (1.0 - self.mask) * (y - t) * torch.exp(-log_s)
        log_det = -log_s.sum(dim=-1)
        return x, log_det
    
class RealNVP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, n_layers: int, event_dim: int = 3):
        super().__init__()
        if dim % event_dim != 0:
            raise ValueError("dim must be divisible by event_dim")

        n_events = dim // event_dim
        layers = []

        for i in range(n_layers):
            event_mask = torch.zeros(n_events, dtype=torch.float64)
            event_mask[i % 2 :: 2] = 1.0
            mask = event_mask.repeat_interleave(event_dim)
            layers.append(AffineCoupling(dim, hidden_dim, mask))

        self.layers = nn.ModuleList(layers)
        self.base = Normal(0.0, 1.0)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        x = z
        log_det = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for layer in self.layers:
            x, ld = layer(x)
            log_det = log_det + ld
        return x, log_det

    def inverse(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = x
        log_det = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            log_det = log_det + ld
        return z, log_det

    def log_q_raw(self, x: Tensor) -> Tensor:
        z, log_det_inv = self.inverse(x)
        log_pz = self.base.log_prob(z).sum(dim=-1)
        return log_pz + log_det_inv

    def sample_raw(self, n: int, *, device: str, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        z = torch.randn(n, self.layers[0].mask.numel(), device=device, dtype=dtype)
        x, log_det = self.forward(z)
        log_pz = self.base.log_prob(z).sum(dim=-1)
        log_q = log_pz - log_det
        return x, log_q
    

# -----------------------------------------------------------------------------
# Rare-event flow model
# -----------------------------------------------------------------------------

class RareGroundMotionFlow(nn.Module):
    def __init__(self, cfg: FlowConfig, gm_model: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.event_dim = 3  # dt_raw, mag_raw, eps
        self.total_dim = self.seq_len * self.event_dim
        self.flow = RealNVP(
            self.total_dim,
            cfg.hidden_dim,
            cfg.n_coupling_layers,
            event_dim=self.event_dim,
        )
        self.gm_model = gm_model if gm_model is not None else DefaultGroundMotionModel()

    def decode(self, raw: Tensor) -> SequenceBatch:
        B = raw.shape[0]
        K = self.seq_len
        x = raw.view(B, K, self.event_dim)

        raw_dt = x[..., 0]
        raw_mag = x[..., 1]
        eps = x[..., 2]

        dt = torch.nn.functional.softplus(raw_dt) + 1.0e-4
        mag = self.cfg.mag_min + (self.cfg.mag_max - self.cfg.mag_min) * torch.sigmoid(raw_mag)

        event_time = torch.cumsum(dt, dim=1)
        mask = (event_time <= self.cfg.t_max).to(raw.dtype)

        distance = torch.full_like(mag, self.cfg.distance_km)
        vs30 = torch.full_like(mag, self.cfg.site_vs30)
        ln_mean, ln_std = self.gm_model(mag, distance, vs30)
        ln_sa = ln_mean + ln_std * eps
        sa = torch.exp(ln_sa)

        return SequenceBatch(
            raw=raw,
            dt=dt,
            mag=mag,
            eps=eps,
            event_time=event_time,
            mask=mask,
            ln_sa=ln_sa,
            sa=sa,
        )

    def sample_sequences(self, n: int) -> Tuple[SequenceBatch, Tensor]:
        raw, log_q_raw = self.flow.sample_raw(n, device=self.cfg.device, dtype=self.cfg.dtype)
        batch = self.decode(raw)
        log_abs_det_constr = self.constraint_log_det(batch)
        log_q_x = log_q_raw - log_abs_det_constr
        return batch, log_q_x

    def constraint_log_det(self, batch: SequenceBatch) -> Tensor:
        """
        Log |d x_phys / d x_raw| for the deterministic coordinate transforms:
        softplus for dt and affine-sigmoid for magnitude. eps is identity.
        """
        raw = batch.raw.view(batch.raw.shape[0], self.seq_len, self.event_dim)
        raw_dt = raw[..., 0]
        raw_mag = raw[..., 1]

        log_d_dt = torch.nn.functional.logsigmoid(raw_dt)
        sig = torch.sigmoid(raw_mag)
        log_d_mag = math.log(self.cfg.mag_max - self.cfg.mag_min) + torch.log(sig) + torch.log1p(-sig)
        return (log_d_dt + log_d_mag).sum(dim=1)

    def encode_physical_to_raw(self, dt: Tensor, mag: Tensor, eps: Tensor) -> Tensor:
        """
        Inverse of the deterministic physical transforms used in decode():
            dt  = softplus(raw_dt) + 1e-4
            mag = mag_min + (mag_max - mag_min) * sigmoid(raw_mag)
            eps = raw_eps
        """
        dt_shift = torch.clamp(dt - 1.0e-4, min=1.0e-12)
        raw_dt = torch.log(torch.expm1(dt_shift))
    
        u = (mag - self.cfg.mag_min) / (self.cfg.mag_max - self.cfg.mag_min)
        u = torch.clamp(u, min=1.0e-12, max=1.0 - 1.0e-12)
        raw_mag = torch.log(u) - torch.log1p(-u)
    
        raw = torch.stack([raw_dt, raw_mag, eps], dim=-1)
        return raw.reshape(dt.shape[0], -1)

    @torch.no_grad()
    def log_flow_density_on_physical_batch(self, batch: SequenceBatch) -> Tensor:
        raw = self.encode_physical_to_raw(batch.dt, batch.mag, batch.eps)
        log_q_raw = self.flow.log_q_raw(raw)
    
        batch_with_raw = SequenceBatch(
            raw=raw,
            dt=batch.dt,
            mag=batch.mag,
            eps=batch.eps,
            event_time=batch.event_time,
            mask=batch.mask,
            ln_sa=batch.ln_sa,
            sa=batch.sa,
        )
        log_abs_det = self.constraint_log_det(batch_with_raw)
    
        return log_q_raw - log_abs_det

    def log_original_density(self, batch: SequenceBatch) -> Tensor:
        """
        Proper density on the full fixed-length augmented state.
        Events after t_max still have density; they simply do not contribute
        to the performance function.
        """
        lam = torch.as_tensor(self.cfg.lambda_rate, dtype=batch.raw.dtype, device=batch.raw.device)
        beta = torch.as_tensor(self.cfg.beta, dtype=batch.raw.dtype, device=batch.raw.device)
    
        logp_dt = exponential_logpdf(batch.dt, lam)
        logp_mag = truncated_exponential_logpdf(batch.mag, beta, self.cfg.mag_min, self.cfg.mag_max)
        logp_eps = normal_logpdf(
            batch.eps,
            torch.zeros_like(batch.eps),
            torch.ones_like(batch.eps),
        )
    
        return logp_dt.sum(dim=1) + logp_mag.sum(dim=1) + logp_eps.sum(dim=1)

    def performance(self, batch: SequenceBatch) -> Tensor:
        # Rare event: at least one event exceeds threshold.
        masked_sa = batch.sa * batch.mask
        max_sa, _ = masked_sa.max(dim=1)
        return max_sa

    def log_penalty(self, batch: SequenceBatch) -> Tensor:
        s = self.performance(batch)
        gamma = self.cfg.gm_threshold
        alpha = self.cfg.penalty_alpha
        return -alpha * torch.relu(gamma - s)

    def log_target_unnormalized(self, batch: SequenceBatch) -> Tensor:
        return self.log_original_density(batch) + self.log_penalty(batch)

    def objective(self, n: int) -> Dict[str, Tensor]:
        batch, log_q = self.sample_sequences(n)
        log_h = self.log_target_unnormalized(batch)
        loss = (log_q - log_h).mean()
        rare_prob_surr = torch.sigmoid(self.cfg.penalty_alpha * (self.performance(batch) - self.cfg.gm_threshold)).mean()
        return {
            "loss": loss,
            "batch": batch,
            "log_q": log_q,
            "log_h": log_h,
            "rare_prob_surr": rare_prob_surr,
        }
  
    @torch.no_grad()
    def sample_original_raw(self, n: int) -> Tensor:
        cfg = self.cfg
        device = cfg.device
        dtype = cfg.dtype
    
        lam = torch.as_tensor(cfg.lambda_rate, dtype=dtype, device=device)
        beta = torch.as_tensor(cfg.beta, dtype=dtype, device=device)
    
        u_dt = torch.rand((n, cfg.seq_len), dtype=dtype, device=device)
        dt = -torch.log1p(-u_dt) / lam
        dt_shift = torch.clamp(dt - 1.0e-4, min=1.0e-12)
        raw_dt = torch.log(torch.expm1(dt_shift))
    
        u_m = torch.rand((n, cfg.seq_len), dtype=dtype, device=device)
        width = cfg.mag_max - cfg.mag_min
        mag = cfg.mag_min - torch.log(
            1.0 - u_m * (1.0 - torch.exp(-beta * width))
        ) / beta
        uu = (mag - cfg.mag_min) / (cfg.mag_max - cfg.mag_min)
        uu = torch.clamp(uu, min=1.0e-12, max=1.0 - 1.0e-12)
        raw_mag = torch.log(uu) - torch.log1p(-uu)
    
        eps = torch.randn((n, cfg.seq_len), dtype=dtype, device=device)
    
        raw = torch.stack([raw_dt, raw_mag, eps], dim=-1).reshape(n, -1)
        return raw
    
    
    @torch.no_grad()
    def sample_original_sequences(self, n: int) -> SequenceBatch:
        raw = self.sample_original_raw(n)
        return self.decode(raw)
    
    @torch.no_grad()
    def estimate_rare_event_probability_mixture(self, n: int, eps_mix: float = 0.2) -> Dict[str, Tensor]:
        n_p = int(round(eps_mix * n))
        n_q = n - n_p
    
        raw_q, _ = self.flow.sample_raw(n_q, device=self.cfg.device, dtype=self.cfg.dtype)
        raw_p = self.sample_original_raw(n_p)
    
        raw_all = torch.cat([raw_q, raw_p], dim=0)
        batch = self.decode(raw_all)
    
        log_p = self.log_original_density(batch)
    
        log_q_flow_raw = self.flow.log_q_raw(raw_all)
        batch_with_raw = SequenceBatch(
            raw=raw_all,
            dt=batch.dt,
            mag=batch.mag,
            eps=batch.eps,
            event_time=batch.event_time,
            mask=batch.mask,
            ln_sa=batch.ln_sa,
            sa=batch.sa,
        )
        log_abs_det = self.constraint_log_det(batch_with_raw)
        log_q_flow = log_q_flow_raw - log_abs_det
    
        log_q_orig = log_p
    
        log_q_mix = torch.logaddexp(
            math.log(1.0 - eps_mix) + log_q_flow,
            math.log(eps_mix) + log_q_orig,
        )
    
        indicator = (self.performance(batch) >= self.cfg.gm_threshold)
        log_w = log_p - log_q_mix
    
        log_mean_w = torch.logsumexp(log_w, dim=0) - math.log(n)
        mean_weight = torch.exp(log_mean_w)
    
        if indicator.any():
            log_mean_p = torch.logsumexp(log_w[indicator], dim=0) - math.log(n)
            probability = torch.exp(log_mean_p)
        else:
            probability = torch.zeros((), dtype=log_w.dtype, device=log_w.device)
    
        w_centered = torch.exp(log_w - log_w.max())
        ess = w_centered.sum().square() / (w_centered.square().sum() + 1e-12)
    
        return {
            "probability": probability,
            "indicator_rate_under_qmix": indicator.to(log_w.dtype).mean(),
            "mean_weight": mean_weight,
            "effective_sample_size_approx": ess,
            "max_sa_mean": self.performance(batch).mean(),
            "mean_event_count": batch.mask.sum(dim=1).mean(),
            "std_event_count": batch.mask.sum(dim=1).std(),
            "log_w_mean": log_w.mean(),
            "log_w_std": log_w.std(),
            "log_w_min": log_w.min(),
            "log_w_max": log_w.max(),
            "n_flow_samples": torch.tensor(float(n_q), dtype=log_w.dtype, device=log_w.device),
            "n_orig_samples": torch.tensor(float(n_p), dtype=log_w.dtype, device=log_w.device),
        }
    
    @torch.no_grad()
    def diagnose_flow_importance_weights(self, n: int) -> Dict[str, Tensor]:
        """
        Correct normalization diagnostic:
            E_q[p(x)/q(x)] = 1
        where q is the learned flow density on physical space.
        """
        batch, log_q = self.sample_sequences(n)
        log_p = self.log_original_density(batch)
    
        log_w = log_p - log_q
        log_mean_w = torch.logsumexp(log_w, dim=0) - math.log(n)
        mean_w = torch.exp(log_mean_w)
    
        w_centered = torch.exp(log_w - log_w.max())
        ess = w_centered.sum().square() / (w_centered.square().sum() + 1e-12)
    
        return {
            "mean_p_over_q_under_q": mean_w,
            "ess_under_q": ess,
            "log_w_mean": log_w.mean(),
            "log_w_std": log_w.std(),
            "log_w_min": log_w.min(),
            "log_w_max": log_w.max(),
            "mean_event_count": batch.mask.sum(dim=1).mean(),
            "std_event_count": batch.mask.sum(dim=1).std(),
        }
    
    @torch.no_grad()
    def estimate_rare_event_probability_flow_only(self, n: int) -> Dict[str, Tensor]:
        batch, log_q = self.sample_sequences(n)
        log_p = self.log_original_density(batch)
        indicator = (self.performance(batch) >= self.cfg.gm_threshold)
    
        log_w = log_p - log_q
    
        # stable mean weight
        log_mean_w = torch.logsumexp(log_w, dim=0) - math.log(n)
        mean_weight = torch.exp(log_mean_w)
    
        # stable probability estimate
        if indicator.any():
            log_mean_p = torch.logsumexp(log_w[indicator], dim=0) - math.log(n)
            probability = torch.exp(log_mean_p)
        else:
            probability = torch.zeros((), dtype=log_w.dtype, device=log_w.device)
    
        w_centered = torch.exp(log_w - log_w.max())
        ess = w_centered.sum().square() / (w_centered.square().sum() + 1e-12)
    
        return {
            "probability": probability,
            "indicator_rate_under_q": indicator.to(log_w.dtype).mean(),
            "mean_weight": mean_weight,
            "effective_sample_size_approx": ess,
            "max_sa_mean": self.performance(batch).mean(),
            "mean_event_count": batch.mask.sum(dim=1).mean(),
            "std_event_count": batch.mask.sum(dim=1).std(),
            "log_w_mean": log_w.mean(),
            "log_w_std": log_w.std(),
            "log_w_min": log_w.min(),
            "log_w_max": log_w.max(),
        }
    
    @torch.no_grad()
    def estimate_rare_event_probability_crude(self, n: int) -> Dict[str, Tensor]:
        batch = self.sample_original_sequences(n)
        indicator = (self.performance(batch) >= self.cfg.gm_threshold).to(batch.sa.dtype)
        return {
            "probability": indicator.mean(),
            "max_sa_mean": self.performance(batch).mean(),
            "mean_event_count": batch.mask.sum(dim=1).mean(),
            "std_event_count": batch.mask.sum(dim=1).std(),
        }
    
    @torch.no_grad()
    def analytic_hazard_and_poisson_exceedance(
        self,
        gm_levels,
        exposure_time: float,
        n_mag: int = 400,
    ):
        cfg = self.cfg
        device = cfg.device
        dtype = cfg.dtype
    
        y = torch.as_tensor(gm_levels, dtype=dtype, device=device)
        if y.ndim == 0:
            y = y[None]
    
        m = torch.linspace(cfg.mag_min, cfg.mag_max, n_mag, dtype=dtype, device=device)
    
        beta = torch.as_tensor(cfg.beta, dtype=dtype, device=device)
        width = cfg.mag_max - cfg.mag_min
        f_m = beta * torch.exp(-beta * (m - cfg.mag_min)) / (1.0 - torch.exp(-beta * width))
    
        distance = torch.full_like(m, cfg.distance_km)
        vs30 = torch.full_like(m, cfg.site_vs30)
        mu_ln, sig_ln = self.gm_model(m, distance, vs30)
    
        z = (torch.log(y)[:, None] - mu_ln[None, :]) / sig_ln[None, :]
        p_exc_cond = 0.5 * torch.erfc(z / math.sqrt(2.0))
    
        nu_y = cfg.lambda_rate * torch.trapz(p_exc_cond * f_m[None, :], m, dim=1)
        p_exc_T = 1.0 - torch.exp(-nu_y * exposure_time)
    
        return nu_y, p_exc_T


# -----------------------------------------------------------------------------
# Training helper
# -----------------------------------------------------------------------------

def train_flow_simple(
    model: RareGroundMotionFlow,
    *,
    steps: int = 5000,
    batch_size: int = 512,
    lr: float = 1e-4,
    weight_decay: float = 1e-6,
    print_every: int = 250,
) -> Dict[str, list]:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist: Dict[str, list] = {"loss": [], "rare_prob_surr": []}

    for step in range(1, steps + 1):
        out = model.objective(batch_size)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        hist["loss"].append(float(loss.detach().cpu()))
        hist["rare_prob_surr"].append(float(out["rare_prob_surr"].detach().cpu()))

        if step % print_every == 0 or step == 1:
            print(
                f"step={step:6d}  loss={hist['loss'][-1]: .5e}  "
                f"surrogate-rare={hist['rare_prob_surr'][-1]:.4f}"
            )
            
    return hist

def train_flow_staged(model, stages=((500,2.0),(500,5.0),(500,10.0),(500,20.0),(1000,35.0)),
                      batch_size=256, lr=2e-4, print_every=200):
    hist = {"loss": [], "rare_prob_surr": []}
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    for n_steps, alpha in stages:
        model.cfg.penalty_alpha = alpha
        for step in range(1, n_steps + 1):
            out = model.objective(batch_size)
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            hist["loss"].append(float(loss.detach().cpu()))
            hist["rare_prob_surr"].append(float(out["rare_prob_surr"].detach().cpu()))

            if step % print_every == 0 or step == 1:
                print(
                    f"alpha={alpha:5.1f} step={step:6d} loss={hist['loss'][-1]: .5e} "
                    f"surrogate-rare={hist['rare_prob_surr'][-1]:.4f}"
                )
    return hist

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = FlowConfig(
        seq_len=225,
        hidden_dim=256,
        n_coupling_layers=20,
        mag_min=4.0,
        mag_max=8.5,
        lambda_rate=3.0,
        b_value=1.0,
        t_max=50.0,
        gm_threshold=1.25,
        penalty_alpha=35.0,
        site_vs30=500.0,
        distance_km=20.0,
        device="cpu",
        dtype=torch.float64,
    )

    model = RareGroundMotionFlow(cfg).to(device=cfg.device, dtype=cfg.dtype)

    _ = train_flow_staged(model, batch_size=256, lr=2e-4, print_every=200)

    est = model.estimate_rare_event_probability_mixture(4000, eps_mix=0.2)

    print("\nMixture importance-sampling estimate:")
    for k, v in est.items():
        if torch.is_tensor(v):
            v = v.detach().cpu().item()
        print(f"  {k}: {v}")

    crude = model.estimate_rare_event_probability_crude(20000)
    print("\nCrude Monte Carlo under original model:")
    for k, v in crude.items():
        if torch.is_tensor(v):
            v = v.detach().cpu().item()
        print(f"  {k}: {v}")

    nu, pT = model.analytic_hazard_and_poisson_exceedance(
        gm_levels=cfg.gm_threshold,
        exposure_time=cfg.t_max,
        n_mag=600,
    )

    print(f"\nAnalytical full 50-year exceedance probability: {pT}")
    print(f"Analytical hazard rate: {nu}")

    mu = cfg.lambda_rate * cfg.t_max
    tail_prob = poisson_tail_prob(mu, cfg.seq_len)
    print(f"Poisson tail P[N(T) > seq_len] = {tail_prob:.3e}")

    diag = model.diagnose_flow_importance_weights(5000)
    print("\nFlow importance-weight diagnostic:")
    for k, v in diag.items():
        if torch.is_tensor(v):
            v = v.detach().cpu().item()
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()