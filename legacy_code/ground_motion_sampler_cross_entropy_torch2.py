#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:11:44 2026

@author: glavrent

Fully differentiable Torch rewrite of the cross-entropy objective + optimization
for lambda and b (via log-parameters), while keeping the Monte Carlo sampling
(non-differentiable) as in the original script.

Key points:
- Sampling (events + ground motions) remains stochastic and uses numpy/pygmm.
- The *cross-entropy objective* and all densities used inside it are computed in torch
  and are differentiable w.r.t. log_lambda and log_b.
- We avoid SciPy minimize; use torch.optim.LBFGS (quasi-Newton style).
"""

from __future__ import annotations

## Load Libraries
#general
import copy
import numpy as np
from typing import Any, Callable, Mapping, Optional, Tuple, List, Dict
# plotting (optional, kept from original)
import matplotlib as mpl
from matplotlib import pyplot as plt
# external ground motion model libraries
import pygmm
# torch
import torch

# -----------------------------------------------
# Utility Functions
# -----------------------------------------------
def rejection_sample(
    pdf: Callable[[float], float],
    xmin: float,
    xmax: float,
    fmax: float,
    rng: Optional[np.random.Generator] = None,
    n_samp: int = 1,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    if xmax <= xmin:
        raise ValueError("Require xmax > xmin.")
    if fmax <= 0.0:
        raise ValueError("fmax must be positive.")

    samples = np.empty(n_samp, dtype=float)
    j = 0
    while j < n_samp:
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(0.0, fmax)
        if y <= pdf(x):
            samples[j] = x
            j += 1
    return samples


def event_mag_and_time_sample(
    time: float,
    params: Mapping[str, Any],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, dict]:
    """
    Same as original: Poisson inter-arrival (Exp(rate))
    and truncated exponential magnitude (via rejection sampling).
    """
    if rng is None:
        rng = np.random.default_rng()

    try:
        rate = float(params["lambda"])
        m_min = float(params["mag_min"])
        m_max = float(params["mag_max"])
        loc = np.array(params["loc"])
        dip = float(params["dip"])
    except KeyError as e:
        raise KeyError(f"Missing required parameter: {e.args[0]}") from e

    if "beta" in params:
        beta = float(params["beta"])
    elif "b" in params:
        beta = np.log(10.0) * float(params["b"])
    else:
        raise KeyError("Missing magnitude slope parameter: provide 'beta' or 'b'.")

    if rate <= 0.0:
        raise ValueError("params['lambda'] must be > 0.")
    if beta <= 0.0:
        raise ValueError("Magnitude slope beta must be > 0.")
    if m_max <= m_min:
        raise ValueError("Require params['mag_max'] > params['mag_min'].")

    dt = rng.exponential(scale=1.0 / rate)

    # truncated exponential on [m_min, m_max]
    mag_pdf = (
        lambda mag: beta
        * np.exp(-beta * (mag - m_min))
        / (1.0 - np.exp(-beta * (m_max - m_min)))
    )
    mag = rejection_sample(mag_pdf, m_min, m_max, mag_pdf(m_min))[0]

    time_new = time + dt
    event = {"mag": mag, "dip": dip, "time": time_new, "dt": dt, "loc": loc}
    return time_new, event


def sample_gm(event: dict, site: dict, period: float = 1.0) -> dict:
    """
    Same as original:
    - compute distance
    - get ln-mean and ln-std from pygmm
    - draw lognormal sample (numpy)
    """
    r = np.linalg.norm(event["loc"] - site["loc"])
    s = pygmm.model.Scenario(
        mag=event["mag"],
        dist_jb=r,
        dist_x=r,
        dist_rup=r,
        dip=event["dip"],
        v_s30=site["vs30"],
    )
    m_gmm = pygmm.ChiouYoungs2014(s)
    m_mean = float(m_gmm.interp_ln_spec_accels(periods=period))
    m_sd = float(m_gmm.interp_ln_stds(periods=period))
    m = np.random.lognormal(mean=m_mean, sigma=m_sd)

    gm = {
        "time": event["time"],
        "dt": event["dt"],
        "mag": event["mag"],
        "dip": event["dip"],
        "r": r,
        "vs30": site["vs30"],
        "gm": m,
        # store conditional ln-mean and ln-std so CE objective is torch-only
        "ln_mean": m_mean,
        "ln_sd": m_sd,
    }
    return gm


def gm_performance_fun(chain_gm: List[dict], gm_thres: float) -> bool:
    gm_array = np.array([scen_gm["gm"] for scen_gm in chain_gm])
    return bool((gm_array >= gm_thres).any())

# -----------------------------------------------
# Density Functions: Torch differentiable densities
# -----------------------------------------------
_LOG_2PI = float(np.log(2.0 * np.pi))
_LN10 = float(np.log(10.0))


def _to_torch(x, *, dtype=torch.float64, device="cpu") -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def exponential_pdf_torch(x: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    # rate * exp(-rate * x)
    return rate * torch.exp(-rate * x)


def lognorm_pdf_torch(x: torch.Tensor, mu_ln: torch.Tensor, sigma_ln: torch.Tensor) -> torch.Tensor:
    """
    Lognormal with parameters:
      ln(X) ~ Normal(mu_ln, sigma_ln)
    pdf(x) = 1/(x*sigma*sqrt(2π)) * exp(-(ln x - mu)^2/(2 sigma^2))
    """
    eps = torch.finfo(x.dtype).tiny
    x = torch.clamp(x, min=eps)
    sigma_ln = torch.clamp(sigma_ln, min=eps)

    logx = torch.log(x)
    z = (logx - mu_ln) / sigma_ln
    return torch.exp(-0.5 * z * z) / (x * sigma_ln * torch.sqrt(torch.tensor(2.0 * np.pi, dtype=x.dtype, device=x.device)))


def density_event_torch(mag: torch.Tensor, dt: torch.Tensor, lam: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Matches your original density_event behavior (even though it is not the truncated GR pdf).
    Original used: exponential_pdf(mag, beta) * exponential_pdf(dt, rate)
    with beta = ln(10)*b and rate=lambda.
    """
    beta = (_LN10) * b
    return exponential_pdf_torch(mag, beta) * exponential_pdf_torch(dt, lam)


def density_gm_torch(
    gm: torch.Tensor,
    mag: torch.Tensor,
    dt: torch.Tensor,
    lam: torch.Tensor,
    b: torch.Tensor,
    ln_mean: torch.Tensor,
    ln_sd: torch.Tensor,
) -> torch.Tensor:
    return density_event_torch(mag, dt, lam, b) * lognorm_pdf_torch(gm, ln_mean, ln_sd)

# -----------------------------------------------
# Torch: differentiable cross-entropy
# -----------------------------------------------
def cross_entropy_torch(
    chains_gm: List[List[dict]],
    h_array_np: np.ndarray,
    ssc_params_prop: Mapping[str, Any],
    ssc_params_samp: Mapping[str, Any],
    log_lambda: torch.Tensor,
    log_b: torch.Tensor,
    log_gm_bias: torch.Tensor,
    log_gm_sig: torch.Tensor,
    *,
    dtype=torch.float64,
    device="cpu",
) -> torch.Tensor:
    """
    Differentiable CE objective w.r.t. log_lambda, log_b.

    We compute:
      ce = Σ_j h_j * exp( Σ_i [log pdf_prob_ji - log pdf_samp_ji] ) * Σ_i log pdf_samp_upd_ji

    where:
      - pdf_prob uses proposal params (fixed)
      - pdf_samp uses current sampling params (fixed for this CE iteration)
      - pdf_samp_upd uses updated (lam,b) parameters (variables)
    """
    # fixed params (torch scalars)
    lam_prop     = _to_torch(float(ssc_params_prop["lambda"]), dtype=dtype, device=device)
    b_prop       = _to_torch(float(ssc_params_prop["b"]), dtype=dtype, device=device)
    gm_bias_prop = _to_torch(float(ssc_params_prop["gm_bias"]), dtype=dtype, device=device)
    gm_sig_prop  = _to_torch(float(ssc_params_prop["gm_sig"]), dtype=dtype, device=device)

    lam_samp     = _to_torch(float(ssc_params_samp["lambda"]), dtype=dtype, device=device)
    b_samp       = _to_torch(float(ssc_params_samp["b"]), dtype=dtype, device=device)
    gm_bias_samp = _to_torch(float(ssc_params_samp["gm_bias"]), dtype=dtype, device=device)
    gm_sig_samp  = _to_torch(float(ssc_params_samp["gm_sig"]), dtype=dtype, device=device)

    # variables (torch)
    lam_upd     = torch.exp(log_lambda)
    b_upd       = torch.exp(log_b)
    gm_bias_upd = torch.exp(log_gm_bias)
    gm_sig_upd  = torch.exp(log_gm_sig)
    
    # h array
    h = _to_torch(h_array_np.astype(np.float64), dtype=dtype, device=device)

    ce = torch.zeros((), dtype=dtype, device=device)

    eps = torch.finfo(torch.float64).tiny

    for j, chain in enumerate(chains_gm):
        if len(chain) == 0:
            continue

        gm      = _to_torch([sc["gm"] for sc in chain],      dtype=dtype, device=device)
        mag     = _to_torch([sc["mag"] for sc in chain],     dtype=dtype, device=device)
        dt      = _to_torch([sc["dt"] for sc in chain],      dtype=dtype, device=device)
        ln_mean = _to_torch([sc["ln_mean"] for sc in chain], dtype=dtype, device=device)
        ln_sd   = _to_torch([sc["ln_sd"] for sc in chain],   dtype=dtype, device=device)

        pdf_prob = density_gm_torch(gm, mag, dt, lam_prop, b_prop, ln_mean * gm_bias_prop, ln_sd * gm_sig_prop)
        pdf_samp = density_gm_torch(gm, mag, dt, lam_samp, b_samp, ln_mean * gm_bias_samp, ln_sd * gm_sig_samp)
        pdf_upd  = density_gm_torch(gm, mag, dt, lam_upd,  b_upd,  ln_mean * gm_bias_upd,  ln_sd * gm_sig_upd)
        
        print("density prob: ", pdf_prob[0])
        print("density samp: ", pdf_samp[0])
        print("density upd:  ", pdf_upd[0])

        # numerical safety
        pdf_prob = torch.clamp(pdf_prob, min=eps)
        pdf_samp = torch.clamp(pdf_samp, min=eps)
        pdf_upd = torch.clamp(pdf_upd, min=eps)

        w = h[j] * torch.exp(torch.sum(torch.log(pdf_prob) - torch.log(pdf_samp)))
        ce = ce + w * torch.sum(torch.log(pdf_upd))
        #print('w: ',w)
        #print(' torch.sum(torch.log(pdf_upd)) :',  torch.sum(torch.log(pdf_upd)))

    return ce

# -----------------------------------------------
# One CE iteration (sampling + torch opt)
# -----------------------------------------------
def iter_cross_entropy(
    ssc_params_prop: Dict[str, Any],
    ssc_params_samp: Dict[str, Any],
    site_params: Dict[str, Any],
    gm_thres: float,
    *,
    period: float = 1.0,
    device: str = "cpu",
    dtype=torch.float64,
    lbfgs_max_iter: int = 60,
) -> Tuple[Dict[str, Any], float]:
    
    
    # 1) Sample chains (non-differentiable)
    chains_gm: List[List[dict]] = []
    h_array: List[bool] = []

    if flag_verbose:
        print("Sampling:")

    rng = np.random.default_rng()

    for k in range(n_chains):
        chains_gm.append([])
        time = 0.0

        while True:
            time, event = event_mag_and_time_sample(time, ssc_params_prop, rng=rng)
            if time > t_max:
                break

            chains_gm[k].append(sample_gm(event, site_params, period=period))

            if flag_verbose:
                print(f"  mag: {chains_gm[k][-1]['mag']:.2f}, gm: {chains_gm[k][-1]['gm']:.4f} g")

        h_array.append(gm_performance_fun(chains_gm[k], gm_thres))

    h_array_np = np.array(h_array, dtype=float)

    # 2) Torch optimization of (log_lambda, log_b) to maximize CE
    #    (i.e., minimize -CE)
    log_lambda = torch.tensor(
        float(np.log(ssc_params_samp["lambda"])),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    log_b = torch.tensor(
        float(np.log(ssc_params_samp["b"])),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    
    log_gm_bias = torch.tensor(
        float(np.log(ssc_params_samp["gm_bias"])),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    
    log_gm_sig = torch.tensor(
        float(np.log(ssc_params_samp["gm_sig"])),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    

    params = [log_lambda, log_b, log_gm_bias, log_gm_sig]
    optimizer = torch.optim.Adam(
        params,
        lr=0.01,
        # max_iter=lbfgs_max_iter,
        # line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        ce = cross_entropy_torch(
            chains_gm       = chains_gm,
            h_array_np      = h_array_np,
            ssc_params_prop = ssc_params_prop,
            ssc_params_samp = ssc_params_samp,
            log_lambda      = log_lambda,
            log_b           = log_b,
            log_gm_bias     = log_gm_bias,
            log_gm_sig      = log_gm_sig,
            dtype           = dtype,
            device          = device,
        )
        loss = -ce
        loss.backward()
        return loss

    optimizer.step(closure)

    # 3) Evaluate final CE value
    with torch.no_grad():
        ce_final = cross_entropy_torch(
            chains_gm       = chains_gm,
            h_array_np      = h_array_np,
            ssc_params_prop = ssc_params_prop,
            ssc_params_samp = ssc_params_samp,
            log_lambda      = log_lambda,
            log_b           = log_b,
            log_gm_bias     = log_gm_bias,
            log_gm_sig      = log_gm_sig,
            dtype           = dtype,
            device          = device,
        ).item()

    # 4) Update sampling params
    ssc_params_samp_upd = copy.deepcopy(ssc_params_samp)
    ssc_params_samp_upd["lambda"]  = float(torch.exp(log_lambda).detach().cpu().item())
    ssc_params_samp_upd["b"]       = float(torch.exp(log_b).detach().cpu().item())
    ssc_params_samp_upd["gm_bias"] = float(torch.exp(log_gm_bias).detach().cpu().item())
    ssc_params_samp_upd["gm_sig"]  = float(torch.exp(log_gm_sig).detach().cpu().item())

    return ssc_params_samp_upd, float(ce_final)

# -----------------------------------------------
# Main (kept close to original)
# -----------------------------------------------
flag_verbose = False

# number of chains and length
n_chains = 10
t_max = 50

# ground motion threshold
gm_thres = 0.5

# params
ssc_params = {
    "loc": np.array([0.0, 10.0]),
    "dip":     90.0,
    "lambda":  10.0,
    "mag_min": 4.0,
    "mag_max": 8.5,
    "b":       1.0,
    "gm_bias": 1.0,
    "gm_sig":  1.0
}
site_params = {"loc": np.array([0.0, 0.0]), "vs30": 500.0}

# number of CE iterations
n_iter = 10

# initialize arrays
ce_array:      List[float] = []
rate_array:    List[float] = []
b_array:       List[float] = []
gm_bias_array: List[float] = []
gm_sig_array:  List[float] = []
ssc_params_array: List[Dict[str, Any]] = []

ssc_params_upd = copy.deepcopy(ssc_params)

print("Cross-entropy optimization (Torch differentiable objective)")
for j in range(n_iter):
    print(" iteration:", j + 1)

    ssc_params_upd, ce_val = iter_cross_entropy(
        ssc_params_prop=ssc_params,
        ssc_params_samp=ssc_params_upd,
        site_params=site_params,
        gm_thres=gm_thres,
        period=1.0,
        device="cpu",
        lbfgs_max_iter=60,
    )

    ssc_params_array.append(copy.deepcopy(ssc_params_upd))
    rate_array.append(ssc_params_upd["lambda"])
    b_array.append(ssc_params_upd["b"])
    gm_bias_array.append(ssc_params_upd["gm_bias"])
    gm_sig_array.append(ssc_params_upd["gm_sig"])
    ce_array.append(ce_val)

    print(f"   CE: {ce_val:.6e}, lambda: {ssc_params_upd['lambda']:.6g}, b: {ssc_params_upd['b']:.6g}")
    
    
#plot shear-wave velocity
fname_fig = 'parameter_evolution'
fig, ax = plt.subplots(figsize = (15,10))
ax.scatter(range(n_iter), b_array, label='B-value')
# ax.scatter(range(n_iter), rate_array, label='Rate')
#edit figure
ax.set_xlabel(r'Iteration',     fontsize=35)
ax.set_ylabel(r'Parameter Value', fontsize=35)
ax.legend(loc='lower left', fontsize=35)
ax.grid(which='both')
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)

fig.tight_layout()
# fig.savefig(dir_fig + fname_fig + '.png')
