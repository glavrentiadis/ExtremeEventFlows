#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 13:16:29 2026

@author: glavrent
"""

## Load Libraries
# general
from __future__ import annotations
from typing import Any, Callable, Mapping, Optional, Tuple
# numerical and scientific libraries
import scipy
import numpy as np
# statistics libraries
import pandas as pd
#ground motion libraries
import pygmm

## 
def rejection_sample(pdf: Callable[[float], float],
                     xmin: float,
                     xmax: float,                     
                     fmax: float,
                     rng: Optional[np.random.Generator] = None,
                     n_samp: int = 1
                     ) -> np.ndarray:

    #define random generator    
    if rng is None:
        rng = np.random.default_rng()

    #define x & y limits
    if xmax <= xmin:
        raise ValueError("Require xmax > xmin.")
    if fmax <= 0.0:
        raise ValueError("fmax must be positive.")

    #list of samples
    samples = np.empty(n_samp, dtype=float)

    #sample counter
    j = 0
    
    #generate samples
    while j < n_samp:
        
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(0.0, fmax)
        
        if y <= pdf(x):
            samples[j] = x
        else:
            continue
        
        j += 1
        
    return samples

def event_mag_and_time_sample(
    time: float,
    params: Mapping[str, Any],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, dict]:
    """
    Sample an event inter-arrival time and magnitude.

    Time: Poisson process inter-arrival time
        dt ~ Exp(rate)

    Magnitude: truncated exponential (equivalent to truncated Gutenberg–Richter
    in natural-log form) on [m_min, m_max]
        p(m) ∝ exp(-beta (m - m_min))

    Parameters
    ----------
    time
        Current simulation time.
    params
        Model parameters. Required keys:
            - 'lambda'  : float, event rate (>0)
            - 'mag_min' : float
            - 'mag_max' : float (must be > mag_min)
        Magnitude slope specified by either:
            - 'beta' : float (>0), or
            - 'b'    : float (>0), where beta = ln(10) * b
    rng
        Numpy random number generator for reproducibility.

    Returns
    -------
    time_new, event
        Updated time and an event dictionary with keys: 'mag', 'time', 'dt'.
    """
    if rng is None:
        rng = np.random.default_rng()

    #parse and validate parameters
    try:
        rate = float(params["lambda"])
        m_min = float(params["mag_min"])
        m_max = float(params["mag_max"])
        loc   = np.array(params["loc"])
        dip   = float(params["dip"])
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

    #sample inter-arrival time
    dt = rng.exponential(scale=1.0 / rate)
    
    #sample magnitude from truncated exponential
    mag_pdf = lambda mag: beta * np.exp(-beta * (mag - m_min) ) / ( 1.0 - np.exp(-beta * (m_max - m_min)) )
    mag = rejection_sample(mag_pdf, m_min, m_max, mag_pdf(m_min))[0]    

    #update and return
    time_new = time + dt
    event = {"mag": mag, "dip": dip, "time": time_new, "dt": dt, 'loc':loc}
    return time_new, event

def sample_gm(event, site, period:float =1.):
    
    #compute rupture distance
    r = np.linalg.norm(event["loc"] - site["loc"])
    
    #scenario definition
    s = pygmm.model.Scenario(mag=event['mag'], dist_jb=r, dist_x=r, dist_rup=r, dip=event['dip'], v_s30=site['vs30'])
    m_gmm = pygmm.ChiouYoungs2014(s)
    #evaluate mean and std
    m_mean = float( m_gmm.interp_ln_spec_accels(periods=period) )
    m_sd   = float( m_gmm.interp_ln_stds(periods=period) )
    #take random sample
    m = np.random.lognormal(mean=m_mean, sigma=m_sd)
    
    #collect random sample info
    gm = {'time':event['time'], 'mag':event['mag'], 'dip':event['dip'], 'r':r, 'vs30':site['vs30'], 'gm':m}
    
    return gm
 
#
flag_verbose = True 
   
#
n_chains = 100
t_max = 50
#ground motion threshold
gm_thres = 0.5

#params 
ssc_params =  {'loc':np.array([0., 10.]), 
               'dip': 90.,
               'lambda':10, 
               'mag_min':4.0,
               'mag_max':9.0, 
               'b':1.}
site_params = {'loc':np.array([0., 0.]),
               'vs30': 500.}

#initalize chain
chains_gm = []


#iterate over chains
if flag_verbose: print("Sampling:")
for k in range(n_chains):
    
    #initalize current chain
    chains_gm.append([])
    time = 0.
    
    #sample chain unit end time
    while True:
        #sample event
        time, event = event_mag_and_time_sample(time, ssc_params)
        
        #stop once time is reached
        if time > t_max:
            break 
        
        #sample ground motion
        chains_gm[k].append( sample_gm(event, site_params) )
        
        #print output
        if flag_verbose:
            print('  mag: %.2f, gm: %.4f g'%(chains_gm[k][-1]['mag'], chains_gm[k][-1]['gm']))

    #summarize sampled events into a dataframe
    chains_gm[k] = pd.DataFrame(chains_gm[k])

#threshold 
chain_gm_thres = [np.any(c_gm.gm >= gm_thres) for c_gm in chains_gm]

#compute probability of exceedance
P_gm = np.sum(chain_gm_thres) / len(chain_gm_thres)

#print probability of exceedance
print("Probability of exceedence:\n\tP(SA>%.2f in %i years)=%.3f"%(gm_thres, t_max, P_gm))





