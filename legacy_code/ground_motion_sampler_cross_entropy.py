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
#
import copy
# numerical and scientific libraries
import scipy
import numpy as np
from scipy.optimize import minimize
# statistics libraries
import pandas as pd
# ploting libraries
import matplotlib as mpl
from matplotlib import pyplot as plt
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
    gm = {'time':event['time'], 'dt':event['dt'], 'mag':event['mag'], 'dip':event['dip'], 'r':r, 'vs30':site['vs30'], 'gm':m}
    
    return gm


def exponential_pdf(x, rate):
    return rate * np.exp(-rate * x)

def density_event(event, params):
    
    #parse and validate parameters
    try:
        mag   = float(event["mag"])
        dt    = float(event["dt"])
        rate  = float(params["lambda"])
        m_min = float(params["mag_min"])
        m_max = float(params["mag_max"])
        if "beta" in params:
            beta = float(params["beta"])
        elif "b" in params:
            beta = np.log(10.0) * float(params["b"])
        else:
            raise KeyError("Missing magnitude slope parameter: provide 'beta' or 'b'.")
    except KeyError as e:
        raise KeyError(f"Missing required parameter: {e.args[0]}") from e
        
        
    #evaluate pdf to a constant    
    pdf = exponential_pdf(mag, beta) * exponential_pdf(dt, rate)
    
    return pdf    
    

def conditional_density_gm(gm, event, site, period:float = 1):
    
    #compute rupture distance
    if 'r' in event.keys():
        r = event['r']
    else:
        r = np.linalg.norm(event["loc"] - site["loc"])
    
    #scenario definition
    s = pygmm.model.Scenario(mag=event['mag'], dist_jb=r, dist_x=r, dist_rup=r, dip=event['dip'], v_s30=site['vs30'])
    m_gmm = pygmm.ChiouYoungs2014(s)
    #evaluate mean and std
    m_mean = float( m_gmm.interp_ln_spec_accels(periods=period) )
    m_sd   = float( m_gmm.interp_ln_stds(periods=period) )
    
    pdf = scipy.stats.lognorm.pdf(gm, scale=np.exp(m_mean), s=m_sd)
    
    return pdf

def density_gm(gm, event, site, ssc_params, period:float = 1):
    
    # import pdb; pdb.set_trace()
    pdf  = density_event(event, ssc_params)
    pdf *= conditional_density_gm(gm, event, site, period)
 
    return pdf 

def density_chain(chain_gm, ssc_params, param_lambda=None, param_b=None):

    #update labmda    
    if not param_lambda is None:
        ssc_params['lambda'] = param_lambda
    if not param_b is None:
        ssc_params['b'] = param_b
    
    #evaluate ground motion parameter
    return [ float( density_gm(scen_gm['gm'], scen_gm, scen_gm, ssc_params) ) for scen_gm in chain_gm]
    

def gm_performance_fun(chain_gm, gm_thres):
    
    gm_array = np.array([scen_gm['gm'] for scen_gm in chain_gm])
    
    return ( gm_array >= gm_thres ).any()

def cross_entropy(chains_gm, h_array, pdf_prob, pdf_samp, ssc_params_samp, param_lambda=None, param_b=None):
    
    #number of samples
    n_samp = len(h_array)
    
    #evaluate new sampling probability
    pdf_samp_upd = [ np.array( density_chain(chain_gm, ssc_params_samp, param_lambda=param_lambda, param_b=param_b) ) for chain_gm in chains_gm] 
    
    #evaluate cross entropy
    ce = 0.
    for j in range(n_samp):
        ce += h_array[j] * np.exp( np.sum( np.log(pdf_prob[j]) - np.log(pdf_samp[j]) ) ) * np.sum( np.log(pdf_samp_upd[j]) )
    
    return float(ce)

def iter_cross_entropy(ssc_params_prop, ssc_params_samp, site_params, gm_thres):
    
    #initalize chain
    chains_gm = []

    #performance array 
    h_array = []
    
    
    #generate samples using proposal distribution (t_{i-1})    
    #iterate over chains
    if flag_verbose: print("Sampling:")
    for k in range(n_chains):
        
        #initalize current chain
        chains_gm.append([])
        time = 0.
        
        #sample chain unit end time
        while True:
            #sample event
            time, event = event_mag_and_time_sample(time, ssc_params_prop)
            
            #stop once time is reached
            if time > t_max:
                break 
            
            #sample ground motion
            chains_gm[k].append( sample_gm(event, site_params) )
            
            #evaluate pdf (sampling distribution)
            pdf_samp = density_gm(chains_gm[k][-1]['gm'], event, site_params, ssc_params_samp)
            pdf_prob = density_gm(chains_gm[k][-1]['gm'], event, site_params, ssc_params_prop)
            chains_gm[k][-1]["pdf_samp"] = pdf_samp
            chains_gm[k][-1]["pdf_prob"] = pdf_prob
            
            #print output
            if flag_verbose:
                print('  mag: %.2f, gm: %.4f g'%(chains_gm[k][-1]['mag'], chains_gm[k][-1]['gm']))
                
        
        #compute performance value
        h_array.append( gm_performance_fun(chains_gm[k], gm_thres) )
     
    #summarize perfomance array
    h_array = np.array(h_array)
    
    #summarize sampled pdf
    pdf_samp = [ np.array([scen_gm['pdf_samp'] for scen_gm in chain_gm]) for chain_gm in chains_gm] 
    pdf_prob = [ np.array([scen_gm['pdf_prob'] for scen_gm in chain_gm]) for chain_gm in chains_gm] 


    #objective function handle: cross entropy
    obj_fun = lambda param: -1. * cross_entropy(chains_gm, h_array, pdf_prob, pdf_samp, ssc_params_samp, 
                                                param_lambda=np.exp(param[0]), param_b=np.exp(param[1]))
    
    #find new rate by maximizing cross entropy 
    param_upd = minimize(obj_fun, [np.log(ssc_params_samp['lambda']), np.log(ssc_params_samp['b'])] )
    #compute final cross entropy
    ce_val = -1. * obj_fun(param_upd)
    
    #update source parameters
    ssc_params_samp_upd = copy.deepcopy(ssc_params_samp)
    ssc_params_samp_upd['lambda'] = np.exp(param_upd[0])
    ssc_params_samp_upd['b']      = np.exp(param_upd[1])

    return ssc_params_samp_upd, ce_val  
    

#printout for debuging
flag_verbose = False 
   
#number of chains and lenght
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

#number of cross-entropy iteration
n_iter = 10

#initialize arrays
ce_array   = []
rate_array = []
b_array    = []
ssc_params_array = []

ssc_params_upd = ssc_params

#cross entropy optimization
print("Cross-entropy optimization")
for j in range(n_iter):
    
    print(" iteration: ",j+1)
    
    ssc_params_upd, ce_val = iter_cross_entropy(ssc_params, ssc_params_upd, site_params, gm_thres)
    
    ssc_params_array.append(ssc_params_upd)
    rate_array.append(ssc_params_array['lambda'])
    b_array.append(ssc_params_array['b'])
    ce_array.append(ce_val)
    
#
# iter_array = np.arange(n_iter)
# #magnitude-distance distribution
# fname_fig = 'distribution_M-R_taiwan'
# fig, ax = plt.subplots(figsize = (10,10), ncol=2)
# ax[0].plot(iter_array, rate_array, 'd', markersize=12)
# ax[1].plot(iter_array, b_array,    's', markersize=12)
# #edit figure
# ax.set_xlim([0., 50.])
# ax.set_ylim([3., 8.])
# ax.set_xlabel(r'$R_{rup}$ ($km$)', fontsize=35)
# # ax.set_xlabel(r'Hypocenter Distance (km)',  fontsize=35)
# ax.set_ylabel(r'Magnitude',                 fontsize=35)
# ax.legend(loc='lower right', fontsize=35)
# ax.grid(which='both')
# ax.tick_params(axis='x', labelsize=30)
# ax.tick_params(axis='y', labelsize=30)
# fig.tight_layout()

