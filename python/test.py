# Purpose: Test one iteration of M-estimation
# Author: Catie Wiener
# Date: Feb 10, 2026

import numpy as np
import pandas as pd
import time

from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit
from delicatessen.utilities import aggregate_efuncs

from python.helpers import estimate_weights_and_risks
from python.simulate import generate_one_sim

np.random.seed(24601)

pl = 0.45 # Prevalence of outcome predictor L
pa = 0.2  # Probability of treatment (randomized)

py11 = 0.7  # P(Y^1=1|L=1) = 0.7
py10 = 0.15 # P(Y^1=1|L=0) = 0.15
py01 = 0.9  # P(Y^0=1|L=1) = 0.9
py00 = 0.05 # P(Y^0=1|L=0) = 0.05

d = generate_one_sim(1, n = 500)

start_cpu_time = time.perf_counter()
# mestimation approach/ timing 
d["intercept"] = 1.0

s = np.asarray(d['s']) # s = 0 for target pop, s = 1 to analysis pop
W = np.asarray(d[['intercept', 'l']]) 
group = np.asarray(d['id'])

y_no_nan = np.asarray(d['y'].fillna(-1))
a_no_nan = np.asarray(d['a'].fillna(-1))

def psi(theta):
    # Dividing parameters into corresponding parts and labels from slides
    alpha = theta[0:2]              # Propensity score model coefficients
    beta  = theta[2:4]              # Sampling model coefficients
    mu1, mu0   = theta[4], theta[5] # Weighted risks IOSW x IPTW
    gamma1, gamma0 = theta[6], theta[7] # IPTW weighted risks
    delta, delta_ipt = theta[8], theta[9] # risk differences

    # Logistic regression model for propensity score among analysis sample
    ee_logit_ps = ee_regression(theta=alpha,       # Regression model
                                y=a_no_nan,               # ... for exposure
                                X=W,               # ... given confounders
                                model='logistic')  # ... logistic model
    ee_logit_ps = ee_logit_ps*s # only s = 1 (analysis sample) contributes

    # Logistic regression model for target pop probability (full pop contributes)
    ee_logit_samp = ee_regression(theta=beta,       # Regression model
                                y=s,               # ... for being sampled
                                X=W,               # ... given confounders
                                model='logistic')  # ... logistic model

    # Calculate weights
    pscore = inverse_logit(np.dot(W, alpha))                # Propensity score
    iptw = (s)*(a_no_nan/pscore + (1-a_no_nan)/(1-pscore))  # Corresponding weights for analysis sample
    
    psample = inverse_logit(np.dot(W, beta)) # Probability of being in analysis sample 
    iosw = s*(1-psample)/psample # Weights are inverse odds of being in analysis sample (only for those in sample)

    wt = iptw*iosw # multiply weights goether

    # IPTWxIOSW
    # Estimating function for  causal risk under a=1
    ee_r1 = a_no_nan*wt*(y_no_nan - mu1)       # Weighted conditional mean (HAJEK)
    # Estimating function for causal risk under a=0
    ee_r0 = (1-a_no_nan)*wt*(y_no_nan - mu0)    # Weighted conditional mean (HAJEK)
    # Estimating function for causal RD
    ee_rd = np.ones(d.shape[0])*((mu1 - mu0) - delta)

    # IPTW ONLY
    # Estimating function for causal risk under a=1
    ee_ipt_r1 = s*(a_no_nan*y_no_nan*iptw - gamma1)         # Weighted conditional mean
    
    # Estimating function for causal risk under a=0
    ee_ipt_r0 = s*((1-a_no_nan)*y_no_nan*iptw - gamma0)   # Weighted conditional mean
    # Estimating function for ccausal RD
    ee_ipt_rd = np.ones(d.shape[0])*((gamma1 - gamma0) - delta_ipt)

    ee_group_level = aggregate_efuncs(np.vstack([ee_logit_ps,
                                                ee_logit_samp,
                                                ee_r1, ee_r0,
                                                ee_ipt_r1, ee_ipt_r0,
                                                ee_rd, ee_ipt_rd]),
                                        group = group)
    # Returning stacked estimating functions in order of parameters
    return ee_group_level

estr = MEstimator(psi, init=[-0.99, -0.15, 0.66, -0.53, -0.035, 0.002, 
                                0.327, 0.329, 0.432, 0.397])
estr.estimate()

# Formatting results into a nice table
result = pd.DataFrame()
result['Param'] = ['alpha_0', 'alpha_1', 
                'beta_0', 'beta_1', 
                'mu_1', 'mu_0',
                'ipt_mu_1', 'ipt_mu_0', 
                'delta', 'delta_ipt']
result['Coef'] = estr.theta
ci = estr.confidence_intervals()
result['SE'] = np.sqrt(np.diag(estr.variance))
result['LCL'] = ci[:, 0]
result['UCL'] = ci[:, 1]

end_cpu_time = time.perf_counter()
mest_cpu_time =end_cpu_time - start_cpu_time


# bootstrapping test
cols = ["id", "l", "a", "y", "s"] # necessary columns to subset
df = d.copy()
B = 500

start_cpu_time = time.perf_counter()

ss = np.random.SeedSequence([123, 1])
rng = np.random.default_rng(ss)

# ---- precompute cluster structure once ----
# factorize gives cluster codes 0..m-1 (fast to sample)
cluster_codes, cluster_uniques = pd.factorize(df["id"], sort=False)
m = len(cluster_uniques)

# list of row-index arrays, one per cluster code
# (this is the key speedup vs dict of sub-DataFrames)
idx_by_cluster = [np.flatnonzero(cluster_codes == j) for j in range(m)]

# point estimate on original data
res = estimate_weights_and_risks(df)

# bootstrap
results = [None] * B
for b in range(B):
    sampled = rng.integers(0, m, size=m)  # sample cluster *codes* with replacement
    boot_idx = np.concatenate([idx_by_cluster[j] for j in sampled], axis=0)

    # take rows in one shot
    df_boot = df.iloc[boot_idx]  # keep original index; no need to reset unless your estimator requires it
    res_b = estimate_weights_and_risks(df_boot)

    res_b["bootstrap"] = b + 1
    results[b] = res_b

results_df = pd.concat(results, ignore_index=True)

se_df = (
    results_df
    .groupby("method", as_index=False)
    .agg(risk_diff_se=("rd", "std"))
)
res = (
    res.merge(se_df, on="method", how="left")
        .rename(columns={"risk_diff_se": "bootstrap_se"})
)

end_cpu_time = time.perf_counter()
boot_cpu_time =end_cpu_time - start_cpu_time