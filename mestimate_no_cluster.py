# Purpose: Use M-estimation to estimate risk differences using IPTW and IOSW x IPTW
# Author: Catie Wiener
# Date: Feb 2, 2026

import numpy as np
import pandas as pd
from delicatessen import MEstimator
from delicatessen.estimating_equations import ee_regression
from delicatessen.utilities import inverse_logit

# read in data/random/combined.csv
all_sims = np.load("data/random/simulation_combined.npy", allow_pickle=True).item()

d = all_sims[0]
d["intercept"] = 1.0

s = np.asarray(d['s'])
a = np.asarray(d['a'])
W = np.asarray(d[['intercept', 'l']])
Y = np.asarray(d['y'])

y_no_nan = np.asarray(d['y'].fillna(-999))
a_no_nan = np.asarray(d['a'].fillna(-999))

def psi(theta):
    # Dividing parameters into corresponding parts and labels from slides
    alpha = theta[0:2]              # Propensity score model coefficients
    beta  = theta[2:4]              # Sampling model coefficients
    mu0, mu1 = theta[4], theta[5]   # Causal risks
    delta1 = theta[6]               # Causal contrast

    # Logistic regression model for propensity score among analysis sample
    ee_logit_ps = ee_regression(theta=alpha,       # Regression model
                                y=a_no_nan,               # ... for exposure
                                X=W,               # ... given confounders
                                model='logistic')  # ... logistic model

    # Transforming logistic model coefficients into causal parameters
    pscore = inverse_logit(np.dot(W, alpha))    # Propensity score
    iptw = (1-s)*(a_no_nan/pscore + (1-a_no_nan)/(1-pscore))  # Corresponding weights

    # Logistic regression model for target pop probability
    ee_logit_samp = ee_regression(theta=beta,       # Regression model
                                  y=s,               # ... for being sampled
                                  X=W,               # ... given confounders
                                  model='logistic')  # ... logistic model

    # Transforming logistic model coefficients into causal parameters
    sampling_p = inverse_logit(np.dot(W, beta))    # Sampling probability among sampled
    iosw = (1-s)*(1-sampling_p) / sampling_p

    # Combined weights
    wt = iosw * iptw

    # Estimating function for causal risk under a=1
    ee_r1 = a_no_nan*y_no_nan*wt - mu1         # Weighted conditional mean

    # Estimating function for causal risk under a=0
    ee_r0 = (1-a_no_nan)*y_no_nan*wt - mu0     # Weighted conditional mean

    # Estimating function for causal risk difference
    ee_rd = np.ones(d.shape[0])*((mu1 - mu0) - delta1)

    # Returning stacked estimating functions in order of parameters
    return np.vstack([ee_logit_ps,   # EF of propensity score model
                      ee_logit_samp, # EF of sampling model
                      ee_r0,      # EF of causal risk a=0
                      ee_r1,      # EF of causal risk a=1
                      ee_rd])     # EF of causal contrast

estr = MEstimator(psi, init=[0, 0, 0, 0, 0.5, 0.5, 0])
estr.estimate()

# Formatting results into a nice table
result = pd.DataFrame()
result['Param'] = ['alpha_0', 'alpha_1', 'beta_0', 'beta_1', 'mu_0', 'mu_1', 'delta']
result['Coef'] = estr.theta
ci = estr.confidence_intervals()
result['LCL'] = ci[:, 0]
result['UCL'] = ci[:, 1]
result.round(2)

np.save("data/results/mestimation_results.npy", h)