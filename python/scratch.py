import numpy as np
import pandas as pd

mest = np.load("data/results/mestimation_results.npy", allow_pickle=True)

d = combined
d["intercept"] = 1.0

s = np.asarray(d['s']) # s = 0 for target pop, s = 1 to analysis pop
W = np.asarray(d[['intercept', 'l']]) 
group = np.asarray(d['id'])
sample = (d["trial"] == 1).astype(int)
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
                                y=sample,               # ... for being sampled
                                X=W,               # ... given confounders
                                model='logistic')  # ... logistic model

    # Calculate weights
    pscore = inverse_logit(np.dot(W, alpha))                # Propensity score
    iptw = (s)*(a_no_nan/pscore + (1-a_no_nan)/(1-pscore))  # Corresponding weights for analysis sample
    
    psample = inverse_logit(np.dot(W, beta)) # Probability of being in analysis sample 
    iosw = s*(psample)/(1-psample) # Weights are inverse odds of being in analysis sample (only for those in sample)

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

estr = MEstimator(psi, init=[0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0])
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
result['sim'] = k + 1