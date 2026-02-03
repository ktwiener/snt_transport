# Purpose: Estimate risk differences using IPTW and IOSW x IPTW with bootstrap SEs
# Author: Catie Wiener
# Date: Feb 2, 2026

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from helpers import estimate_weights_and_risks

# read in data/random/combined.csv
all_sims = np.load("data/random/simulation_combined.npy", allow_pickle=True).item()
all_bootstraps = np.load("data/random/bootstrap_indices.npy", allow_pickle=True).item()

sim_results = [None]*len(all_sims)

cols = ["l", "a", "y", "s"] # necessary columns to subset
B = all_bootstraps[0].shape[1] # number of bootstraps

for k in all_sims.keys():
    df = all_sims[k][cols]
    bootstrap_indices = all_bootstraps[k]
    
    res = estimate_weights_and_risks(df)
    results = [None] * B
    def one_boot(b):
        indices = bootstrap_indices[:, b]
        df_boot = df.loc[indices]
        res_b = estimate_weights_and_risks(df_boot)
        res_b["bootstrap"] = b + 1
        #results[b] = res_b
        return res_b
    
    results = Parallel(n_jobs=-1)(delayed(one_boot)(b) for b in range(B))
    results_df = pd.concat(results)
    # group by method and compute standard errors for risk difference
    results_df = results_df.groupby("method").agg(risk_diff_se   = ("rd", "std")).reset_index()
    res["bootstrap_se"] = results_df["risk_diff_se"].values
    # create column in res that indicates sim number 
    res["sim"] = k + 1
    sim_results[k] = res

np.save("data/results/bootstrap_results.npy", sim_results)
final_boots = pd.concat(sim_results)
final_boots.to_csv("data/results/bootstrap_results.csv")
