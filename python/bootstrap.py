# Purpose: Estimate risk differences using IPTW and IOSW x IPTW with bootstrap SEs
# Author: Catie Wiener
# Date: Feb 2, 2026

import numpy as np
import pandas as pd
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from joblib import Parallel, delayed
from helpers import estimate_weights_and_risks

# read in data/random/combined.csv
all_sims = np.load("data/random/simulation_combined.npy", allow_pickle=True).item()
all_bootstraps = np.load("data/random/bootstrap_indices.npy", allow_pickle=True).item()

sims_n = len(all_sims)
#sims_n = 7
sim_results = [None]*sims_n

cols = ["l", "a", "y", "s"] # necessary columns to subset
B = all_bootstraps[0].shape[1] # number of bootstraps

start_cpu_time = time.perf_counter()

def run_one_sim_bs(k):
#for k in all_sims.keys():
    df = all_sims[k][cols].copy()
    bootstrap_indices = all_bootstraps[k].copy()
    
    res = estimate_weights_and_risks(df)
    results = [None] * B
    for b in range(B):
        indices = bootstrap_indices[:, b]
        df_boot = df.loc[indices]
        res_b = estimate_weights_and_risks(df_boot)
        res_b["bootstrap"] = b + 1
        results[b] = res_b
    
    results_df = pd.concat(results)
    # group by method and compute standard errors for risk difference
    results_df = results_df.groupby("method").agg(risk_diff_se   = ("rd", "std")).reset_index()
    res["bootstrap_se"] = results_df["risk_diff_se"].values
    # create column in res that indicates sim number 
    res["sim"] = k + 1
    return res


n_jobs = max(1, os.cpu_count() - 1)  # leave one core free
boot_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
    delayed(run_one_sim_bs)(k) for k in range(sims_n)
)
end_cpu_time = time.perf_counter()
elapsed_cpu_time =end_cpu_time - start_cpu_time
print(f"CPU time: {elapsed_cpu_time:.4f} seconds")

fin = pd.concat(boot_results)
fin.to_csv("data/results/boot_results.csv")

np.savetxt("data/results/boot_time.txt", [elapsed_cpu_time])