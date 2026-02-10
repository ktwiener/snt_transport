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
#all_bootstraps = np.load("data/random/bootstrap_indices.npy", allow_pickle=True).item()

sims_n = len(all_sims)
n = 5000
#sims_n = 7

sim_results = [None]*sims_n

cols = ["id", "l", "a", "y", "s"] # necessary columns to subset
B = 500

start_cpu_time = time.perf_counter()

def run_one_sim_bs(k):
    ss = np.random.SeedSequence([123, k])
    rng = np.random.default_rng(ss)

    df = all_sims[k][cols].copy()

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