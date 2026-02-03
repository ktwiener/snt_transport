# Purpose: Simulate sequential nested trial data for two visits and select
#          bootstrap resamples. 
# Author: Catie Wiener
# Date: Feb 1, 2026

import numpy as np
import pandas as pd
from helpers import generate_outcome

np.random.seed(24601)

sims = 10  # Number of simulations
B = 300    # Number of bootstrap samples
n = 5000  # Sample size
pl = 0.45 # Prevalence of outcome predictor L
pa = 0.5  # Probability of treatment (randomized)

py11 = 0.7  # P(Y^1=1|L=1) = 0.7
py10 = 0.15 # P(Y^1=1|L=0) = 0.15
py01 = 0.9  # P(Y^0=1|L=1) = 0.9
py00 = 0.05 # P(Y^0=1|L=0) = 0.05

truth = (py11 - py01)*pl + (py10 - py00)*(1-pl)
sim_data = {}
bootstrap_indices = {}

for k in range(sims):
# trial 1

    trial1 = pd.DataFrame({
        "sim": k + 1,
        "s": 0, # s = 0 for iosw eventually
        "trial": 1, # denote trial
        "id":  np.tile(np.arange(1, n + 1), 1),
        "l": np.random.binomial(1, pl, size=n*1),  # L ~ Bern(pl)
        "a": np.random.binomial(1, pa, size=n*1)   # A ~ Bern(pa)
    })

    trial1 = generate_outcome(trial1, py11, py10, py01, py00)


    # trial 2

    ## filter to individuals without the outcome
    trial2 = trial1.loc[trial1["y"] == 0].copy()
    # re-randomize A
    trial2["a"] = np.random.binomial(1, trial2["a"]*1 + (1-trial2["a"])*pa)  
    trial2["trial"] = 2
    # generate new outcomes from trial 2
    trial2 = generate_outcome(trial2, py11, py10, py01, py00)

    # combine trials
    analysis_data = pd.concat([trial1, trial2])
    analysis_data["s"] = 1  # s = 1 for analysis data

    trial1["y"] = np.nan  # set outcome to missing for target pop
    trial1["a"] = np.nan  # set exposure to missing for target pop

    # create target population and stack with combined data
    combined= pd.concat([analysis_data, trial1]).sort_values("id")

    # bootstrap ids for each simulation
    bootstrap_indices[k] = np.random.randint(0, n, size=(n, B))

    sim_data[k] = combined





# save out o data/random folder
np.save("data/random/simulation_combined.npy", sim_data)

np.save("data/random/bootstrap_indices.npy", bootstrap_indices)