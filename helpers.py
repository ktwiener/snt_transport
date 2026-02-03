import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def generate_outcome(df, py11, py10, py01, py00):
    """
    Generate potential outcomes Y^1, Y^0 and observed Y
    under consistency, conditional on L and A.
    """
    df = df.copy()

    df["ya1"] = np.random.binomial(
        1,
        df["l"] * py11 + (1 - df["l"]) * py10
    )

    df["ya0"] = np.random.binomial(
        1,
        df["l"] * py01 + (1 - df["l"]) * py00
    )

    df["y"] = df["a"] * df["ya1"] + (1 - df["a"]) * df["ya0"]

    return df


def estimate_risks_rd(df, a_col="a", y_col="y", w_col=None):
    """
    Estimate risk under A=0 and A=1, and risk difference (1-0).
    Returns a 1-row DataFrame with columns: risk0, risk1, rd, n0, n1.
    """
    out = {}

    for a in (0, 1):
        d = df.loc[df[a_col] == a, [y_col] + ([w_col] if w_col else [])].copy()
        n = len(d)
        if w_col is None:
            risk = d[y_col].mean() if n > 0 else np.nan
        else:
            w = d[w_col].to_numpy()
            y = d[y_col].to_numpy()
            risk = (w @ y) / w.sum() if (n > 0 and w.sum() > 0) else np.nan
            n = w.sum()

        out[f"risk{a}"] = risk
        out[f"n{a}"] = n

    out["rd"] = out["risk1"] - out["risk0"]
    return pd.DataFrame([out])

def estimate_weights (df):
    """
    Estimate IPTW and IOSW weights and add to DataFrame.
    Returns DataFrame with new columns: ps, iptw, p_sample, iosw, iosw_iptw.
    """
    df = df.copy()
    # filter to analysis sample
    ad = df.loc[df["s"] == 1].copy()

    # propensity score model fit on analysis data
    ps_model = smf.logit("a ~ l", data = ad).fit()
    ad["ps"] = ps_model.predict(ad)

    # iptw = a/ps + (1-a)/(1-ps)
    ad["iptw"] = ad["a"]/ad["ps"] + (1 - ad["a"])/(1 - ad["ps"])
    
    # sampling model fit on combined data
    sampling_model = smf.logit("s ~ l", data = df).fit()
    ad["p_sample"] = sampling_model.predict(ad) # predict on analysis data
    
    # iosw = (1 - p_sample)/p_sample * p(s=1)/p(s=0)
    p_s1 = ad["p_sample"].mean() # marginal probability of being in the analysis data
    ad["iosw"] = (1 - ad["p_sample"]) / ad["p_sample"] * (p_s1 / (1 - p_s1))

    # iosw x iptw
    ad["iosw_iptw"] = ad["iosw"] * ad["iptw"]

    return ad

def estimate_weights_and_risks(df):
    """
    Estimate weights and risks/risk differences on the analysis data.
    Returns a DataFrame with results for crude, IPTW, and IOSW x IPTW methods.
    """
    ad = estimate_weights(df)
    
    # estimate the risks and risk differences 
    crude = estimate_risks_rd(ad, a_col="a", y_col="y")
    iptw  = estimate_risks_rd(ad, a_col="a", y_col="y", w_col="iptw")
    iosw_iptw = estimate_risks_rd(ad, a_col="a", y_col="y", w_col="iosw_iptw")

    # concatenate into a single dataframe and label rows
    crude["method"] = "Crude"
    iptw["method"] = "IPTW"
    iosw_iptw["method"] = "IOSW x IPTW"
    res = pd.concat([crude, iptw, iosw_iptw])

    return(res)

def estimate_and_bootstrap(df, bootstrap_indices):
    """
    Estimate weights and risks/risk differences on original data and
    on each bootstrap sample defined by bootstrap_indices.
    Returns a DataFrame with results for original data and each bootstrap sample.
    """
    results = []

    # estimate on original data
    orig_res = estimate_weights_and_risks(df)
    orig_res["bootstrap"] = 0
    results.append(orig_res)

    # estimate on each bootstrap sample (contains IDS to sample)
    B = bootstrap_indices.shape[1]
    for b in range(B):
        indices = bootstrap_indices[:, b]
        boot_df = df.iloc[indices].reset_index(drop=True)
        boot_res = estimate_weights_and_risks(boot_df)
        boot_res["bootstrap"] = b + 1  # bootstrap samples indexed from 1
        results.append(boot_res)

    return pd.concat(results, ignore_index=True)

