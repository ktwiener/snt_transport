library(dplyr)

pl = 0.45 # Prevalence of outcome predictor L
pa = 0.5  # Probability of treatment (randomized)

py11 = 0.7  # P(Y^1=1|L=1) = 0.7
py10 = 0.15 # P(Y^1=1|L=0) = 0.15
py01 = 0.9  # P(Y^0=1|L=1) = 0.9
py00 = 0.05 # P(Y^0=1|L=0) = 0.05

trial1_y1 = pl*py11 + (1-pl)*py10
trial1_y0 = pl*py01 + (1-pl)*py00
truth = trial1_y1 - trial1_y0

mest <- read.csv("data/results/mest_results.csv")

boot <- read.csv("data/results/boot_results.csv")

mest |>
  dplyr::filter(Param %in% c("delta", "delta_ipt", "mu_1", "mu_0", "ipt_mu_1", "ipt_mu_0")) |>
  dplyr::mutate(
    true_val = case_when(
      Param %in% c("mu_1", "ipt_mu_1") ~ trial1_y1,
      Param %in% c("mu_0", "ipt_mu_0") ~ trial1_y0,
      TRUE ~ truth
    ),
    in_ci = LCL <= true_val & true_val <= UCL
  ) |> 
  dplyr::group_by(Param) |>
  dplyr::summarize(
    n = dplyr::n(),
    exp_val = mean(Coef),
    bias = mean(Coef - true_val),
    ase = sqrt(mean(SE^2)),
    cov = mean(in_ci),
    ese = sd(Coef),
    rmse = sqrt(mean((Coef-true_val)^2))
  ) |>
  dplyr::filter(Param %in% c("delta", "delta_ipt"))-> mest_res

boot |>
  dplyr::group_by(method) |>
  dplyr::summarize(
    n = dplyr::n(),
    exp_rd = mean(rd),
    bias = exp_rd - truth,
    exp_risk0 = mean(risk0),
    exp_risk1 = mean(risk1),
    ese = sd(rd),
    ase = sqrt(mean(bootstrap_se^2)),
    cov = mean((rd - 1.96*bootstrap_se) <= truth & truth <= rd + 1.96*bootstrap_se),
    rmse = sqrt(mean((rd-truth)^2))
  ) -> boot_res

