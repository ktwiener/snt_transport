
source("R/helpers.R")
sims = 100  # Number of simulations
B = 300    # Number of bootstrap samples
n = 5000  # Sample size
pl = 0.45 # Prevalence of outcome predictor L
pa = 0.5  # Probability of treatment (randomized)

py11 = 0.7  # P(Y^1=1|L=1) = 0.7
py10 = 0.15 # P(Y^1=1|L=0) = 0.15
py01 = 0.9  # P(Y^0=1|L=1) = 0.9
py00 = 0.05 # P(Y^0=1|L=0) = 0.05

all_sims <- vector("list", sims)
boot_inds <- vector("list", sims)
each_boot <- vector("list", B)

set.seed(24601)

for (k in seq(sims)){
  trial1 <- data.frame(
    sim = k,
    s = 0,
    trial = 1,
    id = seq(n),
    l  = rbinom(n, 1, pl),
    a = rbinom(n, 1, pa)
  )
  
  trial1 <- generate_outcome(trial1, py11, py10, py01, py00)
  
  trial2 <- trial1[trial1$y == 0, ]
  
  trial2$a <- rbinom(nrow(trial2), 1, 
                     with(trial2, a*1 + (1-a)*pa))
  trial2$trial <- 2
  
  trial2 <- generate_outcome(trial2, py11, py10, py01, py00)
  
  analysis_data <- rbind(trial1, trial2)
  analysis_data$s <- 1
  
  trial1$y <- NA
  trial1$a <- NA
  
  res <- rbind(analysis_data, trial1)
  all_sims[[k]] <- res
  
  # bootstrap indices
  inds <- matrix(sample(1:n, n*B, replace = TRUE),
                 nrow = n, ncol = B)
  boot_inds[[k]] <- inds
}

saveRDS(all_sims, "data/random/all_sims.rds")
saveRDS(boot_inds, "data/random/boot_inds.rds")
