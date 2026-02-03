generate_outcome <- function(df, py11, py10, py01, py00){
  n <- nrow(df)
  
  df$ya1 <- with(df, rbinom(n, 1, l*py11 + (1-l)*py10))
  df$ya0 <- with(df, rbinom(n, 1, l*py01 + (1-l)*py00))
  df$y <- with(df, a*ya1 + (1-a)*ya0)
  
  return(df)
}

