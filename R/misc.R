comp.ase.uni <- function(phi, SIGMA, phi_true, SIGMA_true, P_max,
                         start = 0.001, end = 0.499, interval = 0.01){
  w <- seq(start, end, by = interval)
  n_t <- dim(phi)[2]
  TT <- (P_max+1):(n_t-P_max)
  true_sd <- compute_spec(phi = phi_true, SIGMA = SIGMA_true, w = w, P_max = P_max, ch1 = 1, ch2 = 1)
  est_sd <- compute_spec(phi = phi, SIGMA = SIGMA, w = w, P_max = P_max, ch1 = 1, ch2 = 1)
  ase <- mean((est_sd[[1]][TT, ] - true_sd[[1]][TT, ])^2)
  return(ase)
}


obtain_TVAR <- function(phi_fwd, phi_bwd, n_I, d){
  tmp <- run_whittle(phi_fwd = phi_fwd, phi_bwd = phi_bwd, n_I = n_I)
  return(tmp[[d]]$forward)
}

obtain_resid_var <- function(phi_chol_fwd, SIGMA, n_t, n_I, d){
  St <- array(NA, dim = c(n_I, n_I, n_t))
  upper_tri <- diag(n_I)
  for(i in 1:n_t){
    upper_tri[upper.tri(upper_tri)] <- phi_chol_fwd[i, , d]
    St[, , i] <- t(upper_tri) %*% diag(SIGMA[n_t-d, ]) %*% upper_tri
  }
  return(St)
}
