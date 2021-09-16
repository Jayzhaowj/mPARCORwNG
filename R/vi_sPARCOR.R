vi_sPARCOR <- function(y, d, d1, d2, e1, e2, a_xi, a_tau, learn_a_xi, learn_a_tau, iter_max,
                       ind, S_0, epsilon, skip, delta){
  K <- dim(y)[2]
  n_t <- dim(y)[1]

  if(skip){
    ## skip the first stage by using PARCOR model
    phi_fwd <- array(0, dim = c(n_t, K^2))
    phi_bwd <- array(0, dim = c(n_t, K^2))
    if(missing(delta)){
      ### Set up discount ###
      grid_seq <- seq(0.95, 1, 0.01)
      tmp <- as.matrix(expand.grid(grid_seq, grid_seq))
      tmp_dim <- dim(tmp)
      delta <- array(dim = c(tmp_dim[1], K^2, 1))
    }
    result_skip <- run_parcor(F1 = t(y), delta = delta, P = 1, S_0 = S_0*diag(K),
                              DIC = FALSE, uncertainty = FALSE)

    result <- vi_shrinkTVP(y_fwd = t(result_skip$F1_fwd), y_bwd = t(result_skip$F1_bwd),
                           d = d, d1 = d1, d2 = d2, e1 = e1, e2 = e2,
                           a_xi = a_xi, a_tau = a_tau, learn_a_xi = learn_a_xi, learn_a_tau = learn_a_tau,
                           iter_max = iter_max, ind = ind, S_0 = S_0, epsilon = epsilon, skip = skip)
    result$beta$f[, , 1] <- t(result_skip$phi_fwd[, , 1])
    result$beta$b[, , 1] <- t(result_skip$phi_bwd[, , 1])
  }else{
    result <- vi_shrinkTVP(y_fwd = y, y_bwd = y, d = d, d1 = d1, d2 = d2, e1 = e1, e2 = e2,
                           a_xi = a_xi, a_tau = a_tau, learn_a_xi = learn_a_xi, learn_a_tau = learn_a_tau,
                           iter_max = iter_max, ind = ind, S_0 = S_0, epsilon = epsilon, skip = skip)
  }
  return(result)
}

