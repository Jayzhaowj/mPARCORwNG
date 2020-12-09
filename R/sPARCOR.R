#####################################
####### run shrinkage PARCOR ########
#####################################

sPARCOR <- function(y,
                    d,
                    niter = 10000,
                    nburn = round(niter / 2),
                    nthin = 1,
                    learn_a_xi = TRUE,
                    learn_a_tau = TRUE,
                    a_xi = 0.1,
                    a_tau = 0.1,
                    learn_kappa2 = TRUE,
                    learn_lambda2 = TRUE,
                    kappa2 = 20,
                    lambda2 = 20,
                    hyperprior_param,
                    c_tuning_par_xi = 1,
                    c_tuning_par_tau = 1,
                    display_progress = TRUE,
                    ret_beta_nc = FALSE,
                    delta, S_0, sample_size = 500,
                    chains = 5, uncertainty = FALSE){
  K <- dim(y)[2]
  # y dimension: n_t * n_I
  result1 <- run_parcor_parallel(F1 = t(y), delta = delta, P = 1, S_0 = S_0, sample_size = sample_size,
                                 chains = chains, DIC = DIC, uncertainty = uncertainty)
  result2 <- shrinkTVP(yf = t(result1$F1_fwd), yb = t(result1$F1_bwd),
                       d = d, niter = niter, nburn = nburn, nthin = nthin,
                       learn_a_xi = learn_a_xi, learn_a_tau = learn_a_tau, a_xi = a_xi,
                       a_tau = a_tau, learn_kappa2 = learn_kappa2, learn_lambda2 = learn_lambda2,
                       kappa2 = kappa2, lambda2 = lambda2, hyperprior_param = hyperprior_param,
                       c_tuning_par_xi = c_tuning_par_xi, c_tuning_par_tau = c_tuning_par_tau,
                       display_progress = display_progress, ret_beta_nc = ret_beta_nc)
  ### extract forward part
  beta_tmp <- simplify2array(result2$beta$f)
  phi_fwd <- apply(beta_tmp, 1:3, mean)
  phi_fwd <- aperm(phi_fwd, perm = c(2, 1, 3))
  phi_fwd <- abind::abind(result1$phi_fwd, phi_fwd)
  ### extract backward part
  beta_tmp <- simplify2array(result2$beta$b)
  phi_bwd <- apply(beta_tmp, 1:3, mean)
  phi_bwd <- aperm(phi_bwd, perm = c(2, 1, 3))
  phi_bwd <- abind::abind(result1$phi_bwd, phi_bwd)

  ### extract forward SIGMA
  SIGMA_mean <- apply(simplify2array(result2$SIGMA$f), 1:2, mean)

  tmp <- PAR_to_AR_fun(phi_fwd = phi_fwd, phi_bwd = phi_bwd, n_I = K)
  ar <- tmp[[d]]$forward
  return(list(phi_fwd = phi_fwd, phi_bwd = phi_bwd, SIGMA = SIGMA_mean,
              ar = ar))

}
