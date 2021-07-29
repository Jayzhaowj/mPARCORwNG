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
                    S_0){
  K <- dim(y)[2]
  n_t <- dim(y)[1]

  # y dimension: n_t * n_I
  #result1 <- run_parcor_parallel(F1 = t(y), delta = delta, P = 1, S_0 = S_0, sample_size = sample_size,
  #                               chains = chains, DIC = DIC, uncertainty = uncertainty)
  result <- shrinkTVP(y = t(y), d = d, S_0 = S_0, niter = niter, nburn = nburn, nthin = nthin,
                       learn_a_xi = learn_a_xi, learn_a_tau = learn_a_tau, a_xi = a_xi,
                       a_tau = a_tau, learn_kappa2 = learn_kappa2, learn_lambda2 = learn_lambda2,
                       kappa2 = kappa2, lambda2 = lambda2, hyperprior_param = hyperprior_param,
                       c_tuning_par_xi = c_tuning_par_xi, c_tuning_par_tau = c_tuning_par_tau,
                       display_progress = display_progress, ret_beta_nc = ret_beta_nc)
  if(uncertainty){
    phi_fwd <- array(0, dim = c(n_t, K^2, 1))
    phi_bwd <- array(0, dim = c(n_t, K^2, 1))
    for(i in 1:((niter - nburn)/nthin)){
      result$beta$f[[i]] <- aperm(result2$beta$f[[i]], perm = c(2,1,3))
      result$beta$b[[i]] <- aperm(result2$beta$b[[i]], perm = c(2,1,3))
    }


    # result2$beta$f[[i]] <- abind::abind(result1$phi_fwd, aperm(result2$beta$f[[i]], perm = c(2,1,3)))
    # result2$beta$f[[i]] <- aperm(result2$beta$f[[i]], perm = c(2,1,3))
    # #browser()
    # result2$beta$b[[i]] <- abind::abind(result1$phi_bwd, aperm(result2$beta$b[[i]], perm = c(2,1,3)))
    # result2$beta$b[[i]] <- aperm(result2$beta$b[[i]], perm = c(2,1,3))
    return(list(phi_fwd = result$beta$f,
                phi_bwd = result$beta$b,
                SIGMA = result$SIGMA$f))
  }else{
    ### extract forward part
    beta_tmp <- simplify2array(result$beta$f)
    phi_fwd <- apply(beta_tmp, 1:3, mean)
    phi_fwd <- aperm(phi_fwd, perm = c(2, 1, 3))

    ### extract backward part
    beta_tmp <- simplify2array(result$beta$b)
    phi_bwd <- apply(beta_tmp, 1:3, mean)
    phi_bwd <- aperm(phi_bwd, perm = c(2, 1, 3))



    ### extract forward SIGMA
    SIGMA <- apply(simplify2array(result$SIGMA$f), 1:3, mean)

    ### transfer PARCOR coefficients to AR coefficients
    tmp <- PAR_to_AR_fun(phi_fwd = phi_fwd, phi_bwd = phi_bwd, n_I = K)
    ar <- tmp[[d]]$forward
    return(list(phi_fwd = phi_fwd, phi_bwd = phi_bwd, SIGMA = SIGMA,
                ar = ar))
  }

}
