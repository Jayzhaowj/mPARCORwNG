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
                    S_0, delta, uncertainty=FALSE, ind = TRUE, skip = TRUE,
                    cpus = 1){
  K <- dim(y)[2]
  n_t <- dim(y)[1]

  if(skip){
    ## skip the first stage,

    phi_fwd <- array(0, dim = c(n_t, K^2))
    phi_bwd <- array(0, dim = c(n_t, K^2))

    if(missing(delta)){
      ### Set up discount ###
      grid_seq <- seq(0.95, 1, 0.01)
      tmp <- as.matrix(expand.grid(grid_seq, grid_seq))
      tmp_dim <- dim(tmp)
      delta <- array(dim = c(tmp_dim[1], K^2, 1))
    }
    result_skip <- run_parcor_parallel(F1 = t(y), delta = delta, P = 1, S_0 = S_0*diag(K),
                                       sample_size = (niter-nburn)/nthin, DIC = FALSE, uncertainty = TRUE)
    result <- shrinkTVP(y_fwd = t(result_skip$F1_fwd), y_bwd = t(result_skip$F1_bwd),
                        d = d, S_0 = S_0, niter = niter, nburn = nburn, nthin = nthin,
                        learn_a_xi = learn_a_xi, learn_a_tau = learn_a_tau, a_xi = a_xi,
                        a_tau = a_tau, learn_kappa2 = learn_kappa2, learn_lambda2 = learn_lambda2,
                        kappa2 = kappa2, lambda2 = lambda2, hyperprior_param = hyperprior_param,
                        c_tuning_par_xi = c_tuning_par_xi, c_tuning_par_tau = c_tuning_par_tau,
                        display_progress = display_progress, ret_beta_nc = ret_beta_nc, ind = ind, skip = skip)
    for(i in 1:((niter - nburn)/nthin)){
      for(j in (1+1):(n_t-1)){
        phi_fwd[j, ] <- tryCatch(rmvn(n = 1, mu = as.vector(result_skip$phi_fwd[, j, 1]),
                             sigma = result_skip$Cnt_fwd[[1]][[j]]), error = function(e){as.vector(result_skip$phi_fwd[,j,1])})
        phi_bwd[j, ] <- tryCatch(rmvn(n = 1, mu = as.vector(result_skip$phi_bwd[, j, 1]),
                             sigma = result_skip$Cnt_bwd[[1]][[j]]), error = function(e){as.vector(result_skip$phi_bwd[,j,1])})
      }
      result$beta$f[[i]][, , 1] <- phi_fwd
      result$beta$b[[i]][, , 1] <- phi_bwd

      result$beta$f[[i]] <- aperm(result$beta$f[[i]], perm = c(2,1,3))
      result$beta$b[[i]] <- aperm(result$beta$b[[i]], perm = c(2,1,3))
    }
  }else{
    result <- shrinkTVP(y_fwd = y, y_bwd = y,
                        d = d, S_0 = S_0, niter = niter, nburn = nburn, nthin = nthin,
                        learn_a_xi = learn_a_xi, learn_a_tau = learn_a_tau, a_xi = a_xi,
                        a_tau = a_tau, learn_kappa2 = learn_kappa2, learn_lambda2 = learn_lambda2,
                        kappa2 = kappa2, lambda2 = lambda2, hyperprior_param = hyperprior_param,
                        c_tuning_par_xi = c_tuning_par_xi, c_tuning_par_tau = c_tuning_par_tau,
                        display_progress = display_progress, ret_beta_nc = ret_beta_nc, ind = ind, skip = skip)

    for(i in 1:((niter - nburn)/nthin)){
      result$beta$f[[i]] <- aperm(result$beta$f[[i]], perm = c(2,1,3))
      result$beta$b[[i]] <- aperm(result$beta$b[[i]], perm = c(2,1,3))
    }
  }

  ### transform PARCOR coefficients to TVVAR coefficients
  phi_fwd <- result$beta$f
  phi_bwd <- result$beta$b
  sfInit(parallel = TRUE, cpus = cpus, type = "SOCK")
  sfLibrary(PARCOR)
  sfLibrary(mPARCORwNG)
  sfExport("phi_fwd", "phi_bwd", "K", "d")
  ar_coef_sample <- sfLapply(1:((niter-nburn)/nthin), function(i) obtain_TVAR(result$beta$f[[i]], result$beta$b[[i]], K, d))
  sfStop()
  if(uncertainty){
    return(list(phi_fwd = result$beta$f,
                phi_bwd = result$beta$b,
                chol_fwd = result$beta_chol$f,
                chol_bwd = result$beta_chol$b,
                ar = ar_coef_sample,
                SIGMA = result$SIGMA$f))
  }else{
    ### extract forward part
    phi_fwd <- apply(simplify2array(result$beta$f), 1:3, mean)
    phi_fwd <- aperm(phi_fwd, perm = c(2, 1, 3))
    if(!ind){
      beta_chol_fwd <- apply(simplify2array(result$beta_chol$f), 1:3, mean)
    }

    ### extract backward part
    phi_bwd <- apply(simplify2array(result$beta$b), 1:3, mean)
    phi_bwd <- aperm(phi_bwd, perm = c(2, 1, 3))
    if(!ind){
      beta_chol_bwd <- apply(simplify2array(result$beta_chol$b), 1:3, mean)
    }

    ### extract forward SIGMA
    SIGMA <- apply(simplify2array(result$SIGMA$f), 1:3, mean)

    ### transfer PARCOR coefficients to AR coefficients
    ar <- apply(simplify2array(ar_coef_sample), 1:3, mean)
    return(list(phi_fwd = phi_fwd,
                phi_bwd = phi_bwd,
                phi_chol_fwd = beta_chol_fwd,
                phi_chol_bwd = beta_chol_bwd,
                SIGMA = SIGMA,
                ar = ar))
  }
}
