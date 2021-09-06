// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <stochvol.h>
#include <progress.hpp>
#include <math.h>
#include "ffbs.h"
#include "DG_vi_update_functions.h"
#include "common_update_functions.h"
#include "sample_parameters.h"
using namespace Rcpp;

// [[Rcpp::export]]
List vi_shrinkTVP(arma::mat y_fwd,
                  arma::mat y_bwd,
                  int d,
                  double d1,
                  double d2,
                  double e1,
                  double e2,
                  double a_xi,
                  double a_tau,
                  bool learn_a_xi,
                  bool learn_a_tau,
                  int iter_max,
                  bool ind,
                  double S_0,
                  double epsilon,
                  bool skip) {

  // Progress bar setup
  arma::vec prog_rep_points = arma::round(arma::linspace(0, iter_max, 50));
  Progress p(50, true);

  // Some necessary dimensions
  int N = y_fwd.n_rows;
  int n_I = y_fwd.n_cols;
  int n_I2 = std::pow(n_I, 2);


  // Some index
  //int m = 1; // current stage
  int N_m;
  int n_1;     // index
  int n_T;     // index

  int start;
  if(!skip){
    start = 1;
  }else{
    start = 2;
  }

  // generate forward and backward prediction error
  arma::cube yf(N, n_I, d+1, arma::fill::none);
  arma::cube yb(N, n_I, d+1, arma::fill::none);
  yf.slice(start - 1) = y_fwd;
  yb.slice(start - 1) = y_bwd;

  arma::vec y_tmp;
  arma::mat x_tmp;
  int d_tmp;

  // generate forward and backward PARCOR
  arma::cube betaf_nc_old(N, n_I2, d, arma::fill::none);
  arma::cube betab_nc_old(N, n_I2, d, arma::fill::none);
  arma::cube betaf_nc_new(N, n_I2, d, arma::fill::none);
  arma::cube betab_nc_new(N, n_I2, d, arma::fill::none);

  arma::cube betaf(N, n_I2, d, arma::fill::none);
  arma::cube betab(N, n_I2, d, arma::fill::none);

  arma::cube sigma2f_old(N, n_I, d, arma::fill::none);
  arma::cube sigma2b_old(N, n_I, d, arma::fill::none);

  arma::cube sigma2f_new(N, n_I, d, arma::fill::none);
  arma::cube sigma2b_new(N, n_I, d, arma::fill::none);

  //arma::mat sigma2f_inv_old(N, d, arma::fill::none);
  //arma::mat sigma2b_inv_old(N, d, arma::fill::none);

  //arma::mat sigma2f_inv_new(N, d, arma::fill::none);
  //arma::mat sigma2b_inv_new(N, d, arma::fill::none);

  arma::cube thetaf_sr_old(n_I, n_I, d, arma::fill::ones);
  arma::cube thetab_sr_old(n_I, n_I, d, arma::fill::ones);

  arma::cube thetaf_sr_new(n_I, n_I, d, arma::fill::ones);
  arma::cube thetab_sr_new(n_I, n_I, d, arma::fill::ones);

  //arma::cube thetaf_old(n_I, n_I, d, arma::fill::none);
  //arma::cube thetab_old(n_I, n_I, d, arma::fill::none);

  //arma::cube thetaf_new(n_I, n_I, d, arma::fill::none);
  //arma::cube thetab_new(n_I, n_I, d, arma::fill::none);

  arma::cube betaf_mean_old(n_I, n_I, d, arma::fill::zeros);
  arma::cube betab_mean_old(n_I, n_I, d, arma::fill::zeros);

  arma::cube betaf_mean_new(n_I, n_I, d, arma::fill::zeros);
  arma::cube betab_mean_new(n_I, n_I, d, arma::fill::zeros);


  //arma::mat beta2f_mean_old(n_I2, d, arma::fill::none);
  //arma::mat beta2b_mean_old(n_I2, d, arma::fill::none);

  //arma::mat beta2f_mean_new(n_I2, d, arma::fill::none);
  //arma::mat beta2b_mean_new(n_I2, d, arma::fill::none);


  arma::cube xi2f_old(n_I, n_I, d, arma::fill::ones);
  arma::cube xi2b_old(n_I, n_I, d, arma::fill::ones);

  arma::cube xi2f_new(n_I, n_I, d, arma::fill::ones);
  arma::cube xi2b_new(n_I, n_I, d, arma::fill::ones);

  arma::cube xi2f_inv_old(n_I, n_I, d, arma::fill::ones);
  arma::cube xi2b_inv_old(n_I, n_I, d, arma::fill::ones);

  arma::cube xi2f_inv_new(n_I, n_I, d, arma::fill::ones);
  arma::cube xi2b_inv_new(n_I, n_I, d, arma::fill::ones);

  arma::cube tau2f_old(n_I, n_I, d, arma::fill::ones);
  arma::cube tau2b_old(n_I, n_I, d, arma::fill::ones);

  arma::cube tau2f_new(n_I, n_I, d, arma::fill::ones);
  arma::cube tau2b_new(n_I, n_I, d, arma::fill::ones);

  arma::cube tau2f_inv_old(n_I, n_I, d, arma::fill::ones);
  arma::cube tau2b_inv_old(n_I, n_I, d, arma::fill::ones);

  arma::cube tau2f_inv_new(n_I, n_I, d, arma::fill::ones);
  arma::cube tau2b_inv_new(n_I, n_I, d, arma::fill::ones);

  arma::vec kappa2f_old(n_I, arma::fill::ones);
  arma::vec kappa2b_old(n_I, arma::fill::ones);
  arma::vec lambda2f_old(n_I, arma::fill::ones);
  arma::vec lambda2b_old(n_I, arma::fill::ones);

  arma::vec kappa2f_new(n_I, arma::fill::ones);
  arma::vec kappa2b_new(n_I, arma::fill::ones);
  arma::vec lambda2f_new(n_I, arma::fill::ones);
  arma::vec lambda2b_new(n_I, arma::fill::ones);

  arma::vec a_xif_new(n_I);
  arma::vec a_tauf_new(n_I);

  arma::vec a_xib_new(n_I);
  arma::vec a_taub_new(n_I);

  arma::vec a_xif_old(n_I);
  arma::vec a_tauf_old(n_I);

  arma::vec a_xib_old(n_I);
  arma::vec a_taub_old(n_I);


  arma::mat beta_nc_tmp;
  arma::mat beta2_nc_tmp;
  arma::cube beta_cov_nc_tmp;

  arma::vec theta_sr_tmp;
  arma::vec theta_tmp;

  arma::vec beta_mean_tmp;
  arma::vec beta2_mean_tmp;

  arma::vec tau2_tmp;
  arma::vec tau2_inv_tmp;
  arma::vec xi2_tmp;
  arma::vec xi2_inv_tmp;

  arma::vec sigma2_tmp;
  arma::vec sigma2_inv_tmp;


  int index;
  //arma::mat C0f_save;
  //arma::mat svf_mu_save;
  //arma::mat svf_phi_save;
  //arma::mat svf_sigma2_save;

  //arma::mat C0b_save;
  //arma::mat svb_mu_save;
  //arma::mat svb_phi_save;
  //arma::mat svb_sigma2_save;

  //if (sv == false){
  //  C0f_save = arma::mat(d, nsave, arma::fill::none);
  //  C0b_save = arma::mat(d, nsave, arma::fill::none);
  //} else {
  //  svf_mu_save = arma::mat(nsave, d, arma::fill::none);
  //  svf_phi_save = arma::mat(nsave, d, arma::fill::none);
  //  svf_sigma2_save = arma::mat(nsave, d, arma::fill::none);

  //  svb_mu_save = arma::mat(nsave, d, arma::fill::none);
  //  svb_phi_save = arma::mat(nsave, d, arma::fill::none);
  //  svb_sigma2_save = arma::mat(nsave, d, arma::fill::none);
  //}

  // Initial values and objects
  sigma2f_new.fill(1.0);
  sigma2b_new.fill(1.0);
  sigma2f_old.fill(1.0);
  sigma2b_old.fill(1.0);

  //sigma2f_inv_new.fill(1.0);
  //sigma2b_inv_new.fill(1.0);
  //sigma2f_inv_old.fill(1.0);
  //sigma2b_inv_old.fill(1.0);

  thetaf_sr_new.fill(1.0);
  thetab_sr_new.fill(1.0);
  thetaf_sr_old.fill(1.0);
  thetab_sr_old.fill(1.0);

  //thetaf_new.fill(1.0);
  //thetab_new.fill(1.0);
  //thetaf_old.fill(1.0);
  //thetab_old.fill(1.0);

  betaf_mean_new.fill(1.0);
  betab_mean_new.fill(1.0);
  betaf_mean_old.fill(1.0);
  betab_mean_old.fill(1.0);

  //beta2f_mean_new.fill(1.0);
  //beta2b_mean_new.fill(1.0);
  //beta2f_mean_old.fill(1.0);
  //beta2b_mean_old.fill(1.0);

  xi2f_new.fill(1.0);
  xi2b_new.fill(1.0);
  xi2f_old.fill(1.0);
  xi2b_old.fill(1.0);

  //xi2f_inv_new.fill(1.0);
  //xi2b_inv_new.fill(1.0);
  //xi2f_inv_old.fill(1.0);
  //xi2b_inv_old.fill(1.0);

  tau2f_new.fill(1.0);
  tau2b_new.fill(1.0);
  tau2f_old.fill(1.0);
  tau2b_old.fill(1.0);



  if (!learn_a_xi){
    a_xif_new.fill(a_xi);
    a_xib_new.fill(a_xi);
    a_xif_old.fill(a_xi);
    a_xib_old.fill(a_xi);
  }
  if (!learn_a_tau){
    a_tauf_new.fill(a_tau);
    a_taub_new.fill(a_tau);
    a_tauf_old.fill(a_tau);
    a_taub_old.fill(a_tau);
  }
  // definition of beta_tilde cholesky
  arma::cube betaf_nc_chol_old;
  arma::cube betab_nc_chol_old;
  arma::cube betaf_nc_chol_new;
  arma::cube betab_nc_chol_new;

  // definition of non central part cholesky
  arma::cube betaf_chol;
  arma::cube betab_chol;

  // definition of theta sr cholesky
  arma::mat thetaf_sr_chol_old;
  arma::mat thetab_sr_chol_old;

  arma::mat thetaf_sr_chol_new;
  arma::mat thetab_sr_chol_new;


  // definition of beta mean
  arma::mat betaf_mean_chol_old;
  arma::mat betab_mean_chol_old;

  arma::mat betaf_mean_chol_new;
  arma::mat betab_mean_chol_new;

  // definition of xi2 cholesky
  arma::mat xi2f_chol_old;
  arma::mat xi2b_chol_old;

  arma::mat xi2f_chol_new;
  arma::mat xi2b_chol_new;

  arma::mat xi2f_inv_chol_old;
  arma::mat xi2b_inv_chol_old;

  arma::mat xi2f_inv_chol_new;
  arma::mat xi2b_inv_chol_new;

  // definition of tau2 cholesky
  arma::mat tau2f_chol_old;
  arma::mat tau2b_chol_old;

  arma::mat tau2f_chol_new;
  arma::mat tau2b_chol_new;

  arma::mat tau2f_inv_chol_old;
  arma::mat tau2b_inv_chol_old;

  arma::mat tau2f_inv_chol_new;
  arma::mat tau2b_inv_chol_new;

  // definition temp lower triangular
  arma::mat tmp_lower_triangular(n_I, n_I, arma::fill::eye);
  arma::mat tmp_beta(n_I, n_I);
  arma::uvec lower_indices = arma::trimatl_ind(size(tmp_lower_triangular), -1);
  arma::uvec all_indices = arma::linspace<arma::uvec>(0, n_I*n_I-1, n_I*n_I);

  if(!ind){
    // definition of beta_tilde cholesky
    betaf_nc_chol_old = arma::cube(N, n_I*(n_I-1)/2, d, arma::fill::none);
    betab_nc_chol_old = arma::cube(N, n_I*(n_I-1)/2, d, arma::fill::none);
    betaf_nc_chol_new = arma::cube(N, n_I*(n_I-1)/2, d, arma::fill::none);
    betab_nc_chol_new = arma::cube(N, n_I*(n_I-1)/2, d, arma::fill::none);

    // definition of non central part cholesky
    betaf_chol = arma::cube(N, n_I*(n_I-1)/2, d, arma::fill::none);
    betab_chol = arma::cube(N, n_I*(n_I-1)/2, d, arma::fill::none);

    // definition of theta sr cholesky
    thetaf_sr_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    thetab_sr_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);

    thetaf_sr_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    thetab_sr_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);


    // definition of beta mean
    betaf_mean_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::zeros);
    betab_mean_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::zeros);

    betaf_mean_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::zeros);
    betab_mean_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::zeros);

    // definition of xi2 cholesky
    xi2f_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    xi2b_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);

    xi2f_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    xi2b_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);

    xi2f_inv_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    xi2b_inv_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);

    xi2f_inv_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    xi2b_inv_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);

    // definition of tau2 cholesky
    tau2f_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    tau2b_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);

    tau2f_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    tau2b_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);

    tau2f_inv_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    tau2b_inv_chol_old = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);

    tau2f_inv_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
    tau2b_inv_chol_new = arma::mat(n_I*(n_I-1)/2, d, arma::fill::ones);
  }

  // Values to check if the sampler failed or not
  bool succesful = true;
  std::string fail;
  int fail_iter;
  int j = 0;

  // Introduce difference
  double diff = 100.0;
  // Begin Gibbs loop
  while( !( diff < epsilon ) && (j < iter_max)){
    diff = 0.0;
    for(int m = start; m < d+1; m++){
      for(int k = 0; k < n_I; k++){
        // Forward
        // ----------------------------
        n_1 = m + 1;
        n_T = N;
        N_m = n_T - n_1 + 1;
        y_tmp = yf.slice(m-1).col(k).rows(n_1-1, n_T-1);
        x_tmp = yb.slice(m-1).rows(n_1-m-1, n_T-m-1);
        if(!ind){
          if(k == 1){
            x_tmp = arma::join_rows(x_tmp, -yf.slice(m-1).col(0).rows(n_1-1, n_T-1));
          }else if(k > 1){
            x_tmp = arma::join_rows(x_tmp, -yf.slice(m-1).cols(0, k-1).rows(n_1-1, n_T-1));
          }

        }
        //if(k!=0){
        //  x_tmp = arma::join_rows(x_tmp, yf.slice(m-1).cols(k+1, n_I-1).rows(n_1-1, n_T-1));
        //}
        d_tmp = x_tmp.n_cols;
        beta_nc_tmp = arma::mat(N_m+1, d_tmp, arma::fill::zeros);
        beta2_nc_tmp = arma::mat(N_m+1, d_tmp, arma::fill::zeros);
        beta_cov_nc_tmp = arma::cube(N_m+1, d_tmp, d_tmp, arma::fill::zeros);

        theta_sr_tmp = thetaf_sr_old.slice(m-1).col(k);
        beta_mean_tmp = betaf_mean_old.slice(m-1).col(k);

        theta_tmp = arma::vec(d_tmp, arma::fill::zeros);
        beta2_mean_tmp = arma::vec(d_tmp, arma::fill::zeros);

        tau2_tmp = arma::vec(d_tmp, arma::fill::zeros);
        xi2_tmp = arma::vec(d_tmp, arma::fill::zeros);

        tau2_inv_tmp = tau2f_inv_old.slice(m-1).col(k);
        xi2_inv_tmp = xi2f_inv_old.slice(m-1).col(k);
        sigma2_tmp = sigma2f_old.slice(m-1).col(k).rows(n_1-1, n_T-1);

        if(!ind){
          if( k == 1){
            theta_sr_tmp = arma::vec(d_tmp, arma::fill::zeros);
            beta_mean_tmp = arma::vec(d_tmp, arma::fill::zeros);
            tau2_inv_tmp = arma::vec(d_tmp, arma::fill::zeros);
            xi2_inv_tmp = arma::vec(d_tmp, arma::fill::zeros);

            theta_sr_tmp(arma::span(0, d_tmp-2)) = thetaf_sr_old.slice(m-1).col(k);
            theta_sr_tmp(d_tmp-1) = thetaf_sr_chol_old(0, m-1);

            beta_mean_tmp(arma::span(0, d_tmp-2)) = betaf_mean_old.slice(m-1).col(k);
            beta_mean_tmp(d_tmp-1) = betaf_mean_chol_old(0, m-1);

            tau2_inv_tmp(arma::span(0, d_tmp-2)) = tau2f_inv_old.slice(m-1).col(k);
            tau2_inv_tmp(d_tmp-1) = tau2f_inv_chol_old(0, m-1);

            xi2_inv_tmp(arma::span(0, d_tmp-2)) = xi2f_inv_old.slice(m-1).col(k);
            xi2_inv_tmp(d_tmp-1) = xi2f_inv_chol_old(0, m-1);
          }else if(k > 1){
            index = arma::sum(arma::linspace(1, k-1, k-1));
            theta_sr_tmp = arma::join_cols(theta_sr_tmp, thetaf_sr_chol_old.col(m-1).rows(index, index+k-1));
            beta_mean_tmp = arma::join_cols(beta_mean_tmp, betaf_mean_chol_old.col(m-1).rows(index, index+k-1));
            tau2_inv_tmp = arma::join_cols(tau2_inv_tmp, tau2f_inv_chol_old.col(m-1).rows(index, index+k-1));
            xi2_inv_tmp = arma::join_cols(xi2_inv_tmp, xi2f_inv_chol_old.col(m-1).rows(index, index+k-1));
          }
        }

        try {
          update_beta_tilde(beta_nc_tmp, beta2_nc_tmp, beta_cov_nc_tmp,
                            y_tmp, x_tmp, theta_sr_tmp, beta_mean_tmp, N_m, S_0, sigma2_tmp);

          betaf_nc_new.slice(m-1).rows(n_1-1, n_T-1).cols(k*n_I, (k+1)*n_I-1) = beta_nc_tmp.cols(0, n_I-1).rows(1, N_m);
          if(!ind){
            if(k == 1){
              betaf_nc_chol_new.slice(m-1).rows(n_1-1, n_T-1).col(0) = beta_nc_tmp.col(n_I).rows(1, N_m);
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              betaf_nc_chol_new.slice(m-1).rows(n_1-1, n_T-1).cols(index, index+k-1) = beta_nc_tmp.cols(n_I, d_tmp-1).rows(1, N_m);
            }
          }


          //(beta2f_nc_new.slice(m-1)).rows(n_1-1, n_T-1) = beta2_nc_tmp.rows(1, N_m);
          sigma2f_new.slice(m-1).col(k).rows(n_1-1, n_T-1) = sigma2_tmp;
          yf.slice(m).col(k).rows(n_1-1, n_T-1) = y_tmp;
          y_tmp = yf.slice(m-1).col(k).rows(n_1-1, n_T-1);
        } catch (...){
          //beta_nc_tmp.fill(arma::datum::nan);
          if (succesful == true){
            fail = "update forward beta_nc";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // update beta mean
        try {
          update_beta_mean(beta_mean_tmp, beta2_mean_tmp, theta_sr_tmp,
                           y_tmp, x_tmp, beta_nc_tmp.rows(1, N_m), 1.0/sigma2_tmp, tau2_inv_tmp);
          betaf_mean_new.slice(m-1).col(k) = beta_mean_tmp(arma::span(0, n_I-1));
          if(!ind){
            if(k == 1){
              betaf_mean_chol_new(0, m-1) = beta_mean_tmp(n_I);
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              betaf_mean_chol_new.col(m-1).rows(index, index+k-1) = beta_mean_tmp(arma::span(n_I, d_tmp-1));
            }
          }

        } catch(...){
          //beta_mean_tmp.fill(nanl(""));
          //beta2_mean_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update forward beta mean & beta mean square";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // update theta sr
        try{
          update_theta_sr(beta_mean_tmp, theta_sr_tmp, theta_tmp, y_tmp, x_tmp,
                          beta_nc_tmp.rows(1, N_m), beta2_nc_tmp.rows(1, N_m),
                          beta_cov_nc_tmp.rows(1, N_m), 1.0/sigma2_tmp, xi2_inv_tmp);
          thetaf_sr_new.slice(m-1).col(k) = theta_sr_tmp(arma::span(0, n_I-1));
          if(!ind){
            if(k == 1){
              thetaf_sr_chol_new(0, m-1) = theta_sr_tmp(n_I);
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              thetaf_sr_chol_new.col(m-1).rows(index, index+k-1) = theta_sr_tmp(arma::span(n_I, d_tmp-1));
            }
          }
          //thetaf_new.slice(m-1).col(k) = theta_tmp;
        } catch(...){
          //theta_sr_tmp.fill(nanl(""));
          //theta_tmp.fill(nanl(""));
          if(succesful == true){
            fail = "update forward theta sr & theta";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // update forward tau2
        //Rcout << "forward tau2" << "\n";
        try {
          update_local_shrink(tau2_tmp, tau2_inv_tmp, beta2_mean_tmp, lambda2f_old(k), a_tauf_old(k));
          tau2f_new.slice(m-1).col(k) = tau2_tmp(arma::span(0, n_I-1));
          tau2f_inv_new.slice(m-1).col(k) = tau2_inv_tmp(arma::span(0, n_I-1));
          if(!ind){
            if(k == 1){
              tau2f_chol_new(0, m-1) = arma::as_scalar(tau2_tmp(n_I));
              tau2f_inv_chol_new(0, m-1) = arma::as_scalar(tau2_inv_tmp(n_I));
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              tau2f_chol_new.col(m-1).rows(index, index+k-1) = tau2_tmp(arma::span(n_I, d_tmp-1));
              tau2f_inv_chol_new.col(m-1).rows(index, index+k-1) = tau2_inv_tmp(arma::span(n_I, d_tmp-1));
            }
          }
        } catch(...) {
          //tau2_tmp.fill(nanl(""));
          //tau2_inv_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update forward tau2 & tau2_inv";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // update forward xi2
        //Rcout << "forward xi" << "\n";
        try {
          update_local_shrink(xi2_tmp, xi2_inv_tmp, theta_tmp, kappa2f_old(k), a_xif_old(k));
          xi2f_new.slice(m-1).col(k) = xi2_tmp(arma::span(0, n_I-1));
          xi2f_inv_new.slice(m-1).col(k) = xi2_inv_tmp(arma::span(0, n_I-1));
          if(!ind){
            if(k == 1){
              xi2f_chol_new(0, m-1) = arma::as_scalar(xi2_tmp(n_I));
              xi2f_inv_chol_new(0, m-1) = arma::as_scalar(xi2_inv_tmp(n_I));
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              xi2f_chol_new.col(m-1).rows(index, index+k-1) = xi2_tmp(arma::span(n_I, d_tmp-1));
              xi2f_inv_chol_new.col(m-1).rows(index, index+k-1) = xi2_inv_tmp(arma::span(n_I, d_tmp-1));
            }
          }
        } catch(...) {
          //xi2_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update forward xi2 & xi2_inv";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // Backward
        // --------------------------------
        n_1 = 1;          // backward index
        n_T = N - m;      // backward index
        N_m = n_T - n_1 + 1;

        y_tmp = yb.slice(m-1).col(k).rows(n_1-1, n_T-1);
        x_tmp = yf.slice(m-1).rows(n_1+m-1, n_T+m-1);

        if(!ind){
          if(k == 1){
            x_tmp = arma::join_rows(x_tmp, -yb.slice(m-1).col(0).rows(n_1-1, n_T-1));
          }else if(k > 1){
            x_tmp = arma::join_rows(x_tmp, -yb.slice(m-1).cols(0, k-1).rows(n_1-1, n_T-1));
          }
        }

        d_tmp = x_tmp.n_cols;
        beta_nc_tmp = arma::mat(N_m+1, d_tmp, arma::fill::zeros);
        beta2_nc_tmp = arma::mat(N_m+1, d_tmp, arma::fill::zeros);
        beta_cov_nc_tmp = arma::cube(N_m+1, d_tmp, d_tmp, arma::fill::zeros);

        theta_sr_tmp = thetab_sr_old.slice(m-1).col(k);
        beta_mean_tmp = betab_mean_old.slice(m-1).col(k);

        theta_tmp = arma::vec(d_tmp, arma::fill::zeros);
        beta2_mean_tmp = arma::vec(d_tmp, arma::fill::zeros);

        tau2_tmp = arma::vec(d_tmp, arma::fill::zeros);
        xi2_tmp = arma::vec(d_tmp, arma::fill::zeros);

        tau2_inv_tmp = tau2b_inv_old.slice(m-1).col(k);
        xi2_inv_tmp = xi2b_inv_old.slice(m-1).col(k);
        sigma2_tmp = sigma2b_old.slice(m-1).col(k).rows(n_1-1, n_T-1);

        if(!ind){
          if( k == 1){
            theta_sr_tmp = arma::vec(d_tmp, arma::fill::zeros);
            beta_mean_tmp = arma::vec(d_tmp, arma::fill::zeros);
            tau2_inv_tmp = arma::vec(d_tmp, arma::fill::zeros);
            xi2_inv_tmp = arma::vec(d_tmp, arma::fill::zeros);

            theta_sr_tmp(arma::span(0, d_tmp-2)) = thetab_sr_old.slice(m-1).col(k);
            theta_sr_tmp(d_tmp-1) = thetab_sr_chol_old(0, m-1);

            beta_mean_tmp(arma::span(0, d_tmp-2)) = betab_mean_old.slice(m-1).col(k);
            beta_mean_tmp(d_tmp-1) = betab_mean_chol_old(0, m-1);

            tau2_inv_tmp(arma::span(0, d_tmp-2)) = tau2b_inv_old.slice(m-1).col(k);
            tau2_inv_tmp(d_tmp-1) = tau2b_inv_chol_old(0, m-1);

            xi2_inv_tmp(arma::span(0, d_tmp-2)) = xi2b_inv_old.slice(m-1).col(k);
            xi2_inv_tmp(d_tmp-1) = xi2b_inv_chol_old(0, m-1);
          }else if(k > 1){
            index = arma::sum(arma::linspace(1, k-1, k-1));
            theta_sr_tmp = arma::join_cols(theta_sr_tmp, thetab_sr_chol_old.col(m-1).rows(index, index+k-1));
            beta_mean_tmp = arma::join_cols(beta_mean_tmp, betab_mean_chol_old.col(m-1).rows(index, index+k-1));
            tau2_inv_tmp = arma::join_cols(tau2_inv_tmp, tau2b_inv_chol_old.col(m-1).rows(index, index+k-1));
            xi2_inv_tmp = arma::join_cols(xi2_inv_tmp, xi2b_inv_chol_old.col(m-1).rows(index, index+k-1));
          }
        }

        try {
          update_beta_tilde(beta_nc_tmp, beta2_nc_tmp, beta_cov_nc_tmp,
                            y_tmp, x_tmp, theta_sr_tmp, beta_mean_tmp, N_m, S_0, sigma2_tmp);
          betab_nc_new.slice(m-1).rows(n_1-1, n_T-1).cols(k*n_I, (k+1)*n_I-1) = beta_nc_tmp.cols(0, n_I-1).rows(1, N_m);
          if(!ind){
            if(k == 1){
              betab_nc_chol_new.slice(m-1).rows(n_1-1, n_T-1).col(0) = beta_nc_tmp.col(n_I).rows(1, N_m);
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              betab_nc_chol_new.slice(m-1).rows(n_1-1, n_T-1).cols(index, index+k-1) = beta_nc_tmp.cols(n_I, d_tmp-1).rows(1, N_m);
            }
          }
          //(beta2b_nc_new.slice(m-1)).rows(n_1-1, n_T-1) = beta2_nc_tmp.rows(1, N_m);
          sigma2b_new.slice(m-1).col(k).rows(n_1-1, n_T-1) = sigma2_tmp;
          yb.slice(m).col(k).rows(n_1-1, n_T-1) = y_tmp;
          y_tmp = yb.slice(m-1).col(k).rows(n_1-1, n_T-1);
        } catch (...){
          //beta_nc_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update backward beta_nc";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // update beta mean
        try {
          update_beta_mean(beta_mean_tmp, beta2_mean_tmp, theta_sr_tmp,
                           y_tmp, x_tmp, beta_nc_tmp.rows(1, N_m), 1.0/sigma2_tmp, tau2_inv_tmp);
          betab_mean_new.slice(m-1).col(k) = beta_mean_tmp(arma::span(0, n_I-1));
          if(!ind){
            if(k == 1){
              betab_mean_chol_new(0, m-1) = beta_mean_tmp(n_I);
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              betab_mean_chol_new.col(m-1).rows(index, index+k-1) = beta_mean_tmp(arma::span(n_I, d_tmp-1));
            }
          }
        } catch(...){
          //beta_mean_tmp.fill(nanl(""));
          //beta2_mean_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update backward beta mean & beta mean square";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // update theta sr
        try{
          update_theta_sr(beta_mean_tmp, theta_sr_tmp, theta_tmp, y_tmp, x_tmp,
                          beta_nc_tmp.rows(1, N_m), beta2_nc_tmp.rows(1, N_m), beta_cov_nc_tmp.rows(1, N_m),
                          1.0/sigma2_tmp, xi2_inv_tmp);
          thetab_sr_new.slice(m-1).col(k) = theta_sr_tmp(arma::span(0, n_I-1));
          if(!ind){
            if(k == 1){
              thetab_sr_chol_new(0, m-1) = theta_sr_tmp(n_I);
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              thetab_sr_chol_new.col(m-1).rows(index, index+k-1) = theta_sr_tmp(arma::span(n_I, d_tmp-1));
            }
          }
        } catch(...){
          //theta_sr_tmp.fill(nanl(""));
          //theta_tmp.fill(nanl(""));
          if(succesful == true){
            fail = "update backward theta sr & theta";
            fail_iter = j + 1;
            succesful = false;
          }
        }
        //
        // update backward tau2
        //Rcout << "backward tau2" << "\n";
        try {
          update_local_shrink(tau2_tmp, tau2_inv_tmp, beta2_mean_tmp, lambda2b_old(k), a_taub_old(k));
          tau2b_new.slice(m-1).col(k) = tau2_tmp(arma::span(0, n_I-1));
          tau2b_inv_new.slice(m-1).col(k) = tau2_inv_tmp(arma::span(0, n_I-1));
          if(!ind){
            if(k == 1){
              tau2b_chol_new(0, m-1) = arma::as_scalar(tau2_tmp(n_I));
              tau2b_inv_chol_new(0, m-1) = arma::as_scalar(tau2_inv_tmp(n_I));
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              tau2b_chol_new.col(m-1).rows(index, index+k-1) = tau2_tmp(arma::span(n_I, d_tmp-1));
              tau2b_inv_chol_new.col(m-1).rows(index, index+k-1) = tau2_inv_tmp(arma::span(n_I, d_tmp-1));
            }
          }
        } catch(...) {
          //tau2_tmp.fill(nanl(""));
          //tau2_inv_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update backward tau2 & tau2_inv";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // update backward xi2
        //Rcout << "backward xi" << "\n";
        try {
          update_local_shrink(xi2_tmp, xi2_inv_tmp, theta_tmp, kappa2b_old(k), a_xib_old(k));
          xi2b_new.slice(m-1).col(k) = xi2_tmp(arma::span(0, n_I-1));
          xi2b_inv_new.slice(m-1).col(k) = xi2_inv_tmp(arma::span(0, n_I-1));
          if(!ind){
            if(k == 1){
              xi2b_chol_new(0, m-1) = arma::as_scalar(xi2_tmp(n_I));
              xi2b_inv_chol_new(0, m-1) = arma::as_scalar(xi2_inv_tmp(n_I));
            }else if(k > 1){
              index = arma::sum(arma::linspace(1, k-1, k-1));
              xi2b_chol_new.col(m-1).rows(index, index+k-1) = xi2_tmp(arma::span(n_I, d_tmp-1));
              xi2b_inv_chol_new.col(m-1).rows(index, index+k-1) = xi2_inv_tmp(arma::span(n_I, d_tmp-1));
            }
          }
        } catch(...) {
          //xi2_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update backward xi2 & xi2_inv";
            fail_iter = j + 1;
            succesful = false;
          }
        }
      }

      // transform back
      if(!ind){
        // forward part
        n_1 = m + 1;
        n_T = N;
        for(int i = n_1-1; i < n_T; i++){
          betaf_chol.slice(m-1).row(i) = (betaf_nc_chol_new.slice(m-1).row(i)) % arma::trans(thetaf_sr_chol_new.col(m-1)) + arma::trans(betaf_mean_chol_new.col(m-1));
          tmp_lower_triangular.elem(lower_indices) = betaf_chol.slice(m-1).row(i);
          yf.slice(m).row(i) = arma::trans(arma::inv(tmp_lower_triangular)*arma::trans(yf.slice(m).row(i)));
          if(arma::any(arma::abs(yf.slice(m).row(i)) > std::pow(10, 10))){
            Rcout << "iteration: " << i << "\n";
            Rcout << "yf: " << yf.slice(m).row(i) << std::endl << "\n";
            Rcout << "lower_triangular: " << arma::inv(tmp_lower_triangular) << std::endl << "\n";
            break;
          }
        }
        //backward part
        n_1 = 1;          // backward index
        n_T = N - m;      // backward index
        for(int i = n_1-1; i < n_T; i++){
          betab_chol.slice(m-1).row(i) = (betab_nc_chol_new.slice(m-1).row(i)) % arma::trans(thetab_sr_chol_new.col(m-1)) + arma::trans(betab_mean_chol_new.col(m-1));
          tmp_lower_triangular.elem(lower_indices) = betab_chol.slice(m-1).row(i);
          yb.slice(m).row(i) = arma::trans(arma::inv(tmp_lower_triangular)*arma::trans(yb.slice(m).row(i)));
          if(arma::any(arma::abs(yb.slice(m).row(i)) > std::pow(10, 10))){
            Rcout << "iteration: " << i << "\n";
            Rcout << "yb: " << yb.slice(m).row(i) << std::endl << "\n";
            Rcout << "betab_nc_chol: " << (betab_nc_chol_old.slice(m-1).row(i)) << "\n";
            Rcout << "thetab_sr_chol: " << arma::trans(thetab_sr_chol_old.col(m-1)) << "\n";
            Rcout << "betab_mean_chol: " << arma::trans(betab_mean_chol_old.col(m-1)) << "\n";
            Rcout << "lower_triangular: " << arma::inv(tmp_lower_triangular) << std::endl << "\n";
            break;
          }
        }
      }
    }

    //std::for_each(yf.begin(), yf.end(), res_protector);
    //std::for_each(yb.begin(), yb.end(), res_protector);

    // update forward kappa2 and lambda2
    for(int k = 0; k < n_I; k++){
      try {
        xi2_tmp = arma::vectorise(xi2f_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1)));
        if(!ind){
          if(k == 1){
            xi2_tmp = arma::join_cols(arma::vectorise(xi2f_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(xi2f_chol_new(0, arma::span(start-1, d-1))));
          }else if(k > 1){
            index = arma::sum(arma::linspace(1, k-1, k-1));
            xi2_tmp = arma::join_cols(arma::vectorise(xi2f_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(xi2f_chol_new(arma::span(index, index+k-1), arma::span(start-1, d-1))));
          }
        }
        kappa2f_new(k) = update_global_shrink(xi2_tmp, a_xif_new(k), d1, d2);
      } catch (...) {
        //kappa2f_new(k) = arma::datum::nan;
        if (succesful == true){
          fail = "update forward kappa2";
          fail_iter = j + 1;
          succesful = false;
        }
      }
    }
    for(int k = 0; k < n_I; k++){
      try {
        tau2_tmp = arma::vectorise(tau2f_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1)));
        if(!ind){
          if(k == 1){
            tau2_tmp = arma::join_cols(arma::vectorise(tau2f_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2f_chol_new(0, arma::span(start-1, d-1))));
          }else if(k > 1){
            index = arma::sum(arma::linspace(1, k-1, k-1));
            tau2_tmp = arma::join_cols(arma::vectorise(tau2f_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2f_chol_new(arma::span(index, index+k-1), arma::span(start-1, d-1))));
          }
        }
        lambda2f_new(k) = update_global_shrink(tau2_tmp, a_tauf_new(k), e1, e2);
      } catch (...) {
        //lambda2f_new(k) = arma::datum::nan;
        if (succesful == true){
          fail = "update forward lambda2";
          fail_iter = j + 1;
          succesful = false;
        }
      }
    }

    // sample backward kappa2 and lambda2
    for(int k = 0; k < n_I; k++){
      try {
        xi2_tmp = arma::vectorise(xi2b_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1)));
        if(!ind){
          if(k == 1){
            xi2_tmp = arma::join_cols(arma::vectorise(xi2b_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(xi2b_chol_new(0, arma::span(start-1, d-1))));
          }else if(k > 1){
            index = arma::sum(arma::linspace(1, k-1, k-1));
            xi2_tmp = arma::join_cols(arma::vectorise(xi2b_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(xi2b_chol_new(arma::span(index, index+k-1), arma::span(start-1, d-1))));
          }
        }
        kappa2b_new(k) = update_global_shrink(xi2_tmp, a_xib_new(k), d1, d2);
      } catch (...) {
        //kappa2b_new(k) = arma::datum::nan;
        if (succesful == true){
          fail = "update backward kappa2";
          fail_iter = j + 1;
          succesful = false;
        }
      }
    }
    for(int k = 0; k < n_I; k++){
      try {
        tau2_tmp = arma::vectorise(tau2b_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1)));
        if(!ind){
          if(k == 1){
            tau2_tmp = arma::join_cols(arma::vectorise(tau2b_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2b_chol_new(0, arma::span(start-1, d-1))));
          }else if(k > 1){
            index = arma::sum(arma::linspace(1, k-1, k-1));
            tau2_tmp = arma::join_cols(arma::vectorise(tau2b_new(arma::span(0, n_I-1), arma::span(k, k), arma::span(start-1, d-1))), arma::vectorise(tau2b_chol_new(arma::span(index, index+k-1), arma::span(start-1, d-1))));
          }
        }
        lambda2b_new(k) = update_global_shrink(tau2_tmp, a_taub_new(k), e1, e2);
      } catch (...) {
        //lambda2b_new(k) = arma::datum::nan;
        if (succesful == true){
          fail = "update backward lambda2";
          fail_iter = j + 1;
          succesful = false;
        }
      }
    }

    // Updating stop criterion
    for(int m=start-1; m < d; m++){
      if(!betaf_mean_new.slice(m).is_finite()){
        Rcout << "betaf_mean" << "\n";
      }
      if(!thetaf_sr_new.slice(m).is_finite()){
        Rcout << "thetaf_sr" << "\n";
      }
      if(!xi2f_new.slice(m).is_finite()){
        Rcout << "xi2f" << "\n";
      }
      if(!tau2f_new.slice(m).is_finite()){
        Rcout << "tau2f_sr" << "\n";
      }
      if(!xi2b_new.slice(m).is_finite()){
        Rcout << "xi2f" << "\n";
      }
      if(!tau2b_new.slice(m).is_finite()){
        Rcout << "tau2f_sr" << "\n";
      }
      if(!betab_mean_new.slice(m).is_finite()){
        Rcout << "betab_mean" << "\n";
      }
      if(!thetab_sr_new.slice(m).is_finite()){
        Rcout << "thetab_sr" << "\n";
      }

      if(!betaf_nc_new.slice(m).is_finite()){
        Rcout << "betaf" << "\n";
      }

      if(!betab_nc_new.slice(m).is_finite()){
        Rcout << "betab" << "\n";
      }
      diff += arma::norm(betaf_mean_new.slice(m) - betaf_mean_old.slice(m), 2);
      diff += arma::norm(thetaf_sr_new.slice(m) - thetaf_sr_old.slice(m), 2);
      diff += arma::norm(betab_mean_new.slice(m) - betab_mean_old.slice(m), 2);
      diff += arma::norm(thetab_sr_new.slice(m) - thetab_sr_old.slice(m), 2);

      diff += arma::norm(xi2f_new.slice(m) - xi2f_old.slice(m), 2);
      diff += arma::norm(tau2f_new.slice(m) - tau2f_old.slice(m), 2);
      diff += arma::norm(xi2b_new.slice(m) - xi2b_old.slice(m), 2);
      diff += arma::norm(tau2b_new.slice(m) - tau2b_old.slice(m), 2);

      diff += arma::norm(betaf_nc_new.slice(m).rows(d, N-d-1) - betaf_nc_old.slice(m).rows(d, N-d-1), 2);
      diff += arma::norm(betab_nc_new.slice(m).rows(d, N-d-1) - betab_nc_old.slice(m).rows(d, N-d-1), 2);

      if(!ind){
        diff += arma::norm(thetaf_sr_chol_new.col(m) - thetaf_sr_chol_old.col(m), 2);
        diff += arma::norm(thetab_sr_chol_new.col(m) - thetab_sr_chol_old.col(m), 2);

        diff += arma::norm(betaf_mean_chol_new.col(m) - betaf_mean_chol_old.col(m), 2);
        diff += arma::norm(betab_mean_chol_new.col(m) - betab_mean_chol_old.col(m), 2);

        diff += arma::norm(tau2f_chol_new.col(m) - tau2f_chol_old.col(m));
        diff += arma::norm(tau2b_chol_new.col(m) - tau2b_chol_old.col(m));

        diff += arma::norm(xi2f_chol_new.col(m) - xi2f_chol_old.col(m));
        diff += arma::norm(xi2b_chol_new.col(m) - xi2b_chol_old.col(m));

        diff += arma::norm(betaf_nc_chol_new.slice(m).rows(d, N-d-1) - betaf_nc_chol_old.slice(m).rows(d, N-d-1), 2);
        diff += arma::norm(betab_nc_chol_new.slice(m).rows(d, N-d-1) - betab_nc_chol_old.slice(m).rows(d, N-d-1), 2);
      }
    }
    if(!kappa2f_new.is_finite()){
      Rcout << "kappa2f" << "\n";
    }
    if(!lambda2f_new.is_finite()){
      Rcout << "lambda2f" << "\n";
    }

    if(!kappa2b_new.is_finite()){
      Rcout << "kappa2b" << "\n";
    }
    if(!lambda2b_new.is_finite()){
      Rcout << "lambda2b" << "\n";
    }

    diff += arma::norm(kappa2f_new - kappa2f_old, 2);
    diff += arma::norm(lambda2f_new - lambda2f_old, 2);
    diff += arma::norm(kappa2b_new - kappa2b_old, 2);
    diff += arma::norm(lambda2b_new - lambda2b_old, 2);


    // update old state
    betaf_mean_old = betaf_mean_new;
    betab_mean_old = betab_mean_new;


    thetaf_sr_old = thetaf_sr_new;
    thetab_sr_old = thetab_sr_new;

    sigma2f_old = sigma2f_new;
    sigma2b_old = sigma2b_new;

    xi2f_old = xi2f_new;
    xi2b_old = xi2b_new;
    xi2f_inv_old = xi2f_inv_new;
    xi2b_inv_old = xi2b_inv_new;

    tau2f_old = tau2f_new;
    tau2b_old = tau2b_new;
    tau2f_inv_old = tau2f_inv_new;
    tau2b_inv_old = tau2b_inv_new;

    kappa2f_old = kappa2f_new;
    kappa2b_old = kappa2b_new;

    lambda2f_old = lambda2f_new;
    lambda2b_old = lambda2b_new;

    betaf_nc_old = betaf_nc_new;
    betab_nc_old = betab_nc_new;

    if(!ind){
      betaf_nc_chol_old = betaf_nc_chol_new;
      betab_nc_chol_old = betab_nc_chol_new;

      betaf_mean_chol_old = betaf_mean_chol_new;
      betab_mean_chol_old = betab_mean_chol_new;


      thetaf_sr_chol_old = thetaf_sr_chol_new;
      thetab_sr_chol_old = thetab_sr_chol_new;


      xi2f_chol_old = xi2f_chol_new;
      xi2b_chol_old = xi2b_chol_new;
      xi2f_inv_chol_old = xi2f_inv_chol_new;
      xi2b_inv_chol_old = xi2b_inv_chol_new;

      tau2f_chol_old = tau2f_chol_new;
      tau2b_chol_old = tau2b_chol_new;
      tau2f_inv_chol_old = tau2f_inv_chol_new;
      tau2b_inv_chol_old = tau2b_inv_chol_new;
    }



    // Increment progress bar
    if (arma::any(prog_rep_points == j)) {
      p.increment();
    }
    //Rcout << "iteration:" << j << "\n";

    j += 1;
  }

  for(int m = start-1; m < d; m++){
    for(int i = 0; i < N; i++){
      betaf.slice(m).row(i) = (betaf_nc_new.slice(m).row(i)) % arma::trans(arma::vectorise(thetaf_sr_new.slice(m))) + arma::trans(arma::vectorise(betaf_mean_new.slice(m)));
      betab.slice(m).row(i) = (betab_nc_new.slice(m).row(i)) % arma::trans(arma::vectorise(thetab_sr_new.slice(m))) + arma::trans(arma::vectorise(betab_mean_new.slice(m)));
      if(!ind){
        // forward part
        tmp_beta.elem(all_indices) = betaf.slice(m).row(i);
        tmp_lower_triangular.elem(lower_indices) = betaf_chol.slice(m).row(i);
        betaf.slice(m).row(i) = arma::trans(arma::vectorise(tmp_beta*arma::inv(arma::trans(tmp_lower_triangular))));

        // backward part
        tmp_beta.elem(all_indices) = betab.slice(m).row(i);
        tmp_lower_triangular.elem(lower_indices) = betab_chol.slice(m).row(i);
        betab.slice(m).row(i) = arma::trans(arma::vectorise(tmp_beta*arma::inv(arma::trans(tmp_lower_triangular))));
      }
    }
  }



  // return everything as a nested list (due to size restrictions on Rcpp::Lists)
  if(ind){
    return Rcpp::List::create(_["SIGMA"] = List::create(_["f"] = sigma2f_old, _["b"] = sigma2b_old),
                              _["theta_sr"] = List::create(_["f"] = thetaf_sr_old, _["b"] = thetab_sr_old),
                              _["beta_mean"] = List::create(_["f"] = betaf_mean_old, _["b"] = betab_mean_old),
                              _["beta_nc"] = List::create(_["f"] = betaf_nc_old, _["b"] = betab_nc_old),
                              _["beta"] = List::create(_["f"] = betaf, _["b"] = betab),
                              _["xi2"] = List::create(_["f"] = xi2f_old, _["b"] = xi2b_old),
                              _["tau2"] = List::create(_["f"] = tau2f_old, _["b"] = tau2b_old),
                              _["xi2_inv"] = List::create(_["f"] = xi2f_inv_old, _["b"] = xi2b_inv_old),
                              _["tau2_inv"] = List::create(_["f"] = tau2f_inv_old, _["b"] = tau2b_inv_old),
                              _["kappa2"] = List::create(_["f"] = kappa2f_old, _["b"] = kappa2b_old),
                              _["lambda2"] = List::create(_["f"] = lambda2f_old, _["b"] = lambda2b_old),
                              _["iter"] = j,
                              _["diff"] = diff,
                              _["success_vals"] = List::create(
                                _["success"] = succesful,
                                _["fail"] = fail,
                                _["fail_iter"] = fail_iter)
    );
  }else{
    return Rcpp::List::create(_["SIGMA"] = List::create(_["f"] = sigma2f_old, _["b"] = sigma2b_old),
                              _["theta_sr"] = List::create(_["f"] = thetaf_sr_old, _["b"] = thetab_sr_old),
                              _["beta_mean"] = List::create(_["f"] = betaf_mean_old, _["b"] = betab_mean_old),
                              _["beta_nc"] = List::create(_["f"] = betaf_nc_old, _["b"] = betab_nc_old),
                              _["beta"] = List::create(_["f"] = betaf, _["b"] = betab),
                              _["beta_chol"] = List::create(_["f"] = betaf_chol, _["b"] = betab_chol),
                              _["xi2"] = List::create(_["f"] = xi2f_old, _["b"] = xi2b_old),
                              _["tau2"] = List::create(_["f"] = tau2f_old, _["b"] = tau2b_old),
                              _["xi2_inv"] = List::create(_["f"] = xi2f_inv_old, _["b"] = xi2b_inv_old),
                              _["tau2_inv"] = List::create(_["f"] = tau2f_inv_old, _["b"] = tau2b_inv_old),
                              _["kappa2"] = List::create(_["f"] = kappa2f_old, _["b"] = kappa2b_old),
                              _["lambda2"] = List::create(_["f"] = lambda2f_old, _["b"] = lambda2b_old),
                              _["iter"] = j,
                              _["diff"] = diff,
                              _["success_vals"] = List::create(
                                _["success"] = succesful,
                                _["fail"] = fail,
                                _["fail_iter"] = fail_iter)
    );
  }
}
