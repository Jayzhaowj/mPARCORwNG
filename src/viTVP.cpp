// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <stochvol.h>
#include <progress.hpp>
#include <math.h>
#include "ffbs.h"
#include "DG_vi_update_functions.h"
#include "common_update_functions.h"
using namespace Rcpp;

// [[Rcpp::export]]
List vi_shrinkTVP(arma::mat y,
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
                  double epsilon) {

  // Progress bar setup
  arma::vec prog_rep_points = arma::round(arma::linspace(0, iter_max, 50));
  Progress p(50, true);

  // Some necessary dimensions
  int N = y.n_rows;
  int n_I = y.n_cols;
  int n_I2 = std::pow(n_I, 2);


  // Some index
  //int m = 1; // current stage
  int N_m;
  int n_1;     // index
  int n_T;     // index

  // generate forward and backward prediction error
  arma::cube yf(N, n_I, d+1, arma::fill::none);
  arma::cube yb(N, n_I, d+1, arma::fill::none);
  yf.slice(0) = y;
  yb.slice(0) = y;

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

  arma::cube thetaf_new(n_I, n_I, d, arma::fill::none);
  arma::cube thetab_new(n_I, n_I, d, arma::fill::none);

  arma::cube betaf_mean_old(n_I, n_I, d, arma::fill::ones);
  arma::cube betab_mean_old(n_I, n_I, d, arma::fill::ones);

  arma::cube betaf_mean_new(n_I, n_I, d, arma::fill::ones);
  arma::cube betab_mean_new(n_I, n_I, d, arma::fill::ones);

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

  arma::vec tau2_tmp(n_I, arma::fill::ones);
  arma::vec tau2_inv_tmp;
  arma::vec xi2_tmp(n_I, arma::fill::ones);
  arma::vec xi2_inv_tmp;

  arma::vec sigma2_tmp;
  arma::vec sigma2_inv_tmp;
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

  thetaf_new.fill(1.0);
  thetab_new.fill(1.0);
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
    for(int m = 1; m < d+1; m++){
      for(int k = 0; k < n_I; k++){
        // Forward
        // ----------------------------
        n_1 = m + 1;
        n_T = N;
        N_m = n_T - n_1 + 1;
        y_tmp = yf.slice(m-1).col(k).rows(n_1-1, n_T-1);
        x_tmp = yb.slice(m-1).rows(n_1-m-1, n_T-m-1);
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

        tau2_inv_tmp = tau2f_inv_old.slice(m-1).col(k);
        xi2_inv_tmp = xi2f_inv_old.slice(m-1).col(k);
        sigma2_tmp = sigma2f_old.slice(m-1).col(k).rows(n_1-1, n_T-1);

        try {
          update_beta_tilde(beta_nc_tmp, beta2_nc_tmp, beta_cov_nc_tmp,
                            y_tmp, x_tmp, theta_sr_tmp, beta_mean_tmp, N_m, n_I, sigma2_tmp);
          betaf_nc_new.slice(m-1).rows(n_1-1, n_T-1).cols(k*n_I, (k+1)*n_I-1) = beta_nc_tmp.rows(1, N_m);
          //(beta2f_nc_new.slice(m-1)).rows(n_1-1, n_T-1) = beta2_nc_tmp.rows(1, N_m);
          sigma2f_new.slice(m-1).col(k).rows(n_1-1, n_T-1) = sigma2_tmp;
          yf.slice(m).col(k).rows(n_1-1, n_T-1) = y_tmp;
          y_tmp = yf.slice(m-1).col(k).rows(n_1-1, n_T-1);
        } catch (...){
          beta_nc_tmp.fill(arma::datum::nan);
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
          betaf_mean_new.slice(m-1).col(k) = beta_mean_tmp;
        } catch(...){
          beta_mean_tmp.fill(nanl(""));
          beta2_mean_tmp.fill(nanl(""));
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
          thetaf_sr_new.slice(m-1).col(k) = theta_sr_tmp;
          thetaf_new.slice(m-1).col(k) = theta_tmp;
        } catch(...){
          theta_sr_tmp.fill(nanl(""));
          theta_tmp.fill(nanl(""));
          if(succesful == true){
            fail = "update forward theta sr & theta";
            fail_iter = j + 1;
            succesful = false;
          }
        }
        // update forward tau2
        try {
          update_local_shrink(tau2_tmp, tau2_inv_tmp, beta2_mean_tmp, lambda2f_old(k), a_tauf_old);
          tau2f_new.slice(m-1).col(k) = tau2_tmp;
          tau2f_inv_new.slice(m-1).col(k) = tau2_inv_tmp;
        } catch(...) {
          tau2_tmp.fill(nanl(""));
          tau2_inv_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update forward tau2 & tau2_inv";
            fail_iter = j + 1;
            succesful = false;
          }
        }
        // update forward xi2
        try {
          update_local_shrink(xi2_tmp, xi2_inv_tmp, theta_tmp, kappa2f_old(k), a_xif_old);
          xi2f_new.slice(m-1).col(k) = xi2_tmp;
          xi2f_inv_new.slice(m-1).col(k) = xi2_inv_tmp;
        } catch(...) {
          xi2_tmp.fill(nanl(""));
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

        d_tmp = x_tmp.n_cols;
        beta_nc_tmp = arma::mat(N_m+1, d_tmp, arma::fill::zeros);
        beta2_nc_tmp = arma::mat(N_m+1, d_tmp, arma::fill::zeros);
        beta_cov_nc_tmp = arma::cube(N_m+1, d_tmp, d_tmp, arma::fill::zeros);

        theta_sr_tmp = thetab_sr_old.slice(m-1).col(k);
        beta_mean_tmp = betab_mean_old.slice(m-1).col(k);

        theta_tmp = arma::vec(d_tmp, arma::fill::zeros);
        beta2_mean_tmp = arma::vec(d_tmp, arma::fill::zeros);

        tau2_inv_tmp = tau2b_inv_old.slice(m-1).col(k);
        xi2_inv_tmp = xi2b_inv_old.slice(m-1).col(k);
        sigma2_tmp = sigma2b_old.slice(m-1).col(k).rows(n_1-1, n_T-1);

        try {
          update_beta_tilde(beta_nc_tmp, beta2_nc_tmp, beta_cov_nc_tmp,
                            y_tmp, x_tmp, theta_sr_tmp, beta_mean_tmp, N_m, n_I, sigma2_tmp);
          (betab_nc_new.slice(m-1)).rows(n_1-1, n_T-1).cols(k*n_I, (k+1)*n_I-1) = beta_nc_tmp.rows(1, N_m);
          //(beta2b_nc_new.slice(m-1)).rows(n_1-1, n_T-1) = beta2_nc_tmp.rows(1, N_m);
          sigma2b_new.slice(m-1).col(k).rows(n_1-1, n_T-1) = sigma2_tmp;
          yb.slice(m).col(k).rows(n_1-1, n_T-1) = y_tmp;
          y_tmp = yb.slice(m-1).col(k).rows(n_1-1, n_T-1);
        } catch (...){
          beta_nc_tmp.fill(nanl(""));
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
          betab_mean_new.slice(m-1).col(k) = beta_mean_tmp;
          //beta2b_mean_new.slice(m-1).row(k) = beta2_mean_tmp;
        } catch(...){
          beta_mean_tmp.fill(nanl(""));
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
          thetab_sr_new.slice(m-1).col(k) = theta_sr_tmp;
          thetab_new.slice(m-1).col(k) = theta_tmp;
        } catch(...){
          theta_sr_tmp.fill(nanl(""));
          theta_tmp.fill(nanl(""));
          if(succesful == true){
            fail = "update backward theta sr & theta";
            fail_iter = j + 1;
            succesful = false;
          }
        }
        //
        // update backward tau2
        try {
          update_local_shrink(tau2_tmp, tau2_inv_tmp, beta2_mean_tmp, lambda2f_old(k), a_tauf_old);
          tau2b_new.slice(m-1).col(k) = tau2_tmp;
          tau2b_inv_new.slice(m-1).col(k) = tau2_inv_tmp;
        } catch(...) {
          tau2_tmp.fill(nanl(""));
          tau2_inv_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update backward tau2 & tau2_inv";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // update backward xi2
        try {
          update_local_shrink(xi2_tmp, xi2_inv_tmp, theta_tmp, kappa2f_old(k), a_xif_old);
          xi2b_new.slice(m-1).col(k) = xi2_tmp;
          xi2b_inv_new.slice(m-1).col(k) = xi2_inv_tmp;
        } catch(...) {
          xi2_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "update backward xi2 & xi2_inv";
            fail_iter = j + 1;
            succesful = false;
          }
        }




      }

    }
    // update forward kappa2 and lambda2
    for(int k = 0; k < n_I; k++){
      try {
        //arma::vec xi2f_tmp = xi2f_samp.row(k);
        kappa2f_new(k) = update_global_shrink(arma::vectorise(xi2f_new.col(k)),
                                              a_xif_new(k), d1, d2, n_I*d);
      } catch (...) {
        kappa2f_new(k) = arma::datum::nan;
        if (succesful == true){
          fail = "update forward kappa2";
          fail_iter = j + 1;
          succesful = false;
        }
      }
    }
    for(int k = 0; k < n_I; k++){
      try {
        //arma::vec tau2f_tmp = tau2f_samp.row(k);
        lambda2f_new(k) = update_global_shrink(arma::vectorise(tau2f_new.col(k)),
                                               a_tauf_new(k), e1, e2, n_I*d);

      } catch (...) {
        lambda2f_new(k) = arma::datum::nan;
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
        //arma::vec xi2f_tmp = xi2f_samp.row(k);
        kappa2b_new(k) = update_global_shrink(arma::vectorise(xi2b_new.col(k)),
                                              a_xib_new(k), d1, d2, n_I*d);
      } catch (...) {
        kappa2b_new(k) = arma::datum::nan;
        if (succesful == true){
          fail = "update backward kappa2";
          fail_iter = j + 1;
          succesful = false;
        }
      }
    }
    for(int k = 0; k < n_I; k++){
      try {
        //arma::vec tau2f_tmp = tau2f_samp.row(k);
        lambda2b_new(k) = update_global_shrink(arma::vectorise(tau2b_new.col(k)),
                                               a_taub_new(k), e1, e2, d*n_I);

      } catch (...) {
        lambda2b_new(k) = arma::datum::nan;
        if (succesful == true){
          fail = "update backward lambda2";
          fail_iter = j + 1;
          succesful = false;
        }
      }
    }

    // Updating stop criterion
    for(int m=0; m < d; m++){
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

    // Increment progress bar
    if (arma::any(prog_rep_points == j)) {
      p.increment();
    }
    //Rcout << "iteration:" << j << "\n";

    j += 1;
  }
  for(int m = 0; m < d; m++){
    for(int i = 0; i < N; i++){
      betaf.slice(m).row(i) = (betaf_nc_old.slice(m).row(i)) % arma::trans(arma::vectorise(thetaf_sr_old.slice(m))) + arma::trans(arma::vectorise(betaf_mean_old.slice(m)));
      betab.slice(m).row(i) = (betab_nc_old.slice(m).row(i)) % arma::trans(arma::vectorise(thetab_sr_old.slice(m))) + arma::trans(arma::vectorise(betab_mean_old.slice(m)));
    }
  }
  // return everything as a nested list (due to size restrictions on Rcpp::Lists)
  return Rcpp::List::create(_["SIGMA"] = List::create(_["f"] = sigma2f_old, _["b"] = sigma2b_old),
                            _["theta_sr"] = List::create(_["f"] = thetaf_sr_old, _["b"] = thetab_sr_old),
                            _["theta"] = List::create(_["f"] = thetaf_new, _["b"] = thetab_new),
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
}
