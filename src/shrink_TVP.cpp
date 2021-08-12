// [[Rcpp::depends(RcppArmadillo, RcppProgress)]]
#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <stochvol.h>
#include <progress.hpp>
#include <math.h>
//#include "sample_beta_McCausland.h"
#include "sample_parameters.h"
#include "MH_step.h"
using namespace Rcpp;

// [[Rcpp::export]]
List do_shrinkTVP(arma::mat y,
                  arma::vec a0,
                  double S_0,
                  int d,
                  int niter,
                  int nburn,
                  int nthin,
                  double c0,
                  double g0,
                  double G0,
                  double d1,
                  double d2,
                  double e1,
                  double e2,
                  bool learn_lambda2,
                  bool learn_kappa2,
                  double lambda2,
                  double kappa2,
                  bool learn_a_xi,
                  bool learn_a_tau,
                  double a_xi,
                  double a_tau,
                  double c_tuning_par_xi,
                  double c_tuning_par_tau,
                  double b_xi,
                  double b_tau,
                  double nu_xi,
                  double nu_tau,
                  bool display_progress,
                  bool ret_beta_nc,
                  bool store_burn) {

  // progress bar setup
  arma::vec prog_rep_points = arma::round(arma::linspace(0, niter, 50));
  Progress p(50, display_progress);

  // Import Rs chol function
  Environment base = Environment("package:base");
  Function Rchol = base["chol"];

  // Some necessary dimensions
  int N = y.n_rows;
  int n_I = y.n_cols;
  int n_I2 = std::pow(n_I, 2);
  int nsave;
  if (store_burn){
    nsave = std::floor(niter/nthin);
  } else {
    nsave = std::floor((niter - nburn)/nthin);
  }


  // Some index
  //int m = 1; // current stage
  int N_m;
  int n_1; // index
  int n_T;     // index

  // generate forward and backward prediction error
  arma::cube yf(N, n_I, d+1, arma::fill::none);
  arma::cube yb(N, n_I, d+1, arma::fill::none);
  yf.slice(0) = y;
  yb.slice(0) = y;
  arma::vec y_tmp;


  // generate forward and backward prediction matrix
  arma::mat x_tmp;
  arma::mat x_tilde;
  // generate forward and backward PARCOR
  Rcpp::List betaf_save(nsave);
  Rcpp::List betab_save(nsave);
  arma::cube betaf_samp(N, n_I2, d, arma::fill::none);
  arma::cube betab_samp(N, n_I2, d, arma::fill::none);
  arma::mat beta_nc_tmp;


  Rcpp::List SIGMAf_save(nsave);
  Rcpp::List SIGMAb_save(nsave);

  arma::cube SIGMAf_samp(N, n_I, d, arma::fill::none);
  arma::cube SIGMAb_samp(N, n_I, d, arma::fill::none);

  arma::vec sigma2_tmp;

  Rcpp::List thetaf_sr_save(nsave);
  Rcpp::List thetab_sr_save(nsave);

  Rcpp::List betaf_mean_save(nsave);
  Rcpp::List betab_mean_save(nsave);

  Rcpp::List xi2f_save(nsave);
  Rcpp::List xi2b_save(nsave);

  Rcpp::List tau2f_save(nsave);
  Rcpp::List tau2b_save(nsave);

  // Storage objects
  // conditional storage objects
  arma::mat kappa2f_save;
  arma::mat kappa2b_save;
  arma::mat lambda2f_save;
  arma::mat lambda2b_save;
  if (learn_kappa2){
    kappa2f_save = arma::mat(n_I, nsave, arma::fill::none);
    kappa2b_save = arma::mat(n_I, nsave, arma::fill::none);
  }
  if (learn_lambda2){
    lambda2f_save = arma::mat(n_I, nsave, arma::fill::none);
    lambda2b_save = arma::mat(n_I, nsave, arma::fill::none);
  }

  Rcpp::List betaf_nc_save(nsave);
  Rcpp::List betab_nc_save(nsave);

  arma::cube betaf_nc_samp(N, n_I2, d, arma::fill::zeros);
  arma::cube betab_nc_samp(N, n_I2, d, arma::fill::zeros);

  arma::mat a_xif_save;
  arma::mat a_tauf_save;

  arma::mat a_xib_save;
  arma::mat a_taub_save;

  arma::vec accept_a_xif_tot(n_I, arma::fill::zeros);
  arma::vec accept_a_xif_pre(n_I, arma::fill::zeros);
  arma::vec accept_a_xif_post(n_I, arma::fill::zeros);

  arma::vec accept_a_tauf_tot(n_I, arma::fill::zeros);
  arma::vec accept_a_tauf_pre(n_I, arma::fill::zeros);
  arma::vec accept_a_tauf_post(n_I, arma::fill::zeros);

  arma::vec accept_a_xib_tot(n_I, arma::fill::zeros);
  arma::vec accept_a_xib_pre(n_I, arma::fill::zeros);
  arma::vec accept_a_xib_post(n_I, arma::fill::zeros);

  arma::vec accept_a_taub_tot(n_I, arma::fill::zeros);
  arma::vec accept_a_taub_pre(n_I, arma::fill::zeros);
  arma::vec accept_a_taub_post(n_I, arma::fill::zeros);

  if (learn_a_xi){
    a_xif_save = arma::mat(n_I, nsave, arma::fill::none);
    a_xib_save = arma::mat(n_I, nsave, arma::fill::none);
  }
  if (learn_a_tau){
    a_tauf_save = arma::mat(n_I, nsave, arma::fill::none);
    a_taub_save = arma::mat(n_I, nsave, arma::fill::none);
  }

  arma::mat beta_nc_tmp_tilde;
  arma::mat betaenter;
  arma::mat beta_diff_pre;
  arma::mat beta_diff;

  arma::cube betaf_mean_samp(n_I, n_I, d);
  betaf_mean_samp.fill(0.1);

  arma::cube betab_mean_samp(n_I, n_I, d);
  betab_mean_samp.fill(0.1);

  arma::vec beta_mean_tmp(n_I);
  beta_mean_tmp.fill(0.1);

  arma::cube thetaf_sr_samp(n_I, n_I, d);
  thetaf_sr_samp.fill(0.2);

  arma::cube thetab_sr_samp(n_I, n_I, d);
  thetab_sr_samp.fill(0.2);

  arma::vec theta_sr_tmp(n_I);
  theta_sr_tmp.fill(0.2);

  arma::cube tau2f_samp(n_I, n_I, d);
  tau2f_samp.fill(0.1);

  arma::cube tau2b_samp(n_I, n_I, d);
  tau2b_samp.fill(0.1);

  arma::vec tau2_tmp(n_I);
  tau2_tmp.fill(0.2);

  arma::cube xi2f_samp(n_I, n_I, d);
  xi2f_samp.fill(0.1);

  arma::cube xi2b_samp(n_I, n_I, d);
  xi2b_samp.fill(0.1);

  arma::vec xi2_tmp(n_I);
  xi2_tmp.fill(0.1);

  arma::vec xi_tau_tmp = arma::join_cols(xi2_tmp, tau2_tmp);

  arma::vec kappa2f_samp(n_I);
  kappa2f_samp.fill(20);

  arma::vec lambda2f_samp(n_I);
  lambda2f_samp.fill(0.1);

  arma::vec a_xif_samp(n_I);
  a_xif_samp.fill(0.1);

  arma::vec a_tauf_samp(n_I);
  a_tauf_samp.fill(0.1);

  arma::vec alpha_tmp(2*n_I, arma::fill::ones);

  arma::vec kappa2b_samp(n_I);
  kappa2b_samp.fill(20);

  arma::vec lambda2b_samp(n_I);
  lambda2b_samp.fill(0.1);

  arma::vec a_xib_samp(n_I);
  a_xib_samp.fill(0.1);

  arma::vec a_taub_samp(n_I);
  a_taub_samp.fill(0.1);


  // Override inital values with user specified fixed values
  if (learn_kappa2 == false){
    kappa2f_samp.fill(kappa2);
    kappa2b_samp.fill(kappa2);
  }
  if (learn_lambda2 == false){
    lambda2f_samp.fill(lambda2);
    lambda2b_samp.fill(lambda2);
  }

  if (!learn_a_xi){
    a_xif_samp.fill(a_xi);
    a_xib_samp.fill(a_xi);
  }
  if (!learn_a_tau){
    a_tauf_samp.fill(a_tau);
    a_taub_samp.fill(a_tau);
  }

  // Values to check if the sampler failed or not
  bool succesful = true;
  std::string fail;
  int fail_iter;


  // Introduce additional index post_j that is used to calculate accurate storage positions in case of thinning
  int post_j = 1;

  // Begin Gibbs loop
  for (int j = 0; j < niter; j++){

    for (int m = 1; m < d+1; m++){
      for(int k = 0; k < n_I; k++){

        // Forward
        n_1 = m + 1; // forward index
        n_T = N;     // forward index
        N_m = n_T - n_1 + 1; // time point at stage m

        y_tmp = yf.slice(m-1).col(k).rows(n_1-1, n_T-1);
        x_tmp = yb.slice(m-1).rows(n_1-m-1, n_T-m-1);

        int d_tmp = x_tmp.n_cols;
        beta_nc_tmp = arma::mat(N_m+1, d_tmp, arma::fill::zeros);
        theta_sr_tmp = thetaf_sr_samp.slice(m-1).col(k);
        beta_mean_tmp = betaf_mean_samp.slice(m-1).col(k);
        tau2_tmp = tau2f_samp.slice(m-1).col(k);
        xi2_tmp = xi2f_samp.slice(m-1).col(k);
        sigma2_tmp = SIGMAf_samp.slice(m-1).col(k).rows(n_1-1, n_T-1);

        // sample beta_nc
        try {
          sample_beta_tilde(beta_nc_tmp, y_tmp, x_tmp, theta_sr_tmp, beta_mean_tmp, N_m, n_I, S_0, sigma2_tmp);

          betaf_nc_samp.slice(m-1).rows(n_1-1, n_T-1).cols(k*n_I, (k+1)*n_I-1) = beta_nc_tmp.rows(1, N_m);
          SIGMAf_samp.slice(m-1).col(k).rows(n_1-1, n_T-1) = sigma2_tmp;
          yf.slice(m).col(k).rows(n_1-1, n_T-1) = y_tmp;
          y_tmp = yf.slice(m-1).col(k).rows(n_1-1, n_T-1);
        } catch (...){
          beta_nc_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "sample forward beta_nc";
            fail_iter = j + 1;
            succesful = false;
         }
        }

        x_tilde = x_tmp % beta_nc_tmp.rows(1, N_m);

        // sample forward alpha
        try {
          sample_alpha(alpha_tmp, y_tmp, x_tmp, x_tilde, tau2_tmp, xi2_tmp, sigma2_tmp, a0, n_I, Rchol);
        } catch(...){
          alpha_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "sample forward alpha";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // Weave back into centered parameterization
        beta_mean_tmp = alpha_tmp(arma::span(0, d_tmp-1));
        theta_sr_tmp = alpha_tmp(arma::span(d_tmp, 2*d_tmp-1));
        beta_nc_tmp_tilde = beta_nc_tmp.each_row() % arma::trans(theta_sr_tmp);
        betaenter = beta_nc_tmp_tilde.each_row() + arma::trans(beta_mean_tmp);

        // Difference beta outside of function (for numerical stability)
        beta_diff_pre = arma::diff(beta_nc_tmp, 1, 0);
        beta_diff =  beta_diff_pre.each_row() % arma::trans(theta_sr_tmp);

        // resample alpha
        try {
           resample_alpha_diff(alpha_tmp, betaenter, theta_sr_tmp, beta_mean_tmp, beta_diff, xi2_tmp, tau2_tmp, n_I, N_m);
        } catch(...) {
           alpha_tmp.fill(nanl(""));
           if (succesful == true){
              fail = "resample forward alpha";
              fail_iter = j + 1;
              succesful = false;
           }
        }

        betaf_nc_samp.slice(m-1).rows(n_1-1, n_T-1).cols(k*n_I, (k+1)*n_I-1) = beta_nc_tmp.rows(1-1, N_m-1);
        betaf_mean_samp.slice(m-1).col(k) = alpha_tmp(arma::span(0, n_I-1));
        thetaf_sr_samp.slice(m-1).col(k) = alpha_tmp(arma::span(n_I, 2*n_I-1));
        betaf_samp.slice(m-1).rows(n_1-1, n_T-1).cols(k*n_I, (k+1)*n_I-1) = betaenter.rows(1-1, N_m-1);

        // sample forward tau2
        try {
          sample_tau2(tau2_tmp, beta_mean_tmp, lambda2f_samp(k), a_tauf_samp(k), n_I);
          tau2f_samp.slice(m-1).col(k) = tau2_tmp;
        } catch(...) {
          tau2_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "sample forward tau2";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // sample forward xi2
        try {
          sample_xi2(xi2_tmp, theta_sr_tmp, kappa2f_samp(k), a_xif_samp(k), n_I);
          xi2f_samp.slice(m-1).col(k) = xi2_tmp;
        } catch(...) {
          xi2_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "sample forward xi2";
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
        theta_sr_tmp = thetab_sr_samp.slice(m-1).col(k);
        beta_mean_tmp = betab_mean_samp.slice(m-1).col(k);
        tau2_tmp = tau2b_samp.slice(m-1).col(k);
        xi2_tmp = xi2b_samp.slice(m-1).col(k);
        sigma2_tmp = SIGMAb_samp.slice(m-1).col(k).rows(n_1-1, n_T-1);

        // sample beta_nc
        try {
          sample_beta_tilde(beta_nc_tmp, y_tmp, x_tmp, theta_sr_tmp, beta_mean_tmp, N_m, n_I, S_0, sigma2_tmp);
          betab_nc_samp.slice(m-1).rows(n_1-1, n_T-1).cols(k*n_I, (k+1)*n_I-1) = beta_nc_tmp.rows(1, N_m);
          SIGMAb_samp.slice(m-1).col(k).rows(n_1-1, n_T-1) = sigma2_tmp;
          yb.slice(m).col(k).rows(n_1-1, n_T-1) = y_tmp;
          y_tmp = yb.slice(m-1).col(k).rows(n_1-1, n_T-1);
        } catch (...){
          beta_nc_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "sample backward beta_nc";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        x_tilde = x_tmp % beta_nc_tmp.rows(1, N_m);
        // sample forward alpha
        try {
          sample_alpha(alpha_tmp, y_tmp, x_tmp, x_tilde, tau2_tmp, xi2_tmp, sigma2_tmp, a0, n_I, Rchol);
        } catch(...){
          alpha_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "sample backward alpha";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // Weave back into centered parameterization
        beta_mean_tmp = alpha_tmp(arma::span(0, d_tmp-1));
        theta_sr_tmp = alpha_tmp(arma::span(d_tmp, 2*d_tmp-1));
        beta_nc_tmp_tilde = beta_nc_tmp.each_row() % arma::trans(theta_sr_tmp);
        betaenter = beta_nc_tmp_tilde.each_row() + arma::trans(beta_mean_tmp);

        // Difference beta outside of function (for numerical stability)
        beta_diff_pre = arma::diff(beta_nc_tmp, 1, 0);
        beta_diff =  beta_diff_pre.each_row() % arma::trans(theta_sr_tmp);

        // resample alpha
        try {
          resample_alpha_diff(alpha_tmp, betaenter, theta_sr_tmp, beta_mean_tmp, beta_diff, xi2_tmp, tau2_tmp, n_I, N_m);
        } catch(...) {
          alpha_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "resample backward alpha";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        betab_nc_samp.slice(m-1).rows(n_1-1, n_T-1).cols(k*n_I, (k+1)*n_I-1) = beta_nc_tmp.rows(1-1, N_m-1);
        betab_mean_samp.slice(m-1).col(k) = alpha_tmp(arma::span(0, n_I-1));
        thetab_sr_samp.slice(m-1).col(k) = alpha_tmp(arma::span(n_I, 2*n_I-1));
        betab_samp.slice(m-1).rows(n_1-1, n_T-1).cols(k*n_I, (k+1)*n_I-1) = betaenter.rows(1-1, N_m-1);

        // sample backward tau2
        try {
          sample_tau2(tau2_tmp, beta_mean_tmp, lambda2b_samp(k), a_taub_samp(k), n_I);
          tau2b_samp.slice(m-1).col(k) = tau2_tmp;
        } catch(...) {
          tau2_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "sample backward tau2";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        // sample backward xi2
        try {
          sample_xi2(xi2_tmp, theta_sr_tmp, kappa2b_samp(k), a_xib_samp(k), n_I);
          xi2b_samp.slice(m-1).col(k) = xi2_tmp;
        } catch(...) {
          xi2_tmp.fill(nanl(""));
          if (succesful == true){
            fail = "sample backward xi2";
            fail_iter = j + 1;
            succesful = false;
          }
        }



      }
    }


    if (learn_a_tau){
      for(int k = 0; k < n_I; k++){
        double before = a_tauf_samp(k);
        try {
          double tmp = MH_step(a_tauf_samp(k), c_tuning_par_tau, n_I*d, lambda2f_samp(k), arma::vectorise(betaf_mean_samp.col(k)), b_tau , nu_tau, e1, e2);
          a_tauf_samp(k) = tmp;
        } catch(...){
          a_tauf_samp(k) = arma::datum::nan;
          if (succesful == true){
            fail = "sample forward a_tau";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        if (before != a_tauf_samp(k)){
          accept_a_tauf_tot(k) += 1;
          if (j < nburn){
            accept_a_tauf_pre(k) += 1;
          } else {
            accept_a_tauf_post(k) += 1;
          }
        }
      }

    }

    // sample kappa2 and lambda2, if the user specified it
    if (learn_kappa2){
      for(int k = 0; k < n_I; k++){
        try {
          //arma::vec xi2f_tmp = xi2f_samp.row(k);
          double tmp = sample_kappa2(arma::vectorise(xi2f_samp.col(k)), a_xif_samp(k), d1, d2, (d)*n_I);
          kappa2f_samp(k) = tmp;
        } catch (...) {
          kappa2f_samp(k) = arma::datum::nan;
          if (succesful == true){
            fail = "sample forward kappa2";
            fail_iter = j + 1;
            succesful = false;
          }
        }
      }
    }

    if (learn_lambda2){
      for(int k = 0; k < n_I; k++){
        try {
          //arma::vec tau2f_tmp = tau2f_samp.row(k);
          double tmp = sample_lambda2(arma::vectorise(tau2f_samp.col(k)), a_tauf_samp(k), e1, e2, (d)*n_I);
          lambda2f_samp(k) = tmp;
        } catch (...) {
          lambda2f_samp(k) = arma::datum::nan;
          if (succesful == true){
            fail = "sample forward lambda2";
            fail_iter = j + 1;
            succesful = false;
          }
        }
      }
    }

    // step d)
    // sample a_xib and a_taub with MH
    if (learn_a_xi){
      //arma::cout << "size of a_xif_samp" << arma::size(a_xif_samp) << arma::endl;
      for(int k = 0; k < n_I; k++){
        double before = a_xib_samp(k);
        //arma::cout << "size of kappa2f_samp " << arma::size(kappa2f_samp) << arma::endl;
        //arma::cout << "size of thetaf_sr_samp " << arma::size(thetab_sr_samp) << arma::endl;
        try {
          double tmp = MH_step(a_xib_samp(k), c_tuning_par_xi, (d)*n_I, kappa2b_samp(k), arma::vectorise(thetab_sr_samp.col(k)), b_xi, nu_xi, d1, d2);
          a_xib_samp(k) = tmp;
        } catch(...) {
          a_xib_samp(k) = arma::datum::nan;
          if (succesful == true){
            fail = "sample forward a_xi";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        if (before != a_xib_samp(k)){
          accept_a_xib_tot(k) += 1;
          if (j < nburn){
            accept_a_xib_pre(k) += 1;
          } else {
            accept_a_xib_post(k) += 1;
          }
        }
      }

    }


    if (learn_a_tau){
      for(int k = 0; k < n_I; k++){
        double before = a_taub_samp(k);
        try {
          double tmp = MH_step(a_taub_samp(k), c_tuning_par_tau, (d)*n_I, lambda2b_samp(k), arma::vectorise(betab_mean_samp.col(k)), b_tau , nu_tau, e1, e2);
          a_taub_samp(k) = tmp;
        } catch(...){
          a_taub_samp(k) = arma::datum::nan;
          if (succesful == true){
            fail = "sample forward a_tau";
            fail_iter = j + 1;
            succesful = false;
          }
        }

        if (before != a_taub_samp(k)){
          accept_a_taub_tot(k) += 1;
          if (j < nburn){
            accept_a_taub_pre(k) += 1;
          } else {
            accept_a_taub_post(k) += 1;
          }
        }
      }

    }


    // sample kappa2 and lambda2, if the user specified it
    if (learn_kappa2){
      for(int k = 0; k < n_I; k++){
        try {
          //arma::vec xi2f_tmp = xi2f_samp.row(k);
          double tmp = sample_kappa2(arma::vectorise(xi2b_samp.col(k)), a_xib_samp(k), d1, d2, (d)*n_I);
          kappa2b_samp(k) = tmp;
        } catch (...) {
          kappa2b_samp(k) = arma::datum::nan;
          if (succesful == true){
            fail = "sample forward kappa2";
            fail_iter = j + 1;
            succesful = false;
          }
        }
      }
    }

    if (learn_lambda2){
      for(int k = 0; k < n_I; k++){
        try {
          //arma::vec tau2f_tmp = tau2f_samp.row(k);
          double tmp = sample_lambda2(arma::vectorise(tau2b_samp.col(k)), a_taub_samp(k), e1, e2, (d)*n_I);
          lambda2b_samp(k) = tmp;
        } catch (...) {
          lambda2b_samp(k) = arma::datum::nan;
          if (succesful == true){
            fail = "sample forward lambda2";
            fail_iter = j + 1;
            succesful = false;
          }
        }
      }
    }

    // adjust start of storage point depending on store_burn
    int nburn_new = nburn;
    if(store_burn){
      nburn_new = 0;
    }


    // Increment index i if burn-in period is over
    if (j > nburn_new){
      post_j++;
    }

    // Store everything
    if ((post_j % nthin == 0) && (j >= nburn_new)){
      // Caluclate beta
      // This is in the if condition to save unnecessary computations if beta is not saved
      //arma::cout << "size of beta_nc_samp: " << arma::size(betaf_nc_samp) << arma::endl;
      //arma::cout << "size of other:" << arma::size((betaf_nc_samp.each_col() % thetaf_sr_samp)) << arma::endl;
      //arma::mat betaf =  (betaf_nc_samp.each_col() % thetaf_sr_samp).each_col() + betaf_mean_samp;
      //arma::mat betab =  (betab_nc_samp.each_col() % thetab_sr_samp).each_col() + betab_mean_samp;

      //arma::cout << "size of SIGMAf_save: " << arma::size(SIGMAf_save) << arma::endl;
      //arma::cout << "size of SIGMAb_save: " << arma::size(SIGMAb_save) << arma::endl;

      SIGMAf_save((post_j-1)/nthin) = SIGMAf_samp;
      SIGMAb_save((post_j-1)/nthin) = SIGMAb_samp;

      //arma::cout << "size of thetaf_sr_save: " << arma::size(thetaf_sr_save) << arma::endl;
      //arma::cout << "size of thetab_sr_save: " << arma::size(thetab_sr_save) << arma::endl;

      thetaf_sr_save((post_j-1)/nthin) = thetaf_sr_samp;
      thetab_sr_save((post_j-1)/nthin) = thetab_sr_samp;

      //arma::cout << "size of thetaf_sr_samp: " << arma::size(thetaf_sr_samp) << arma::endl;
      //arma::cout << "size of thetab_sr_samp: " << arma::size(thetab_sr_samp) << arma::endl;
      betaf_mean_save((post_j-1)/nthin) = betaf_mean_samp;
      betab_mean_save((post_j-1)/nthin) = betab_mean_samp;

      //arma::cout << "size of betaf: " << arma::size(betaf_samp) << arma::endl;
      //arma::cout << "size of betab:" << arma::size(betab_samp) << arma::endl;
      //arma::cout << "size of betaf_save: " << arma::size(betaf_save) << arma::endl;
      //arma::cout << "size of betab_save:" << arma::size(betab_save) << arma::endl;
      betaf_save((post_j-1)/nthin) = betaf_samp;
      betab_save((post_j-1)/nthin) = betab_samp;

      //arma::cout << "size of xi2f_save: " << arma::size(xi2f_save) << arma::endl;
      //arma::cout << "size of xi2b_save: " << arma::size(xi2b_save) << arma::endl;
      xi2f_save((post_j-1)/nthin) = xi2f_samp;
      xi2b_save((post_j-1)/nthin) = xi2b_samp;

      //arma::cout << "size of tau2f_save: " << arma::size(tau2f_save) << arma::endl;
      //arma::cout << "size of tau2b_save: " << arma::size(tau2b_save) << arma::endl;
      tau2f_save((post_j-1)/nthin) = tau2f_samp;
      tau2b_save((post_j-1)/nthin) = tau2b_samp;
      //m_N_save.slice((post_j-1)/nthin) = m_N_samp;
      //chol_C_N_inv_save.slice((post_j-1)/nthin) = chol_C_N_inv_samp;

      //conditional storing
      //if (ret_beta_nc){
      //  beta_nc_save.slice((post_j-1)/nthin) = beta_nc_samp.t();
      //}

      if (learn_kappa2){
        //arma::cout << "size of kappa2f_save: " << arma::size(kappa2f_save) << arma::endl;
        //arma::cout << "size of kappa2b_save: " << arma::size(kappa2b_save) << arma::endl;
        kappa2f_save.col((post_j-1)/nthin) = kappa2f_samp;
        kappa2b_save.col((post_j-1)/nthin) = kappa2b_samp;
      }
      if (learn_lambda2){
        //arma::cout << "size of lambda2f_save: " << arma::size(lambda2f_save) << arma::endl;
        //arma::cout << "size of lambda2b_save: " << arma::size(lambda2b_save) << arma::endl;
        lambda2f_save.col((post_j-1)/nthin) = lambda2f_samp;
        lambda2b_save.col((post_j-1)/nthin) = lambda2b_samp;
      }

      if (learn_a_xi){
        a_xif_save.col((post_j-1)/nthin) = a_xif_samp;
        a_xib_save.col((post_j-1)/nthin) = a_xib_samp;
      }

      if (learn_a_tau){
        a_tauf_save.col((post_j-1)/nthin) = a_tauf_samp;
        a_taub_save.col((post_j-1)/nthin) = a_taub_samp;
      }

    }

    // Random sign switch
    for (int i = 0; i < d; i++){
      for(int j = 0; j < n_I; j++){
        for(int k = 0; k < n_I; k++){
          if(R::runif(0,1) > 0.5){
            thetaf_sr_samp(j, k, i) = -thetaf_sr_samp(j, k, i);
          }

          if(R::runif(0, 1) > 0.5){
            thetab_sr_samp(j, k, i) = -thetab_sr_samp(j, k, i);
          }
        }
      }
    }

    // Increment progress bar
    if (arma::any(prog_rep_points == j)){
      p.increment();
    }

    // Check for user interrupts
    if (j % 500 == 0){
      Rcpp::checkUserInterrupt();
    }

    // Break loop if succesful is false
    if (!succesful){
      break;
    }
  }

  // return everything as a nested list (due to size restrictions on Rcpp::Lists)
  return List::create(_["SIGMA"] = List::create(_["f"] = SIGMAf_save, _["b"] = SIGMAb_save),
                      _["theta_sr"] = List::create(_["f"] = thetaf_sr_save, _["b"] = thetab_sr_save),
                      _["beta_mean"] = List::create(_["f"] = betaf_mean_save, _["b"] = betab_mean_save),
                      _["beta_nc"] = List::create(_["f"] = betaf_nc_save, _["b"] = betab_nc_save),
                      _["beta"] = List::create(_["f"] = betaf_save, _["b"] = betab_save),
                      _["xi2"] = List::create(_["f"] = xi2f_save, _["b"] = xi2b_save),
                      _["a_xi"] = List::create(_["f"] = a_xif_save, _["b"] = a_xib_save),
                      _["a_xi_acceptance"] = List::create(
                        _["a_xif_acceptance_total"] = accept_a_xif_tot/niter,
                        _["a_xif_acceptance_pre"] = accept_a_xif_pre/nburn,
                        _["a_xif_acceptance_post"] = accept_a_xif_post/(niter - nburn),
                        _["a_xib_acceptance_total"] = accept_a_xib_tot/niter,
                        _["a_xib_acceptance_pre"] = accept_a_xib_pre/nburn,
                        _["a_xib_acceptance_post"] = accept_a_xib_post/(niter - nburn)),
                        _["tau2"] = List::create(_["f"] = tau2f_save, _["b"] = tau2b_save),
                        _["a_tau"] = List::create(_["f"] = a_tauf_save, _["b"] = a_taub_save),
                        _["a_tau_acceptance"] = List::create(
                          _["a_tauf_acceptance_total"] = accept_a_tauf_tot/niter,
                          _["a_tauf_acceptance_pre"] = accept_a_tauf_pre/nburn,
                          _["a_tauf_acceptance_post"] = accept_a_tauf_post/(niter - nburn),
                          _["a_taub_acceptance_total"] = accept_a_taub_tot/niter,
                          _["a_taub_acceptance_pre"] = accept_a_taub_pre/nburn,
                          _["a_taub_acceptance_post"] = accept_a_taub_post/(niter - nburn)
                          ),
                          _["kappa2"] = List::create(_["f"] = kappa2f_save, _["b"] = kappa2b_save),
                          _["lambda2"] = List::create(_["f"] = lambda2f_save, _["b"] = lambda2b_save),
                            _["success_vals"] = List::create(
                              _["success"] = succesful,
                              _["fail"] = fail,
                              _["fail_iter"] = fail_iter)
                            );
}



