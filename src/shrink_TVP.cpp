// [[Rcpp::depends(RcppArmadillo, RcppProgress)]]
#include <RcppArmadillo.h>
#include <stochvol.h>
#include <progress.hpp>
#include <math.h>
//#include "sample_beta_McCausland.h"
#include "sample_parameters.h"
#include "MH_step.h"
using namespace Rcpp;

// [[Rcpp::export]]
List do_shrinkTVP(arma::vec y,
                  arma::vec a0,
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
                  bool store_burn,
                  bool sv,
                  double Bsigma_sv,
                  double a0_sv,
                  double b0_sv,
                  double bmu,
                  double Bmu) {

  // progress bar setup
  arma::vec prog_rep_points = arma::round(arma::linspace(0, niter, 50));
  Progress p(50, display_progress);

  // Import Rs chol function
  Environment base = Environment("package:base");
  Function Rchol = base["chol"];

  // Some necessary dimensions
  int N = y.n_elem;
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
  arma::mat yf(N, d, arma::fill::zeros);
  yf.col(0) = y;
  arma::vec y_tmp;
  arma::mat yb(N, d, arma::fill::zeros);
  yb.col(0) = y;

  // generate forward and backward prediction matrix
  arma::mat xf(N, d, arma::fill::zeros);
  arma::mat xb(N, d, arma::fill::zeros);
  arma::mat x_tmp;

  // generate forward and backward PARCOR
  arma::cube betaf_save(N, d, nsave, arma::fill::none); // forward PARCOR coefficients
  arma::cube betab_save(N, d, nsave, arma::fill::none); // backward PARCOR coefficients
  arma::mat beta_nc_tmp;

  arma::cube sig2f_save(N, d, nsave, arma::fill::none);
  arma::cube sig2b_save(N, d, nsave, arma::fill::none);

  arma::mat thetaf_sr_save(d, nsave, arma::fill::none);
  arma::mat thetab_sr_save(d, nsave, arma::fill::none);

  arma::mat betaf_mean_save(d, nsave, arma::fill::none);
  arma::mat betab_mean_save(d, nsave, arma::fill::none);

  arma::mat xi2f_save(d, nsave, arma::fill::none);
  arma::mat xi2b_save(d, nsave, arma::fill::none);

  arma::mat tau2f_save(d, nsave, arma::fill::none);
  arma::mat tau2b_save(d, nsave, arma::fill::none);
  // Storage objects
  //arma::cube beta_save(N+1, d, nsave, arma::fill::none);
  //arma::cube sig2_save(N,1, nsave, arma::fill::none);
  //arma::mat theta_sr_save(d, nsave, arma::fill::none);
  //arma::mat beta_mean_save(d, nsave, arma::fill::none);
  //arma::mat xi2_save(d, nsave, arma::fill::none);
  //arma::mat tau2_save(d, nsave, arma::fill::none);

  // conditional storage objects
  //arma::vec kappa2_save;
  //arma::vec lambda2_save;
  arma::vec kappa2f_save;
  arma::vec kappa2b_save;
  arma::vec lambda2f_save;
  arma::vec lambda2b_save;
  if (learn_kappa2){
    kappa2f_save = arma::vec(nsave, arma::fill::none);
    kappa2b_save = arma::vec(nsave, arma::fill::none);
  }
  if (learn_lambda2){
    lambda2f_save = arma::vec(nsave, arma::fill::none);
    lambda2b_save = arma::vec(nsave, arma::fill::none);
  }

  arma::cube betaf_nc_save;
  arma::cube betab_nc_save;
  if (ret_beta_nc){
    betaf_nc_save = arma::cube(N, d, nsave, arma::fill::none);
    betab_nc_save = arma::cube(N, d, nsave, arma::fill::none);
  }

  arma::vec a_xif_save;
  arma::vec a_tauf_save;

  arma::vec a_xib_save;
  arma::vec a_taub_save;

  int accept_a_xif_tot = 0;
  int accept_a_xif_pre = 0;
  int accept_a_xif_post = 0;

  int accept_a_tauf_tot = 0;
  int accept_a_tauf_pre = 0;
  int accept_a_tauf_post = 0;

  int accept_a_xib_tot = 0;
  int accept_a_xib_pre = 0;
  int accept_a_xib_post = 0;

  int accept_a_taub_tot = 0;
  int accept_a_taub_pre = 0;
  int accept_a_taub_post = 0;

  if (learn_a_xi){
    a_xif_save = arma::vec(nsave, arma::fill::none);
    a_xib_save = arma::vec(nsave, arma::fill::none);
  }
  if (learn_a_tau){
    a_tauf_save = arma::vec(nsave, arma::fill::none);
    a_taub_save = arma::vec(nsave, arma::fill::none);
  }

  arma::mat C0f_save;
  arma::mat svf_mu_save;
  arma::mat svf_phi_save;
  arma::mat svf_sigma2_save;

  arma::mat C0b_save;
  arma::mat svb_mu_save;
  arma::mat svb_phi_save;
  arma::mat svb_sigma2_save;

  if (sv == false){
    C0f_save = arma::mat(d, nsave, arma::fill::none);
    C0b_save = arma::mat(d, nsave, arma::fill::none);
  } else {
    svf_mu_save = arma::mat(nsave, d, arma::fill::none);
    svf_phi_save = arma::mat(nsave, d, arma::fill::none);
    svf_sigma2_save = arma::mat(nsave, d, arma::fill::none);

    svb_mu_save = arma::mat(nsave, d, arma::fill::none);
    svb_phi_save = arma::mat(nsave, d, arma::fill::none);
    svb_sigma2_save = arma::mat(nsave, d, arma::fill::none);
  }

  // Initial values and objects
  arma::mat x_tmp_tilde;
  arma::mat W_tmp;
  arma::mat beta_nc_tmp_tilde;
  arma::mat betaenter;
  arma::mat beta_diff_pre;
  arma::mat beta_diff;

  arma::mat betaf_nc_samp(d, N, arma::fill::zeros);
  arma::mat betab_nc_samp(d, N, arma::fill::zeros);


  arma::vec betaf_mean_samp(d);
  betaf_mean_samp.fill(0.1);

  arma::vec betab_mean_samp(d);
  betab_mean_samp.fill(0.1);

  arma::colvec beta_mean_tmp(1);
  beta_mean_tmp.fill(0.1);

  arma::vec thetaf_sr_samp(d);
  thetaf_sr_samp.fill(0.2);

  arma::vec thetab_sr_samp(d);
  thetab_sr_samp.fill(0.2);

  arma::colvec theta_sr_tmp(1);
  theta_sr_tmp.fill(0.2);

  arma::vec tau2f_samp(d);
  tau2f_samp.fill(0.1);

  arma::vec tau2b_samp(d);
  tau2b_samp.fill(0.1);

  arma::vec tau2_tmp(1);
  tau2_tmp.fill(0.1);

  arma::vec xi2f_samp(d);
  xi2f_samp.fill(0.1);

  arma::vec xi2b_samp(d);
  xi2b_samp.fill(0.1);

  arma::vec xi2_tmp(1);
  xi2_tmp.fill(0.1);

  arma::vec xi_tau_tmp = arma::join_cols(xi2_tmp, tau2_tmp);

  double kappa2f_samp = 20;

  double lambda2f_samp = 0.1;

  double a_xif_samp = 0.1;

  double a_tauf_samp = 0.1;

  arma::vec kappa2f_lambda_samp = {kappa2f_samp, lambda2f_samp};

  arma::mat hf_samp(N, d, arma::fill::zeros);

  //arma::vec alpha_samp(2*d, arma::fill::ones);
  //arma::vec alpha_samp = arma::join_cols(beta_mean_tmp, theta_sr_tmp);
  arma::vec alpha_tmp(2, arma::fill::ones);

  arma::mat sig2f_samp = arma::exp(hf_samp);
  arma::vec sig2_tmp;
  arma::vec C0f_samp(d, arma::fill::zeros);
  double C0_tmp = 1;

  double kappa2b_samp = 20;

  double lambda2b_samp = 0.1;

  double a_xib_samp = 0.1;

  double a_taub_samp = 0.1;

  arma::vec kappa2b_lambda_samp = {kappa2b_samp, lambda2b_samp};

  arma::mat hb_samp(N, d, arma::fill::zeros);

  //arma::vec alphab_samp(2*d, arma::fill::ones);

  arma::mat sig2b_samp = arma::exp(hb_samp);

  arma::vec C0b_samp(d, arma::fill::zeros);
  //double C0_tmp = 1;

  // SV quantities
  arma::vec svf_para = {-10, 0.5, 1};
  arma::vec svb_para = {-10, 0.5, 1};
  arma::mat mixprob(10, N);
  arma::vec mixprob_vec(mixprob.begin(), mixprob.n_elem, false);
  arma::ivec r(N);
  double h0 = -10;
  double B011inv         = 1e-8;
  double B022inv         = 1e-12;
  bool Gammaprior        = true;
  double MHcontrol       = -1;
  int parameterization   = 3;
  bool centered_baseline = parameterization % 2; // 1 for C, 0 for NC baseline
  int MHsteps = 2;
  bool dontupdatemu = 0;
  double cT = N/2.0;
  double C0_sv = 1.5*Bsigma_sv;
  bool truncnormal = false;
  double priorlatent0 = -1;

  // Values for LPDS
  arma::cube m_N_save(d, 1, nsave);
  arma::cube chol_C_N_inv_save(d, d, nsave);
  arma::vec m_N_samp;
  arma::mat chol_C_N_inv_samp;

  // Override inital values with user specified fixed values
  if (learn_kappa2 == false){
    kappa2f_samp = kappa2;
    kappa2b_samp = kappa2;
  }
  if (learn_lambda2 == false){
    lambda2f_samp = lambda2;
    lambda2b_samp = lambda2;
  }

  if (!learn_a_xi){
    a_xif_samp = a_xi;
    a_xib_samp = a_xi;
  }
  if (!learn_a_tau){
    a_tauf_samp = a_tau;
    a_taub_samp = a_tau;
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
      // forward
      //-----------------------------------
      n_1 = m + 1; // forward index
      n_T = N;     // forward index
      //int nb_1 = 1;
      //int nb_T = N - m;
      N_m = n_T - n_1 + 1; // time point at stage m

      y_tmp = yf(arma::span(n_1-1, n_T-1), m-1); // response for forward part

      sig2_tmp = sig2f_samp(arma::span(n_1-1, n_T-1), m-1);

      beta_nc_tmp = arma::mat(1, N_m+1, arma::fill::zeros);

      x_tmp = arma::mat(N_m, 1, arma::fill::zeros); // predictor for forward part

      x_tmp.col(0) = yb(arma::span(n_1-m-1, n_T-m-1), m-1);
      beta_mean_tmp = betaf_mean_samp(m-1);
      theta_sr_tmp = thetaf_sr_samp(m-1);
      tau2_tmp = tau2f_samp(m-1);
      xi2_tmp = xi2f_samp(m-1);
      // step a)
      // sample time varying beta.tilde parameters (NC parametrization)
      try {
        sample_beta_tilde(beta_nc_tmp, y_tmp, x_tmp, theta_sr_tmp, sig2_tmp, beta_mean_tmp, N_m, 1, Rchol);
        if(m < d){
          yf(arma::span(n_1-1, n_T-1), m) = y_tmp;
        }

        //sample_beta_tilde(beta_nc_tmp, yb_tmp, xf, thetab_sr_samp, sig2b_samp, betab_mean_samp, N, d, m, nb_1, nb_T);
        //yb(arma::span(nb_1-1, nb_T-1)) = yb_tmp;
        //sample_beta_McCausland(beta_nc_samp, y, x, theta_sr_samp, sig2_samp, betaf_mean_samp, m_N_samp, chol_C_N_inv_samp, true, N, d, Rchol);
      } catch (...){
        beta_nc_tmp.fill(nanl(""));
        if (succesful == true){
          fail = "sample beta_nc";
          fail_iter = j + 1;
          succesful = false;
        }
      }

      // step b)
      // sample alpha
      //arma::cout << "size of x_tmp: " << arma::size(x_tmp) << arma::endl;
      //arma::cout << "size of beta_nc_tmp.cols" << arma::size(beta_nc_tmp.cols(1, N_m)) << arma::endl;
      arma::mat x_tmp_tilde = x_tmp % (beta_nc_tmp.cols(1,N_m)).t();
      arma::mat W_tmp = arma::join_rows(x_tmp, x_tmp_tilde);
      y_tmp = yf(arma::span(n_1-1, n_T-1), m-1);
      try {
        sample_alpha(alpha_tmp, y_tmp, x_tmp, W_tmp, tau2_tmp, xi2_tmp, sig2_tmp, a0, 1, Rchol);
      } catch(...){
        alpha_tmp.fill(nanl(""));
        if (succesful == true){
          fail = "sample alpha";
          fail_iter = j + 1;
          succesful = false;
        }
      }

      // Weave back into centered parameterization
      beta_mean_tmp = alpha_tmp(0);
      theta_sr_tmp = alpha_tmp(1);
      beta_nc_tmp_tilde = beta_nc_tmp.each_col() % theta_sr_tmp;
      betaenter = beta_nc_tmp_tilde.each_col() + beta_mean_tmp;

      // Difference beta outside of function (for numerical stability)
      beta_diff_pre = arma::diff(beta_nc_tmp, 1, 1);
      beta_diff =  beta_diff_pre.each_col() % theta_sr_tmp;
      //arma::cout << "size of beta_diff_pre: " << arma::size(beta_diff_pre) << arma::endl;
      //arma::cout << "size of beta_diff" << arma::size(beta_diff) << arma::endl;
      // step c)
      // resample alpha
      try {
        resample_alpha_diff(alpha_tmp, betaenter, theta_sr_tmp, beta_mean_tmp, beta_diff, xi2_tmp, tau2_tmp, 1, N_m);
      } catch(...) {
        alpha_tmp.fill(nanl(""));
        if (succesful == true){
          fail = "resample alpha";
          fail_iter = j + 1;
          succesful = false;
        }
      }

      // Calculate NC betas with new alpha
      beta_mean_tmp = alpha_tmp(0);
      theta_sr_tmp = alpha_tmp(1);
      beta_nc_tmp = betaenter.each_col() - beta_mean_tmp;
      //arma::cout << "size of beta_nc_tmp: " << arma::size(beta_nc_tmp) << arma::endl;
      //arma::cout << "size of theta_sr_tmp" << arma::size(theta_sr_tmp) << arma::endl;
      beta_nc_tmp.each_col() /= theta_sr_tmp;
      //arma::cout << "size of beta_nc_tmp: " << arma::size(beta_nc_tmp) << arma::endl;
      //arma::cout << "size of theta_sr_tmp" << arma::size(theta_sr_tmp) << arma::endl;
      betaf_nc_samp(m-1, arma::span(n_1 - 1, n_T-1)) = beta_nc_tmp.cols(1, N_m);
      betaf_mean_samp(m-1) = alpha_tmp(0);
      thetaf_sr_samp(m-1) = alpha_tmp(1);
      x_tmp_tilde = x_tmp % (beta_nc_tmp.cols(1,N_m)).t();
      W_tmp = arma::join_rows(x_tmp, x_tmp_tilde);


      //arma::cout << "size of beta_nc_tmp: " << arma::size(beta_nc_tmp) << arma::endl;
      //arma::cout << "size of theta_sr_tmp" << arma::size(theta_sr_tmp) << arma::endl;
      // step e)
      // sample tau2 and xi2
      try {
        sample_tau2(tau2_tmp, beta_mean_tmp, lambda2f_samp, a_tauf_samp, 1);
        tau2f_samp(m-1) = arma::as_scalar(tau2_tmp);
      } catch(...) {
        tau2_tmp.fill(nanl(""));
        if (succesful == true){
          fail = "sample tau2";
          fail_iter = j + 1;
          succesful = false;
        }
      }
      try {
        sample_xi2(xi2_tmp, theta_sr_tmp, kappa2f_samp, a_xif_samp, 1);
        xi2f_samp(m-1) = arma::as_scalar(xi2_tmp);
      } catch(...) {
        xi2_tmp.fill(nanl(""));
        if (succesful == true){
          fail = "sample xi2";
          fail_iter = j + 1;
          succesful = false;
        }
      }
      //arma::cout << "size of beta_nc_tmp: " << arma::size(beta_nc_tmp) << arma::endl;
      //arma::cout << "size of theta_sr_tmp" << arma::size(theta_sr_tmp) << arma::endl;
      // step f)
      // sample sigma2 from homoscedastic or SV case
      try {
        if (sv){
          arma::vec datastand = arma::log(arma::square(y_tmp - x_tmp * beta_mean_tmp - (x_tmp % beta_nc_tmp.cols(1,N).t()) * theta_sr_tmp));

          arma::vec cur_h = arma::log(sig2_tmp);
          stochvol::update_sv(datastand, svf_para, cur_h, h0, mixprob_vec, r, centered_baseline, C0_sv, cT,
                              Bsigma_sv, a0_sv, b0_sv, bmu, Bmu, B011inv, B022inv, Gammaprior,
                              truncnormal, MHcontrol, MHsteps, parameterization, dontupdatemu, priorlatent0);

          sig2f_samp(arma::span(n_1-1, n_T-1), m-1) = arma::exp(cur_h);
        } else {
          sample_sigma2(sig2_tmp, y_tmp, W_tmp, alpha_tmp, c0, C0_tmp, N);
        }
        sig2f_samp(arma::span(n_1-1, n_T-1), m-1) = sig2_tmp;
      } catch(...) {
        sig2_tmp.fill(nanl(""));
        if (succesful == true){
          fail = "sample sigma2";
          fail_iter = j + 1;
          succesful = false;
        }
      }


      if(sv == false){
        try {
          C0_tmp = sample_C0(sig2_tmp, g0, c0, G0);
        } catch(...) {
          C0_tmp = nanl("");
          if (succesful == true){
            fail = "sample C0";
            fail_iter = j + 1;
            succesful = false;
          }
        }
        C0f_samp(m-1) = arma::as_scalar(C0_tmp);
      }
      //arma::cout << "size of beta_nc_tmp: " << arma::size(beta_nc_tmp) << arma::endl;
      //arma::cout << "size of theta_sr_tmp" << arma::size(theta_sr_tmp) << arma::endl;



      // backward
      //-----------------------------------
      n_1 = 1;    // backward index
      n_T = N - m;    // backward index
      N_m = n_T - n_1 + 1;

      y_tmp = yb(arma::span(n_1-1, n_T-1), m-1); // response

      sig2_tmp = sig2b_samp(arma::span(n_1-1, n_T-1), m-1);

      beta_nc_tmp = arma::mat(1, N_m+1, arma::fill::zeros);
      //betab_nc_tmp = arma::mat(1, N_m+1, arma::fill::zeros);

      x_tmp = arma::mat(N_m, 1, arma::fill::zeros);
      //xb_tmp = arma::mat(N_m, 1, arma::fill::zeros);

      //x_tmp.col(0) = yf(arma::span(n_1+m-1, n_T+m-1), m-1);
      x_tmp.col(0) = yf(arma::span(n_1+m-1, n_T+m-1), m-1);
      beta_mean_tmp = betab_mean_samp(m-1);
      theta_sr_tmp = thetab_sr_samp(m-1);
      tau2_tmp = tau2b_samp(m-1);
      xi2_tmp = xi2b_samp(m-1);
      // step a)
      // sample time varying beta.tilde parameters (NC parametrization)
      try {
        sample_beta_tilde(beta_nc_tmp, y_tmp, x_tmp, theta_sr_tmp, sig2_tmp, beta_mean_tmp, N_m, 1, Rchol);
        if(m < d){
          yb(arma::span(n_1-1, n_T-1), m) = y_tmp;
        }

        //sample_beta_tilde(beta_nc_tmp, yb_tmp, xf, thetab_sr_samp, sig2b_samp, betab_mean_samp, N, d, m, nb_1, nb_T);
        //yb(arma::span(nb_1-1, nb_T-1)) = yb_tmp;
        //sample_beta_McCausland(beta_nc_samp, y, x, theta_sr_samp, sig2_samp, betaf_mean_samp, m_N_samp, chol_C_N_inv_samp, true, N, d, Rchol);
      } catch (...){
        beta_nc_tmp.fill(nanl(""));
        if (succesful == true){
          fail = "sample beta_nc";
          fail_iter = j + 1;
          succesful = false;
        }
      }

      // step b)
      // sample alpha
      x_tmp_tilde = x_tmp % (beta_nc_tmp.cols(1,N_m)).t();
      W_tmp = arma::join_rows(x_tmp, x_tmp_tilde);
      y_tmp = yb(arma::span(n_1-1, n_T-1), m-1);
      try {
        sample_alpha(alpha_tmp, y_tmp, x_tmp, W_tmp, tau2_tmp, xi2_tmp, sig2_tmp, a0, 1, Rchol);
      } catch(...){
        alpha_tmp.fill(nanl(""));
        if (succesful == true){
          fail = "sample alpha";
          fail_iter = j + 1;
          succesful = false;
        }
      }

      // Weave back into centered parameterization
      beta_mean_tmp = alpha_tmp(0);
      theta_sr_tmp = alpha_tmp(1);
      beta_nc_tmp_tilde = beta_nc_tmp.each_col() % theta_sr_tmp;
      betaenter = beta_nc_tmp_tilde.each_col() + beta_mean_tmp;
      //arma::cout << "size of beta_nc_tmp: " << arma::size(beta_nc_tmp) << arma::endl;
      //arma::cout << "size of theta_sr_tmp" << arma::size(theta_sr_tmp) << arma::endl;
      // Difference beta outside of function (for numerical stability)
      beta_diff_pre = arma::diff(beta_nc_tmp, 1, 1);

      //arma::cout << "size of beta_diff_pre" << arma::size(beta_diff_pre) << arma::endl;
      beta_diff =  beta_diff_pre.each_col() % theta_sr_tmp;

      // step c)
      // resample alpha
      try {
        resample_alpha_diff(alpha_tmp, betaenter, theta_sr_tmp, beta_mean_tmp, beta_diff, xi2_tmp, tau2_tmp, 1, N_m);
      } catch(...) {
        alpha_tmp.fill(nanl(""));
        if (succesful == true){
          fail = "resample alpha";
          fail_iter = j + 1;
          succesful = false;
        }
      }

      // Calculate NC betas with new alpha
      beta_mean_tmp = alpha_tmp(0);
      theta_sr_tmp = alpha_tmp(1);
      //arma::cout << "size of beta_mean_tmp: " << arma::size(beta_mean_tmp) << arma::endl;
      //arma::cout << "size of betaenter" << arma::size(betaenter) << arma::endl;
      beta_nc_tmp = betaenter.each_col() - beta_mean_tmp;
      beta_nc_tmp.each_col() /= theta_sr_tmp;
      x_tmp_tilde = x_tmp % (beta_nc_tmp.cols(1,N_m)).t();
      W_tmp = arma::join_rows(x_tmp, x_tmp_tilde);

      //arma::cout << "size of beta_nc_tmp: " << arma::size(beta_nc_tmp) << arma::endl;
      //arma::cout << "size of theta_sr_tmp" << arma::size(theta_sr_tmp) << arma::endl;
      betab_nc_samp(m-1, arma::span(n_1 - 1, n_T-1)) = beta_nc_tmp.cols(1, N_m);
      betab_mean_samp(m-1) = alpha_tmp(0);
      thetab_sr_samp(m-1) = alpha_tmp(1);
      // step e)
      // sample tau2 and xi2
      try {
        sample_tau2(tau2_tmp, beta_mean_tmp, lambda2b_samp, a_taub_samp, 1);
        tau2b_samp(m-1) = arma::as_scalar(tau2_tmp);
      } catch(...) {
        tau2_tmp.fill(nanl(""));
        if (succesful == true){
          fail = "sample tau2";
          fail_iter = j + 1;
          succesful = false;
        }
      }
      try {
        sample_xi2(xi2_tmp, theta_sr_tmp, kappa2b_samp, a_xib_samp, 1);
        xi2b_samp(m-1) = arma::as_scalar(xi2_tmp);
      } catch(...) {
        xi2_tmp.fill(nanl(""));
        if (succesful == true){
          fail = "sample xi2";
          fail_iter = j + 1;
          succesful = false;
        }
      }

      // step f)
      // sample sigma2 from homoscedastic or SV case
      try {
        if (sv){
          arma::vec datastand = arma::log(arma::square(y_tmp - x_tmp * beta_mean_tmp - (x_tmp % beta_nc_tmp.cols(1,N_m).t()) * theta_sr_tmp));

          arma::vec cur_h = arma::log(sig2_tmp);
          stochvol::update_sv(datastand, svb_para, cur_h, h0, mixprob_vec, r, centered_baseline, C0_sv, cT,
                              Bsigma_sv, a0_sv, b0_sv, bmu, Bmu, B011inv, B022inv, Gammaprior,
                              truncnormal, MHcontrol, MHsteps, parameterization, dontupdatemu, priorlatent0);

          sig2b_samp = arma::exp(cur_h);
        } else {
          sample_sigma2(sig2_tmp, y_tmp, W_tmp, alpha_tmp, c0, C0_tmp, N_m);
        }
        sig2b_samp(arma::span(n_1-1, n_T-1), m-1) = sig2_tmp;
      } catch(...) {
        sig2_tmp.fill(nanl(""));
        if (succesful == true){
          fail = "sample sigma2";
          fail_iter = j + 1;
          succesful = false;
        }
      }


      if(sv == false){
        try {
          C0_tmp = sample_C0(sig2_tmp, g0, c0, G0);
        } catch(...) {
          C0_tmp = nanl("");
          if (succesful == true){
            fail = "sample C0";
            fail_iter = j + 1;
            succesful = false;
          }
        }
        C0b_samp(m-1) = arma::as_scalar(C0_tmp);
      }

    }


    // step d)
    // sample a_xi and a_tau with MH
    if (learn_a_xi){
      double before = a_xif_samp;
      try {
        a_xif_samp = MH_step(a_xif_samp, c_tuning_par_xi, d, kappa2f_samp, thetaf_sr_samp, b_xi , nu_xi, d1, d2);
      } catch(...) {
        a_xif_samp = nanl("");
        if (succesful == true){
          fail = "sample a_xi";
          fail_iter = j + 1;
          succesful = false;
        }
      }

      if (before != a_xif_samp){
        accept_a_xif_tot += 1;
        if (j < nburn){
          accept_a_xif_pre += 1;
        } else {
          accept_a_xif_post += 1;
        }
      }
    }


    if (learn_a_tau){
      double before = a_tauf_samp;
      try {
        a_tauf_samp = MH_step(a_tauf_samp, c_tuning_par_tau, d, lambda2f_samp, betaf_mean_samp, b_tau , nu_tau, e1, e2);
      } catch(...){
        a_tauf_samp = nanl("");
        if (succesful == true){
          fail = "sample a_tau";
          fail_iter = j + 1;
          succesful = false;
        }
      }

      if (before != a_tauf_samp){
        accept_a_tauf_tot += 1;
        if (j < nburn){
          accept_a_tauf_pre += 1;
        } else {
          accept_a_tauf_post += 1;
        }
      }
    }


    // sample kappa2 and lambda2, if the user specified it
    if (learn_kappa2){
      try {
        kappa2f_samp = sample_kappa2(xi2f_samp, a_xif_samp, d1, d2, d);
      } catch (...) {
        kappa2f_samp = nanl("");
        if (succesful == true){
          fail = "sample kappa2";
          fail_iter = j + 1;
          succesful = false;
        }
      }
    }

    if (learn_lambda2){
      try {
        lambda2f_samp = sample_lambda2(tau2f_samp, a_tauf_samp, e1, e2, d);
      } catch (...) {
        lambda2f_samp = nanl("");
        if (succesful == true){
          fail = "sample lambda2";
          fail_iter = j + 1;
          succesful = false;
        }
      }
    }

    // step d)
    // sample a_xi and a_tau with MH
    if (learn_a_xi){
      double before = a_xib_samp;
      try {
        a_xib_samp = MH_step(a_xib_samp, c_tuning_par_xi, d, kappa2b_samp, thetab_sr_samp, b_xi , nu_xi, d1, d2);
      } catch(...) {
        a_xib_samp = nanl("");
        if (succesful == true){
          fail = "sample a_xi";
          fail_iter = j + 1;
          succesful = false;
        }
      }

      if (before != a_xib_samp){
        accept_a_xib_tot += 1;
        if (j < nburn){
          accept_a_xib_pre += 1;
        } else {
          accept_a_xib_post += 1;
        }
      }
    }


    if (learn_a_tau){
      double before = a_taub_samp;
      try {
        a_taub_samp = MH_step(a_taub_samp, c_tuning_par_tau, d, lambda2b_samp, betab_mean_samp, b_tau , nu_tau, e1, e2);
      } catch(...){
        a_taub_samp = nanl("");
        if (succesful == true){
          fail = "sample a_tau";
          fail_iter = j + 1;
          succesful = false;
        }
      }

      if (before != a_taub_samp){
        accept_a_taub_tot += 1;
        if (j < nburn){
          accept_a_taub_pre += 1;
        } else {
          accept_a_taub_post += 1;
        }
      }
    }


    // sample kappa2 and lambda2, if the user specified it
    if (learn_kappa2){
      try {
        kappa2b_samp = sample_kappa2(xi2b_samp, a_xib_samp, d1, d2, d);
      } catch (...) {
        kappa2b_samp = nanl("");
        if (succesful == true){
          fail = "sample kappa2";
          fail_iter = j + 1;
          succesful = false;
        }
      }
    }

    if (learn_lambda2){
      try {
        lambda2b_samp = sample_lambda2(tau2b_samp, a_taub_samp, e1, e2, d);
      } catch (...) {
        lambda2b_samp = nanl("");
        if (succesful == true){
          fail = "sample lambda2";
          fail_iter = j + 1;
          succesful = false;
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
      arma::mat betaf =  (betaf_nc_samp.each_col() % thetaf_sr_samp).each_col() + betaf_mean_samp;
      arma::mat betab =  (betab_nc_samp.each_col() % thetab_sr_samp).each_col() + betab_mean_samp;

      //arma::cout << "size of sig2f: " << arma::size(sig2f_samp) << arma::endl;
      //arma::cout << "size of sig2b:" << arma::size(sig2b_samp) << arma::endl;

      sig2f_save.slice((post_j-1)/nthin) = sig2f_samp;
      sig2b_save.slice((post_j-1)/nthin) = sig2b_samp;

      thetaf_sr_save.col((post_j-1)/nthin) = thetaf_sr_samp;
      thetab_sr_save.col((post_j-1)/nthin) = thetab_sr_samp;

      betaf_mean_save.col((post_j-1)/nthin) = betaf_mean_samp;
      betab_mean_save.col((post_j-1)/nthin) = betab_mean_samp;

      //arma::cout << "size of betaf: " << arma::size(betaf) << arma::endl;
      //arma::cout << "size of betab:" << arma::size(betab) << arma::endl;
      //arma::cout << "size of betaf_save: " << arma::size(betaf_save) << arma::endl;
      //arma::cout << "size of betab_save:" << arma::size(betab_save) << arma::endl;
      betaf_save.slice((post_j-1)/nthin) = betaf.t();
      betab_save.slice((post_j-1)/nthin) = betab.t();

      xi2f_save.col((post_j-1)/nthin) = xi2f_samp;
      xi2b_save.col((post_j-1)/nthin) = xi2b_samp;

      tau2f_save.col((post_j-1)/nthin) = tau2f_samp;
      tau2b_save.col((post_j-1)/nthin) = tau2b_samp;
      //m_N_save.slice((post_j-1)/nthin) = m_N_samp;
      //chol_C_N_inv_save.slice((post_j-1)/nthin) = chol_C_N_inv_samp;

      //conditional storing
      //if (ret_beta_nc){
      //  beta_nc_save.slice((post_j-1)/nthin) = beta_nc_samp.t();
      //}

      if (learn_kappa2){
        kappa2f_save((post_j-1)/nthin) = kappa2f_samp;
        kappa2b_save((post_j-1)/nthin) = kappa2b_samp;
      }
      if (learn_lambda2){
        lambda2f_save((post_j-1)/nthin) = lambda2f_samp;
        lambda2b_save((post_j-1)/nthin) = lambda2b_samp;
      }

      if (learn_a_xi){
        a_xif_save((post_j-1)/nthin) = a_xif_samp;
        a_xib_save((post_j-1)/nthin) = a_xib_samp;
      }

      if (learn_a_tau){
        a_tauf_save((post_j-1)/nthin) = a_tauf_samp;
        a_taub_save((post_j-1)/nthin) = a_taub_samp;
      }

      if (sv == false){
        C0f_save.col((post_j-1)/nthin) = C0f_samp;
        C0b_save.col((post_j-1)/nthin) = C0b_samp;
      } else {
        svf_mu_save((post_j-1)/nthin) = svf_para(0);
        svf_phi_save((post_j-1)/nthin) = svf_para(1);
        svf_sigma2_save((post_j-1)/nthin) = svf_para(2);

        svb_mu_save((post_j-1)/nthin) = svb_para(0);
        svb_phi_save((post_j-1)/nthin) = svb_para(1);
        svb_sigma2_save((post_j-1)/nthin) = svb_para(2);
      }
    }

    // Random sign switch
    for (int i = 0; i < d; i++){
      if(R::runif(0,1) > 0.5){
        thetaf_sr_samp(i) = -thetaf_sr_samp(i);
      }

      if(R::runif(0, 1) > 0.5){
        thetab_sr_samp(i) = -thetab_sr_samp(i);
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
  return List::create(_["sigma2"] = List::create(_["f"] = sig2f_save, _["b"] = sig2b_save),
                      _["theta_sr"] = List::create(_["f"] = thetaf_sr_save.t(), _["b"] = thetab_sr_save.t()),
                      _["beta_mean"] = List::create(_["f"] = betaf_mean_save.t(), _["b"] = betab_mean_save.t()),
                      _["beta_nc"] = List::create(_["f"] = betaf_nc_save, _["b"] = betab_nc_save),
                      _["beta"] = List::create(_["f"] = betaf_save, _["b"] = betab_save),
                      _["xi2"] = List::create(_["f"] = xi2f_save.t(), _["b"] = xi2b_save.t()),
                      _["a_xi"] = List::create(_["f"] = a_xif_save, _["b"] = a_xib_save),
                      _["a_xi_acceptance"] = List::create(
                        _["a_xif_acceptance_total"] = (double)accept_a_xif_tot/niter,
                        _["a_xif_acceptance_pre"] = (double)accept_a_xif_pre/nburn,
                        _["a_xif_acceptance_post"] = (double)accept_a_xif_post/(niter - nburn),
                        _["a_xib_acceptance_total"] = (double)accept_a_xib_tot/niter,
                        _["a_xib_acceptance_pre"] = (double)accept_a_xib_pre/nburn,
                        _["a_xib_acceptance_post"] = (double)accept_a_xib_post/(niter - nburn)),
                        _["tau2"] = List::create(_["f"] = tau2f_save.t(), _["b"] = tau2b_save.t()),
                        _["a_tau"] = List::create(_["f"] = a_tauf_save, _["b"] = a_taub_save),
                        _["a_tau_acceptance"] = List::create(
                          _["a_tauf_acceptance_total"] = (double)accept_a_tauf_tot/niter,
                          _["a_tauf_acceptance_pre"] = (double)accept_a_tauf_pre/nburn,
                          _["a_tauf_acceptance_post"] = (double)accept_a_tauf_post/(niter - nburn),
                          _["a_taub_acceptance_total"] = (double)accept_a_taub_tot/niter,
                          _["a_taub_acceptance_pre"] = (double)accept_a_taub_pre/nburn,
                          _["a_taub_acceptance_post"] = (double)accept_a_taub_post/(niter - nburn)
                          ),
                          _["kappa2"] = List::create(_["f"] = kappa2f_save, _["b"] = kappa2b_save),
                          _["lambda2"] = List::create(_["f"] = lambda2f_save, _["b"] = lambda2b_save),
                          _["C0"] = List::create(_["f"] = C0f_save, _["b"] = C0b_save),
                          _["sv_mu"] = List::create(_["f"] = svf_mu_save, _["b"] = svb_mu_save),
                          _["sv_phi"] = List::create(_["f"] = svf_phi_save, _["b"] = svb_phi_save),
                          _["sv_sigma2"] = List::create(_["f"] = svf_sigma2_save, _["b"] = svb_sigma2_save),
                          _["LPDS_comp"] = List::create(
                            _["m_N"] = m_N_save,
                            _["chol_C_N_inv"] = chol_C_N_inv_save),
                            _["success_vals"] = List::create(
                              _["success"] = succesful,
                              _["fail"] = fail,
                              _["fail_iter"] = fail_iter)
                            );
}



