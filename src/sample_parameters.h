#ifndef SAMPLE_PARAMETERS_H
#define SAMPLE_PARAMETERS_H

#include <RcppArmadillo.h>

using namespace Rcpp;

void res_protector(double& x);
void sample_beta_tilde(arma::mat& beta_nc_samp, arma::mat& y, arma::mat& x, arma::colvec& theta_sr, arma::mat& SIGMA, arma::colvec& beta_mean, int N, int n_I, arma::mat S_0, Function Rchol); //
void sample_alpha(arma::vec& alpha_samp, arma::mat& y, arma::mat& x, arma::mat& beta_nc, arma::colvec& tau2, arma::colvec& xi2, arma::mat& SIGMA, arma::vec& a0, int n_I, Function Rchol);
void resample_alpha_diff(arma::vec& alpha_samp, arma::mat& betaenter, arma::vec& theta_sr, arma::vec& beta_mean, arma::mat& beta_diff,  arma::vec& xi2, arma::vec& tau2, int d, int N);
void sample_tau2(arma::vec& tau2_samp, arma::vec& beta_mean, arma::vec& lambda2, arma::vec& a_tau, int d);
void sample_xi2(arma::vec& xi2_samp, arma::vec& theta_sr, arma::vec& kappa2, arma::vec& a_xi, int d);
double sample_kappa2(arma::vec xi2, double a_xi, double d1, double d2, int d);
double sample_lambda2(arma::vec tau2, double a_tau, double e1, double e2, int d);
void sample_sigma2(arma::vec& sig2_samp, arma::vec& y, arma::mat& W, arma::vec& alpha, double c0, double C0, int N);
double sample_C0(arma::vec& sig2, double g0, double c0, double G0);
void get_w(arma::mat& W, arma::mat& x, arma::mat& beta_nc_samp, int N, int n_I);
#endif
