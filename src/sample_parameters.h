#ifndef SAMPLE_PARAMETERS_H
#define SAMPLE_PARAMETERS_H

#include <RcppArmadillo.h>

using namespace Rcpp;

void res_protector(double& x);
void sample_beta_tilde(arma::mat& beta_nc_samp, arma::vec& y, arma::mat& x, arma::vec& theta_sr, arma::vec& beta_mean, int N, int n_I, double S_0, arma::vec& St); //
void sample_alpha(arma::vec& alpha_samp, arma::vec& y, arma::mat& x, arma::mat& x_tilde, arma::colvec& tau2, arma::colvec& xi2, arma::vec& SIGMA, int n_I, Function Rchol);
void resample_alpha_diff(arma::vec& alpha_samp, arma::mat betaenter, arma::vec& theta_sr, arma::vec& beta_mean, arma::mat beta_diff,  arma::vec& xi2, arma::vec& tau2, int d, int N);
void sample_local_shrink(arma::vec& local_shrink, const arma::vec& param_vec, double global_shrink, double a);
double sample_global_shrink(const arma::vec& prior_param, double a, double hyper1, double hyper2);

#endif
