#include <RcppArmadillo.h>
#include <math.h>
#include "sample_parameters.h"
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
void update_beta_mean(arma::vec& beta_mean,
                      arma::vec& beta2_mean,
                      arma::vec& theta_sr,
                      const arma::vec& y,
                      const arma::mat& x,
                      const arma::mat& beta_nc,
                      const arma::vec& sigma2_inv,
                      const arma::vec& tau2_inv){
  // beta_nc dimension: N * d
  int d = x.n_cols;
  arma::mat x_tilde = x % beta_nc;
  arma::vec sigma2_beta_mean(d);
  arma::mat part1 = x_tilde * theta_sr;
  for(int i = 0; i < d; i++){
    arma::mat part2 = x * beta_mean;
    sigma2_beta_mean(i) = 1.0/arma::as_scalar(arma::sum(sigma2_inv%arma::square(x.col(i))) + tau2_inv(i));
    arma::mat tmp = (y - part2 - part1 + x.col(i) * beta_mean(i));
    beta_mean(i) = arma::as_scalar(sigma2_beta_mean(i) * (arma::sum(sigma2_inv % (tmp % x.col(i)))));
    beta2_mean(i) = arma::as_scalar(sigma2_beta_mean(i)) + arma::as_scalar(beta_mean(i)*beta_mean(i));
  }
  //std::for_each(beta_mean.begin(), beta_mean.end(), res_protector);
  //std::for_each(beta2_mean.begin(), beta2_mean.end(), res_protector);
}


void update_theta_sr(arma::vec& beta_mean,
                     arma::vec& theta_sr,
                     arma::vec& theta,
                     const arma::vec& y,
                     const arma::mat& x,
                     const arma::mat& beta_nc,
                     const arma::mat& beta2_nc,
                     const arma::cube& beta_cov_nc,
                     const arma::vec& sigma2_inv,
                     const arma::vec& xi2_inv){
  // beta_nc dimension: N * d
  // beta_cov_nc dimension: N*d*d
  int d = x.n_cols;

  arma::mat x2 = arma::pow(x, 2);
  arma::mat x2_tilde = x2 % beta2_nc;
  arma::vec sigma2_theta_sr_mean(d);
  arma::mat part2 = x * beta_mean;
  for(int i = 0; i < d; i++){
    sigma2_theta_sr_mean(i) = 1.0/arma::as_scalar(arma::sum(sigma2_inv%x2_tilde.col(i)) + xi2_inv(i));
    arma::mat part1 = x % beta_cov_nc.slice(i) * theta_sr;
    arma::mat tmp = y % beta_nc.col(i) - beta_nc.col(i) % part2 - part1;
    theta_sr(i) = arma::as_scalar(sigma2_theta_sr_mean(i) * (arma::sum(sigma2_inv % (tmp % x.col(i)))));
    theta(i) = arma::as_scalar(sigma2_theta_sr_mean(i)) + arma::as_scalar(theta_sr(i)*theta_sr(i));
  }
  //std::for_each(theta_sr.begin(), theta_sr.end(), res_protector);
  //std::for_each(theta.begin(), theta.end(), res_protector);
}

