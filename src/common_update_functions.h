
#ifndef COMMON_UPDATE_FUNCTIONS_H
#define COMMON_UPDATE_FUNCTIONS_H

void update_beta_mean(arma::vec& beta_mean,
                      arma::vec& beta2_mean,
                      arma::vec& theta_sr,
                      const arma::vec& y,
                      const arma::mat& x,
                      const arma::mat& beta_nc,
                      const arma::vec& sigma2_inv,
                      const arma::vec& tau2_inv);

void update_theta_sr(arma::vec& beta_mean,
                     arma::vec& theta_sr,
                     arma::vec& theta,
                     const arma::vec& y,
                     const arma::mat& x,
                     const arma::mat& beta_nc,
                     const arma::mat& beta2_nc,
                     const arma::cube& beta_cov_nc,
                     const arma::vec& sigma2_inv,
                     const arma::vec& xi2_inv);


#endif
