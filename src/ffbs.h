#ifndef UPDATE_BETA_TILDE_H
#define UPDATE_BETA_TILDE_H

void update_beta_tilde(arma::mat& beta_nc,
                       arma::mat& beta2_nc,
                       arma::cube& beta_nc_cov,
                       arma::vec& y, arma::mat& x,
                       const arma::vec& theta_sr,
                       const arma::vec& beta_mean, const int N,
                       const double S_0,
                       arma::vec& St);

void update_prediction_error(arma::vec& y, arma::mat& x, arma::mat& beta_nc,
                                  const arma::vec& theta_sr,
                                  const arma::vec& beta_mean, const int N);
#endif
