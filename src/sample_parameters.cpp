// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "do_rgig1.h"
#include <math.h>
using namespace Rcpp;


void res_protector(double& x){
  if (std::abs(x) < DBL_MIN * std::pow(10, 10)){
    double sign = std::copysign(1, x);
    x = DBL_MIN * std::pow(10, 10) * sign;
  }

  if (std::abs(x) > DBL_MAX * std::pow(10, -30)){
    double sign = std::copysign(1, x);
    x = DBL_MAX * std::pow(10, -30) * sign;
  }

  if (std::isnan(x)){
    throw 1;
  }
}


void sample_beta_tilde(arma::mat& beta_nc_samp, arma::vec& y,
                       arma::mat& x, arma::vec& theta_sr,
                       arma::vec& beta_mean,
                       int N, int n_I, double S_0,
                       arma::vec& St){
  // initialization
  int d = x.n_cols;

  arma::mat mt(d, N+1, arma::fill::zeros);
  arma::cube Rt(d, d, N+1, arma::fill::zeros);
  arma::cube Ct(d, d, N+1, arma::fill::zeros);
  arma::mat mT(d, N+1, arma::fill::zeros);
  arma::cube CT(d, d, N+1, arma::fill::zeros);

  arma::vec St_tmp(N+1, arma::fill::zeros);

  double St_sq;
  double S_comp = 0.0;
  double n_0 = 1.0;

  arma::vec yt_star = y - x*beta_mean;
  arma::colvec theta = arma::pow(theta_sr, 2);
  arma::mat theta_sr_diag = arma::diagmat(theta_sr);
  arma::mat Ft = x*theta_sr_diag;


  arma::mat I_d = arma::eye(d, d);
  arma::vec ft(N, arma::fill::zeros);

  double Qt;
  double Qt_inv_sq;
  arma::vec At;
  double et;

  Ct.slice(0) = I_d;
  St_tmp(0) = S_0;

  arma::mat Bt;
  arma::mat Rtp1_inv;

  arma::mat L_upper;
  for(int t = 1; t < N+1; t++){
    Rt.slice(t) = Ct.slice(t-1) + I_d;
    Rt.slice(t) = 0.5*Rt.slice(t) + 0.5*arma::trans(Rt.slice(t));
    ft(t-1) = arma::as_scalar(Ft.row(t-1)*mt.col(t-1));
    Qt = arma::as_scalar(Ft.row(t-1) * Rt.slice(t) * arma::trans(Ft.row(t-1)) + St_tmp(t-1));
    Qt_inv_sq = std::sqrt(1.0/Qt);
    St_sq = std::sqrt(St_tmp(t-1));

    et = yt_star(t-1) - ft(t-1);
    S_comp += St_sq * Qt_inv_sq * et * et * Qt_inv_sq * St_sq;
    St_tmp(t) = (n_0*S_0 + S_comp)/(n_0 + t);
    At = Rt.slice(t) * arma::trans(Ft.row(t-1))/Qt;
    mt.col(t) = mt.col(t-1) + At * et;
    Ct.slice(t) = Rt.slice(t) - At * Qt * arma::trans(At);

  }
  St = St_tmp.rows(1, N);
  mT.col(N) = mt.col(N);
  CT.slice(N) = Ct.slice(N);
  CT.slice(N) = 0.5*CT.slice(N) + 0.5*arma::trans(CT.slice(N));
  arma::mat eps = arma::randn(1, d);
  arma::chol(L_upper, CT.slice(N));
  // bool chol_success = chol(L_upper, CT.slice(N));
  //
  // // Fall back on Rs chol if armadillo fails
  // if(chol_success == false){
  //   Rcpp::NumericMatrix tmp = Rchol(CT.slice(N), true, false, -1);
  //   arma::uvec piv = arma::sort_index(as<arma::vec>(tmp.attr("pivot")));
  //   arma::mat L_upper_tmp = arma::mat(tmp.begin(), d, d, false);
  //   L_upper = L_upper_tmp.cols(piv);
  // }

  beta_nc_samp.row(N) = arma::trans(mT.col(N)) + eps * L_upper;
  y(N-1) = arma::as_scalar(yt_star(N-1) - Ft.row(N-1)*arma::trans(beta_nc_samp.row(N)));
  for(int t = N-1; t >= 0; t--){
    //Rtp1_inv = arma::inv(Rt.slice(t+1));
    //if(!(Rt.slice(t+1)).is_sympd()){
    //  Rt.slice(t+1) = .5*Rt.slice(t+1) + 0.5*arma::trans(Rt.slice(t+1));
    //}
    //Rt.slice(t+1) = .5*Rt.slice(t+1) + 0.5*arma::trans(Rt.slice(t+1));
    //Rtp1_inv = arma::inv_sympd(Rt.slice(t+1));
    Rtp1_inv = arma::inv(Rt.slice(t+1));
    //Rtp1_inv = I_d;
    Bt = Ct.slice(t) * Rtp1_inv;
    //mT.col(t) = mt.col(t) + Bt*(mT.col(t+1) - mt.col(t));
    mT.col(t) = mt.col(t) + Bt*(arma::trans(beta_nc_samp.row(t+1)) - mt.col(t));
    CT.slice(t) = Ct.slice(t) - Bt*(Rt.slice(t+1))*arma::trans(Bt);

    eps = arma::randn(1, d);
    CT.slice(t) = 0.5*CT.slice(t) + 0.5*arma::trans(CT.slice(t));
    chol(L_upper, CT.slice(t));
    // chol_success = chol(L_upper, CT.slice(t));
    // if(chol_success == false){
    //   Rcpp::NumericMatrix tmp = Rchol(CT.slice(t), true, false, -1);
    //   arma::uvec piv = arma::sort_index(as<arma::vec>(tmp.attr("pivot")));
    //   arma::mat L_upper_tmp = arma::mat(tmp.begin(), d, d, false);
    //   L_upper = L_upper_tmp.cols(piv);
    // }
    beta_nc_samp.row(t) = arma::trans(mT.col(t)) + eps * L_upper;
  }
}

// void get_w(arma::mat& W, arma::mat& x, arma::mat& beta_nc_samp, int N, int n_I){
//   arma::mat I_d = arma::eye(n_I, n_I);
//   arma::mat xt;
//   arma::mat xt_tilde;
//   for(int t = 1; t < (N+1); t++){
//     xt = arma::kron(I_d, x.row(t-1));
//     xt_tilde = arma::kron(I_d, x.row(t-1));
//     xt_tilde.each_row() %= (beta_nc_samp.col(t)).t();
//     W.rows((t-1)*n_I, t*n_I-1) = arma::join_rows(xt, xt_tilde);
//   }
// }
void sample_alpha(arma::vec& alpha_samp, arma::vec& y,
                  arma::mat& x, arma::mat& x_tilde,
                  arma::colvec& tau2, arma::colvec& xi2,
                  arma::vec& SIGMA, int n_I, Function Rchol) {
  int d = x.n_cols;

  arma::mat A0_sr = arma::diagmat(arma::sqrt(arma::join_cols(tau2, xi2)));

  arma::mat W = arma::join_rows(x, x_tilde);
  arma::mat W_til = W.t() * arma::diagmat(1.0/SIGMA);
  arma::mat a = W_til * y;
  arma::mat Omega_star = A0_sr * W_til * W * A0_sr + arma::eye(2*d, 2*d);

  arma::mat A_t;
  arma::mat A_t_til;
  bool solved = arma::solve(A_t_til, Omega_star, A0_sr, arma::solve_opts::no_approx);
  //arma::cout << "solved  = " << solved << arma::endl;
  if (solved == 1){
    A_t = A0_sr * A_t_til;
  } else {
    arma::mat A0_inv = arma::diagmat(arma::join_cols(1/tau2, 1/xi2));
    A_t = arma::inv(W_til * W + A0_inv);
  }

  A_t = 0.5 * A_t + 0.5 * arma::trans(A_t);
  //arma::cout << "A_t: " << A_t << arma::endl;
  arma::vec v = rnorm(2*d);

  arma::mat cholA;
  bool chol_success = chol(cholA, A_t);

  // Fall back on Rs chol if armadillo fails (it suppports pivoting)
  if (chol_success == false){
    Rcpp::NumericMatrix tmp = Rchol(A_t, true, false, -1);
    arma::mat cholA_tmp = arma::mat(tmp.begin(), 2*d, 2*d, false);
    arma::uvec piv = arma::sort_index(as<arma::vec>(tmp.attr("pivot")));
    cholA = cholA_tmp.cols(piv);
  }

  alpha_samp = A_t * a + cholA.t() * v;
  std::for_each(alpha_samp.begin(), alpha_samp.end(), res_protector);
}


void resample_alpha_diff(arma::vec& alpha_samp, arma::mat betaenter, arma::vec& theta_sr, arma::vec& beta_mean, arma::mat beta_diff,  arma::vec& xi2, arma::vec& tau2, int d, int N){
  arma::vec sign_sqrt = arma::sign(theta_sr);
  arma::colvec theta_sr_new(d, arma::fill::none);
  int p1_theta = -N/2;

  arma::vec theta(d, arma::fill::none);


  for (int j = 0; j < d; j++){
    double p2_theta = 1/xi2(j);
    double p3_theta = arma::as_scalar(arma::accu(arma::pow(beta_diff.col(j), 2))) +
      std::pow((betaenter(0, j) - beta_mean(j)), 2);

    double res = do_rgig1(p1_theta, p3_theta, p2_theta);
    theta(j) = res;
    theta_sr_new(j) = std::sqrt(arma::as_scalar(theta(j))) * sign_sqrt(j);
  }

  arma::colvec beta_mean_new(d, arma::fill::none);

  for (int j = 0; j < d; j++){
    double sigma2_beta_mean = 1/(1/tau2(j) + 1/(theta(j)));
    double mu_beta_mean = betaenter(0, j) * tau2(j)/(tau2(j) + theta(j));
    beta_mean_new(j) = R::rnorm(mu_beta_mean, std::sqrt(sigma2_beta_mean));
  }

  alpha_samp = arma::join_cols(beta_mean_new, theta_sr_new);
  std::for_each(alpha_samp.begin(), alpha_samp.end(), res_protector);
}


void sample_local_shrink(arma::vec& local_shrink, const arma::vec& param_vec, double global_shrink, double a){
  int d = local_shrink.n_elem;
  arma::vec param_vec2 = arma::pow(param_vec, 2);

  double p1 = a - 0.5;
  double p2 = a * global_shrink;

  for(int j = 0; j < d; j++){

    double p3 = param_vec2(j);
    local_shrink(j) = do_rgig1(p1, p3, p2);

  }
  std::for_each(local_shrink.begin(), local_shrink.end(), res_protector);
}

double sample_global_shrink(const arma::vec& prior_param, double a, double hyper1, double hyper2){
  int d = prior_param.n_elem;
  double hyper1_post = hyper1 + a * d;
  double hyper2_post = hyper2 + arma::mean(prior_param) * a * d * 0.5;
  double res = R::rgamma(hyper1_post, 1.0/hyper2_post);

  res_protector(res);
  return res;
}

