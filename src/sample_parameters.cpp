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
}

void get_w(arma::mat& W, arma::mat& x, arma::mat& beta_nc_samp, int N, int n_I){
  arma::mat I_d = arma::eye(n_I, n_I);
  arma::mat xt;
  arma::mat xt_tilde;
  for(int t = 1; t < (N+1); t++){
    xt = arma::kron(I_d, x.row(t-1));
    xt_tilde = arma::kron(I_d, x.row(t-1));
    xt_tilde.each_row() %= (beta_nc_samp.col(t)).t();
    W.rows((t-1)*n_I, t*n_I-1) = arma::join_rows(xt, xt_tilde);
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

  //Fall back on Rs chol if armadillo fails
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
    Rt.slice(t+1) = .5*Rt.slice(t+1) + 0.5*arma::trans(Rt.slice(t+1));
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
    if(t > 0){
      y(t-1) = arma::as_scalar(yt_star(t-1) - Ft.row(t-1)*arma::trans(beta_nc_samp.row(t)));
    }
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
                  arma::vec& SIGMA, arma::vec& a0, int n_I, Function Rchol) {
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




void sample_xi2(arma::vec& xi2_samp, arma::vec& theta_sr, double kappa2, double a_xi, int d){
  arma::vec theta = arma::pow(theta_sr, 2);


  for (int j = 0; j < d; j++){

    double p1_xi = a_xi - 0.5;
    double p2_xi = a_xi * kappa2;
    double p3_xi = theta(j);

    double res = do_rgig1(p1_xi, p3_xi, p2_xi);

    res_protector(res);

    xi2_samp(j) = res;
  }

}


void sample_tau2(arma::vec& tau2_samp, arma::vec& beta_mean, double lambda2, double a_tau, int d){
  arma::vec tau2(d, arma::fill::none);

  for (int j = 0; j < d; j++){
    double p1_tau = a_tau - 0.5;
    double p2_tau = a_tau * lambda2;
    double p3_tau = std::pow(beta_mean(j), 2);

    double res = do_rgig1(p1_tau, p3_tau, p2_tau);

    res_protector(res);

    tau2_samp(j) = res;

  }
}


double sample_kappa2(arma::vec xi2, double a_xi, double d1, double d2, int d){
  double d1_full = d1 + a_xi * d;
  double d2_full = d2 + arma::as_scalar(arma::mean(xi2)) * a_xi * d * 0.5;
  double kappa2 = R::rgamma(d1_full, 1/d2_full);
  return(kappa2);
}


double sample_lambda2(arma::vec tau2, double a_tau, double e1, double e2, int d){
  double e1_full = e1 + a_tau * d;
  double e2_full = e2 + arma::as_scalar(arma::mean(tau2)) * a_tau * d * 0.5;
  double lambda2 = R::rgamma(e1_full, 1/e2_full);
  return(lambda2);
}


void sample_sigma2(arma::vec& sig2_samp, arma::vec& y, arma::mat& W, arma::vec& alpha, double c0, double C0, int N){
  double a_full = c0 + N/2;
  double b_full = C0 + 0.5 * arma::as_scalar(arma::sum(arma::pow((y - W*alpha), 2)));
  double sig2 = 1/R::rgamma(a_full, 1/b_full);
  sig2_samp.fill(sig2);
}


double sample_C0(arma::vec& sig2, double g0, double c0, double G0){
  double a_full = g0 + c0;
  double b_full = G0 + 1/sig2(0);
  double C0 = R::rgamma(a_full, 1/b_full);
  return(C0);
}



