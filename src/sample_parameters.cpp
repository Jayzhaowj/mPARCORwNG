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



void sample_beta_tilde(arma::mat& beta_nc_samp, arma::mat& y, arma::mat& x, arma::colvec& theta_sr, arma::mat& SIGMA, arma::colvec& beta_mean, int N, int n_I, arma::mat S_0, Function Rchol){
  // initialization
  int d = n_I;
  int d2 = std::pow(n_I, 2);
  arma::mat xt;
  arma::cube Ft(d, d2, N, arma::fill::none);
  arma::mat mt(d2, N+1, arma::fill::zeros);
  arma::cube Rt(d2, d2, N+1, arma::fill::zeros);
  arma::cube Ct(d2, d2, N+1, arma::fill::zeros);
  arma::mat mT(d2, N+1, arma::fill::zeros);
  arma::cube CT(d2, d2, N+1, arma::fill::zeros);
  arma::cube St(d, d, N+1, arma::fill::zeros);
  arma::mat St_sqp;
  arma::mat S_comp(d, d, arma::fill::zeros);


  double n_0 = 1.0;
  arma::mat yt_star(d, N, arma::fill::zeros);
  arma::colvec theta = arma::pow(theta_sr, 2);
  arma::mat theta_sr_diag = arma::diagmat(theta_sr);
  arma::mat I_d2 = arma::eye(d2, d2);
  arma::mat I_d = arma::eye(d, d);
  arma::colvec ft;
  arma::mat Qt;
  arma::mat Qt_inv;
  arma::mat Qt_inv_sq;
  arma::mat At;
  arma::colvec et;
  St.slice(0) = S_0;
  Ct.slice(0) = I_d2;


  arma::mat Bt;
  arma::mat Rtp1_inv;

  arma::mat L_upper;
  for(int t = 1; t < N+1; t++){
    xt = arma::kron(I_d, x.row(t-1));
    yt_star.col(t-1) = arma::trans(y.row(t-1)) - xt*beta_mean;
    Ft.slice(t-1) = xt*theta_sr_diag;
    Rt.slice(t) = Ct.slice(t-1) + I_d2;
    ft = Ft.slice(t-1)*mt.col(t-1);

    St_sqp = arma::sqrtmat_sympd(St.slice(t-1));

    //St_sqp = arma::sqrtmat(St.slice(t-1));
    Qt = Ft.slice(t-1)*Rt.slice(t)*arma::trans(Ft.slice(t-1)) + St.slice(t-1);
    //Qt = 0.5*Qt + 0.5*arma::trans(Qt);
    //Qt_inv = arma::inv_sympd(Qt);
    Qt_inv = arma::inv(Qt);
    Qt_inv = 0.5*Qt_inv + 0.5*arma::trans(Qt_inv);
    Qt_inv_sq = arma::sqrtmat_sympd(Qt_inv);
    //Qt_inv_sq = arma::sqrtmat(Qt_inv);
    et = yt_star.col(t-1) - ft;

    S_comp += St_sqp * Qt_inv_sq * et * arma::trans(et) * Qt_inv_sq * St_sqp;
    St.slice(t) = (n_0*S_0 + S_comp)/(n_0 + t);
    St.slice(t) = 0.5*St.slice(t) + 0.5*arma::trans(St.slice(t));

    //St.slice(t) = I_d;
    At = Rt.slice(t)*arma::trans(Ft.slice(t-1))*Qt_inv;
    mt.col(t) = mt.col(t-1) + At*et;
    Ct.slice(t) = Rt.slice(t) - At*Qt*arma::trans(At);
    //Ct.slice(t) = 0.5*Ct.slice(t) + 0.5*arma::trans(Ct.slice(t));
  }
  SIGMA = St.slice(N);
  mT.col(N) = mt.col(N);
  CT.slice(N) = Ct.slice(N);
  CT.slice(N) = 0.5*CT.slice(N) + 0.5*arma::trans(CT.slice(N));
  arma::mat eps = arma::randn(1, d2);
  bool chol_success = chol(L_upper, CT.slice(N));

  //Fall back on Rs chol if armadillo fails
  if(chol_success == false){
    Rcpp::NumericMatrix tmp = Rchol(CT.slice(N), true, false, -1);
    arma::uvec piv = arma::sort_index(as<arma::vec>(tmp.attr("pivot")));
    arma::mat L_upper_tmp = arma::mat(tmp.begin(), d2, d2, false);
    L_upper = L_upper_tmp.cols(piv);
  }

  beta_nc_samp.col(N) = mT.col(N) + arma::trans(eps * L_upper);
  //beta_nc_samp.col(N) = mT.col(N);
  y.row(N-1) = arma::trans(yt_star.col(N-1) - Ft.slice(N-1)*beta_nc_samp.col(N));
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
    mT.col(t) = mt.col(t) + Bt*(beta_nc_samp.col(t+1) - mt.col(t));
    CT.slice(t) = Ct.slice(t) - Bt*(Rt.slice(t+1))*arma::trans(Bt);
    //CT.slice(t) = 0.5*CT.slice(t) + 0.5*arma::trans(CT.slice(t));
    eps = arma::randn(1, d2);
    CT.slice(t) = 0.5*CT.slice(t) + 0.5*arma::trans(CT.slice(t));
    chol_success = chol(L_upper, CT.slice(t));
    if(chol_success == false){
      Rcpp::NumericMatrix tmp = Rchol(CT.slice(N), true, false, -1);
      arma::uvec piv = arma::sort_index(as<arma::vec>(tmp.attr("pivot")));
      arma::mat L_upper_tmp = arma::mat(tmp.begin(), d2, d2, false);
      L_upper = L_upper_tmp.cols(piv);
    }
    beta_nc_samp.col(t) = mT.col(t) + arma::trans(eps * L_upper);
    //beta_nc_samp.col(t) = mT.col(t);
    if(t > 0){
      y.row(t-1) = arma::trans(yt_star.col(t-1) - Ft.slice(t-1)*beta_nc_samp.col(t));
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
void sample_alpha(arma::vec& alpha_samp, arma::mat& y, arma::mat& x, arma::mat& beta_nc, arma::colvec& tau2, arma::colvec& xi2, arma::mat& SIGMA, arma::vec& a0, int n_I, Function Rchol) {
  int N = y.n_rows;
  int d = std::pow(n_I, 2);
  arma::mat I_d = arma::eye(n_I, n_I);
  arma::mat A0_sr = arma::diagmat(arma::sqrt(arma::join_cols(tau2, xi2)));
  arma::vec prior_a = a0/arma::join_cols(tau2, xi2);
  arma::mat xt;
  arma::mat xt_tilde;
  arma::mat W;
  arma::mat W_til;
  arma::mat a_tmp(2*d, 1, arma::fill::zeros);
  arma::mat Omega_star_tmp(2*d, 2*d, arma::fill::zeros);
  for(int t = 0; t < (N); t++){
    xt = arma::kron(I_d, x.row(t));
    xt_tilde = arma::kron(I_d, x.row(t));
    xt_tilde.each_row() %= (beta_nc.col(t)).t();
    W = arma::join_rows(xt, xt_tilde);
    W_til = W.t() * SIGMA;
    a_tmp += W_til*arma::trans(y.row(t));
    Omega_star_tmp += W_til * W;
  }
  arma::mat a = a_tmp + prior_a;
  //arma::colvec yt = arma::vectorise(arma::trans(y));
  //arma::cout << "SIGMA: " << SIGMA << arma::endl;
  //arma::mat SIGMA_tmp = arma::kron(arma::eye(N, N), SIGMA);
  //arma::mat W_til = W.t() * SIGMA_tmp;
  //arma::mat A0_sr = arma::diagmat(arma::sqrt(arma::join_cols(tau2, xi2)));
  //arma::cout << "size of a0: " << arma::size(a0) << arma::endl;
  //arma::cout << "size of tau2 and xi2: " << arma::size(arma::join_cols(tau2, xi2)) << arma::endl;
  //arma::cout << "size of W_til: " << arma::size(W_til) << arma::endl;
  //arma::cout << "size of A0_sr: " << arma::size(A0_sr) << arma::endl;


  //arma::vec prior_a = a0/arma::join_cols(tau2, xi2);
  //arma::mat a = W_til * yt + prior_a;
  //arma::cout << "size of a: " << arma::size(a) << arma::endl;
  //arma::mat Omega_star = A0_sr * W_til * W * A0_sr + arma::eye(2*d, 2*d);
  arma::mat Omega_star = A0_sr * Omega_star_tmp * A0_sr + arma::eye(2*d, 2*d);
  //arma::cout << "size of Omega_star: " << arma::size(Omega_star) << arma::endl;
  //arma::cout << "Omega_star_tmp: " << Omega_star_tmp << arma::endl;
  arma::mat A_t;
  arma::mat A_t_til;
  bool solved = arma::solve(A_t_til, Omega_star, A0_sr, arma::solve_opts::no_approx);
  //arma::cout << "solved  = " << solved << arma::endl;
  if (solved == 1){
    A_t = A0_sr * A_t_til;
  } else {
    arma::mat A0_inv = arma::diagmat(arma::join_cols(1/tau2, 1/xi2));
    //A_t = arma::inv(W_til * W + arma::inv(diagmat(A0)));
    A_t = arma::inv(Omega_star_tmp + A0_inv);
  }


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


void resample_alpha_diff(arma::vec& alpha_samp, arma::mat& betaenter, arma::vec& theta_sr, arma::vec& beta_mean, arma::mat& beta_diff,  arma::vec& xi2, arma::vec& tau2, int n_I, int N){
  int d = std::pow(n_I, 2);
  arma::vec sign_sqrt = arma::sign(theta_sr);
  arma::colvec theta_sr_new(d, arma::fill::none);
  int p1_theta = -N/2;

  arma::vec theta(d, arma::fill::none);


  for (int j = 0; j < d; j++){
    double p2_theta = 1/xi2(j);
    double p3_theta = arma::as_scalar(arma::accu(arma::pow(beta_diff.row(j), 2))) +
      std::pow((betaenter(j, 0) - beta_mean(j)), 2);

    double res = do_rgig1(p1_theta, p3_theta, p2_theta);
    theta(j) = res;
    theta_sr_new(j) = std::sqrt(arma::as_scalar(theta(j))) * sign_sqrt(j);
  }

  arma::colvec beta_mean_new(d, arma::fill::none);

  for (int j = 0; j < d; j++){
    double sigma2_beta_mean = 1/(1/tau2(j) + 1/(theta(j)));
    double mu_beta_mean = betaenter(j, 0) * tau2(j)/(tau2(j) + theta(j));
    beta_mean_new(j) = R::rnorm(mu_beta_mean, std::sqrt(sigma2_beta_mean));
  }

  alpha_samp = arma::join_cols(beta_mean_new, theta_sr_new);
  std::for_each(alpha_samp.begin(), alpha_samp.end(), res_protector);
}




void sample_xi2(arma::vec& xi2_samp, arma::vec& theta_sr, arma::vec& kappa2, arma::vec& a_xi, int d){
  arma::vec theta = arma::pow(theta_sr, 2);
  arma::vec xi2(d, arma::fill::none);

  for (int j = 0; j < d; j++){

    double p1_xi = a_xi(j) - 0.5;
    double p2_xi = a_xi(j) * kappa2(j);
    double p3_xi = theta(j);

    double res = do_rgig1(p1_xi, p3_xi, p2_xi);

    res_protector(res);

    xi2_samp(j) = res;
  }

}


void sample_tau2(arma::vec& tau2_samp, arma::vec& beta_mean, arma::vec& lambda2, arma::vec& a_tau, int d){
  arma::vec tau2(d, arma::fill::none);

  for (int j = 0; j < d; j++){
    double p1_tau = a_tau(j) - 0.5;
    double p2_tau = a_tau(j) * lambda2(j);
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



