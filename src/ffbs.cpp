// [Rcpp::depends(RcppArmadillo)]
#include <RcppArmadillo.h>
#include <math.h>

using namespace Rcpp;


void update_beta_tilde(arma::mat& beta_nc,
                       arma::mat& beta2_nc,
                       arma::cube& beta_nc_cov,
                       arma::vec& y, arma::mat& x,
                       const arma::vec& theta_sr,
                       const arma::vec& beta_mean, const int N,
                       const double S_0,
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

  arma::colvec theta = arma::pow(theta_sr, 2);
  arma::mat theta_sr_diag = arma::diagmat(theta_sr);
  arma::vec yt_star = y - x*beta_mean;
  arma::mat Ft = x*theta_sr_diag;


  arma::mat I_d = arma::eye(d, d);
  arma::vec ft(N, arma::fill::zeros);
  double Qt;

  double Qt_inv_sq;
  arma::vec At;
  double et;
  Ct.slice(0) = I_d;
  double n_0 = 1.0;
  St_tmp(0) = S_0;
  arma::mat Bt;
  arma::mat Rtp1_inv;

  for(int t = 1; t < N+1; t++){

    Rt.slice(t) = Ct.slice(t-1) + I_d;
    ft(t-1) = arma::as_scalar(Ft.row(t-1)*mt.col(t-1));
    Qt = arma::as_scalar(Ft.row(t-1)*Rt.slice(t)*arma::trans(Ft.row(t-1)) + St_tmp(t-1));
    St_sq = std::sqrt(St_tmp(t-1));

    Qt_inv_sq = std::sqrt(1.0/Qt);
    et = yt_star(t-1) - ft(t-1);
    S_comp += St_sq * Qt_inv_sq * et * et * Qt_inv_sq * St_sq;
    St_tmp(t) = (n_0*S_0 + S_comp)/(n_0 + t);
    At = Rt.slice(t)*arma::trans(Ft.row(t-1))/Qt;
    mt.col(t) = mt.col(t-1) + At*et;
    Ct.slice(t) = Rt.slice(t) - At*Qt*arma::trans(At);
    //Ct.slice(t) = 0.5*Ct.slice(t) + 0.5*arma::trans(Ct.slice(t));
  }
  St = St_tmp.rows(1, N);
  beta_nc.row(N) = arma::trans(mt.col(N));
  CT.slice(N) = Ct.slice(N);
  CT.slice(N) = 0.5*CT.slice(N) + 0.5*arma::trans(CT.slice(N));
  arma::mat tmp = CT.slice(N) + arma::trans(beta_nc.row(N))*beta_nc.row(N);
  beta2_nc.row(N) = arma::trans(tmp.diag());
  tmp.diag().zeros();
  beta_nc_cov.row(N) = tmp;
  //beta_nc_samp.col(N) = mT.col(N);
  y(N-1) = arma::as_scalar(yt_star(N-1) - Ft.row(N-1)*arma::trans(beta_nc.row(N)));
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
    beta_nc.row(t) = arma::trans(mt.col(t) + Bt*(arma::trans(beta_nc.row(t+1)) - mt.col(t)));
    CT.slice(t) = Ct.slice(t) - Bt*(Rt.slice(t+1) - CT.slice(t+1))*arma::trans(Bt);
    CT.slice(t) = 0.5*CT.slice(t) + 0.5*arma::trans(CT.slice(t));
    arma::mat tmp = Ct.slice(t) + arma::trans(beta_nc.row(t))*beta_nc.row(t);
    beta2_nc.row(t) = arma::trans(tmp.diag());
    tmp.diag().zeros();
    beta_nc_cov.row(t) = tmp;
    //beta_nc_samp.col(t) = mT.col(t);
    if(t > 0){
      y(t-1) = arma::as_scalar(yt_star(t-1) - Ft.row(t-1)*arma::trans(beta_nc.row(t)));
    }
  }
}
