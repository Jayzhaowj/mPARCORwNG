// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// pred_dens_mix_approx
arma::vec pred_dens_mix_approx(arma::vec x_test, arma::vec y_test, arma::mat theta_sr, arma::mat beta_mean, arma::vec sig2_samp, bool sv, arma::vec sv_phi, arma::vec sv_mu, arma::vec sv_sigma2, arma::cube chol_C_N_inv_samp, arma::cube m_N_samp, int M, bool log);
RcppExport SEXP _mPARCORwNG_pred_dens_mix_approx(SEXP x_testSEXP, SEXP y_testSEXP, SEXP theta_srSEXP, SEXP beta_meanSEXP, SEXP sig2_sampSEXP, SEXP svSEXP, SEXP sv_phiSEXP, SEXP sv_muSEXP, SEXP sv_sigma2SEXP, SEXP chol_C_N_inv_sampSEXP, SEXP m_N_sampSEXP, SEXP MSEXP, SEXP logSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x_test(x_testSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y_test(y_testSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type theta_sr(theta_srSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type beta_mean(beta_meanSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type sig2_samp(sig2_sampSEXP);
    Rcpp::traits::input_parameter< bool >::type sv(svSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type sv_phi(sv_phiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type sv_mu(sv_muSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type sv_sigma2(sv_sigma2SEXP);
    Rcpp::traits::input_parameter< arma::cube >::type chol_C_N_inv_samp(chol_C_N_inv_sampSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type m_N_samp(m_N_sampSEXP);
    Rcpp::traits::input_parameter< int >::type M(MSEXP);
    Rcpp::traits::input_parameter< bool >::type log(logSEXP);
    rcpp_result_gen = Rcpp::wrap(pred_dens_mix_approx(x_test, y_test, theta_sr, beta_mean, sig2_samp, sv, sv_phi, sv_mu, sv_sigma2, chol_C_N_inv_samp, m_N_samp, M, log));
    return rcpp_result_gen;
END_RCPP
}
// do_shrinkTVP
List do_shrinkTVP(arma::mat y, arma::vec a0, int d, int niter, int nburn, int nthin, double c0, double g0, double G0, double d1, double d2, double e1, double e2, bool learn_lambda2, bool learn_kappa2, double lambda2, double kappa2, bool learn_a_xi, bool learn_a_tau, double a_xi, double a_tau, double c_tuning_par_xi, double c_tuning_par_tau, double b_xi, double b_tau, double nu_xi, double nu_tau, bool display_progress, bool ret_beta_nc, bool store_burn);
RcppExport SEXP _mPARCORwNG_do_shrinkTVP(SEXP ySEXP, SEXP a0SEXP, SEXP dSEXP, SEXP niterSEXP, SEXP nburnSEXP, SEXP nthinSEXP, SEXP c0SEXP, SEXP g0SEXP, SEXP G0SEXP, SEXP d1SEXP, SEXP d2SEXP, SEXP e1SEXP, SEXP e2SEXP, SEXP learn_lambda2SEXP, SEXP learn_kappa2SEXP, SEXP lambda2SEXP, SEXP kappa2SEXP, SEXP learn_a_xiSEXP, SEXP learn_a_tauSEXP, SEXP a_xiSEXP, SEXP a_tauSEXP, SEXP c_tuning_par_xiSEXP, SEXP c_tuning_par_tauSEXP, SEXP b_xiSEXP, SEXP b_tauSEXP, SEXP nu_xiSEXP, SEXP nu_tauSEXP, SEXP display_progressSEXP, SEXP ret_beta_ncSEXP, SEXP store_burnSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    Rcpp::traits::input_parameter< int >::type niter(niterSEXP);
    Rcpp::traits::input_parameter< int >::type nburn(nburnSEXP);
    Rcpp::traits::input_parameter< int >::type nthin(nthinSEXP);
    Rcpp::traits::input_parameter< double >::type c0(c0SEXP);
    Rcpp::traits::input_parameter< double >::type g0(g0SEXP);
    Rcpp::traits::input_parameter< double >::type G0(G0SEXP);
    Rcpp::traits::input_parameter< double >::type d1(d1SEXP);
    Rcpp::traits::input_parameter< double >::type d2(d2SEXP);
    Rcpp::traits::input_parameter< double >::type e1(e1SEXP);
    Rcpp::traits::input_parameter< double >::type e2(e2SEXP);
    Rcpp::traits::input_parameter< bool >::type learn_lambda2(learn_lambda2SEXP);
    Rcpp::traits::input_parameter< bool >::type learn_kappa2(learn_kappa2SEXP);
    Rcpp::traits::input_parameter< double >::type lambda2(lambda2SEXP);
    Rcpp::traits::input_parameter< double >::type kappa2(kappa2SEXP);
    Rcpp::traits::input_parameter< bool >::type learn_a_xi(learn_a_xiSEXP);
    Rcpp::traits::input_parameter< bool >::type learn_a_tau(learn_a_tauSEXP);
    Rcpp::traits::input_parameter< double >::type a_xi(a_xiSEXP);
    Rcpp::traits::input_parameter< double >::type a_tau(a_tauSEXP);
    Rcpp::traits::input_parameter< double >::type c_tuning_par_xi(c_tuning_par_xiSEXP);
    Rcpp::traits::input_parameter< double >::type c_tuning_par_tau(c_tuning_par_tauSEXP);
    Rcpp::traits::input_parameter< double >::type b_xi(b_xiSEXP);
    Rcpp::traits::input_parameter< double >::type b_tau(b_tauSEXP);
    Rcpp::traits::input_parameter< double >::type nu_xi(nu_xiSEXP);
    Rcpp::traits::input_parameter< double >::type nu_tau(nu_tauSEXP);
    Rcpp::traits::input_parameter< bool >::type display_progress(display_progressSEXP);
    Rcpp::traits::input_parameter< bool >::type ret_beta_nc(ret_beta_ncSEXP);
    Rcpp::traits::input_parameter< bool >::type store_burn(store_burnSEXP);
    rcpp_result_gen = Rcpp::wrap(do_shrinkTVP(y, a0, d, niter, nburn, nthin, c0, g0, G0, d1, d2, e1, e2, learn_lambda2, learn_kappa2, lambda2, kappa2, learn_a_xi, learn_a_tau, a_xi, a_tau, c_tuning_par_xi, c_tuning_par_tau, b_xi, b_tau, nu_xi, nu_tau, display_progress, ret_beta_nc, store_burn));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_mPARCORwNG_pred_dens_mix_approx", (DL_FUNC) &_mPARCORwNG_pred_dens_mix_approx, 13},
    {"_mPARCORwNG_do_shrinkTVP", (DL_FUNC) &_mPARCORwNG_do_shrinkTVP, 30},
    {NULL, NULL, 0}
};

RcppExport void R_init_mPARCORwNG(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
