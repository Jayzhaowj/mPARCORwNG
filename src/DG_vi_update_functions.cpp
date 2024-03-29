#include <RcppArmadillo.h>
#include <math.h>
#include <cmath>
#include "unur_bessel_k_nuasympt.h"
#include <boost/math/special_functions/bessel.hpp>
#include "sample_parameters.h"
#include <iostream>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]

void update_local_shrink(arma::vec& local_shrink,
                         arma::vec& local_shrink_inv,
                         const arma::vec& param_vec2,
                         double global_shrink,
                         double a){
  int d = local_shrink.n_elem;

  double p1 = a - 0.5;
  double p2 = a * global_shrink;
  //double term1;
  //double term2;
  //double term3;
  for (int j = 0; j < d; j++){
    double p3 = param_vec2(j);
    double part1 = std::sqrt(p2 * p3);
    //if(abs(part1) < 1){
    //  term1 = 2*p1/p3;
    //  term2 = std::log(R::bessel_k(p1 + 1, part1, true)) - (p1+1);
    //  term3 = std::log(R::bessel_k(p1, part1, true)) - p1;
    //  local_shrink(j) = std::exp(0.5 * std::log(p3) + term2 - term3 - 0.5 * std::log(p2));
    //  local_shrink_inv(j) = std::exp(0.5*std::log(p2) + term2 - term3 - 0.5 * std::log(p3)) - term1;
    //}else{
    local_shrink(j) = boost::math::cyl_bessel_k(p1+1, part1)*std::sqrt(p3)/(boost::math::cyl_bessel_k(p1, part1) * std::sqrt(p2));
    local_shrink_inv(j) = std::sqrt(p2)*boost::math::cyl_bessel_k(p1+1, part1)/(std::sqrt(p3)*boost::math::cyl_bessel_k(p1, part1)) - 2*p1/p3;

    //}

  }

  //std::for_each(local_shrink.begin(), local_shrink.end(), res_protector);
  //std::for_each(local_shrink_inv.begin(), local_shrink_inv.end(), res_protector);
}


double update_global_shrink(const arma::vec& prior_var,
                            double a,
                            double hyper1,
                            double hyper2){
  int d = prior_var.n_elem;

  double hyper1_full = hyper1 + a*d;
  double hyper2_full = hyper2 + arma::mean(prior_var) * a * d * 0.5;
  double global_shrink = hyper1_full/hyper2_full;
  //res_protector(global_shrink);
  return global_shrink;
}
