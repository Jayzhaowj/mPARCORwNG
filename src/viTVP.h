#ifndef VITVP_CPP_H
#define VITVP_CPP_H

List vi_shrinkTVP(arma::mat y_fwd,
                  arma::mat y_bwd,
                  int d,
                  double d1,
                  double d2,
                  double e1,
                  double e2,
                  double a_xi,
                  double a_tau,
                  bool learn_a_xi,
                  bool learn_a_tau,
                  int iter_max,
                  bool ind,
                  double S_0,
                  double epsilon,
                  bool skip);
#endif
