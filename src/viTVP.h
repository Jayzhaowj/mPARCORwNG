#ifndef VITVP_CPP_H
#define VITVP_CPP_H

List vi_shrinkTVP(arma::mat y,
                  int d,
                  double d1,
                  double d2,
                  double e1,
                  double e2,
                  double lambda2,
                  double kappa2,
                  bool learn_a_xi,
                  bool learn_a_tau,
                  int iter_max,
                  double epsilon);
#endif
