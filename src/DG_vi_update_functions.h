#ifndef DG_SAMPLING_FUNCTIONS_H
#define DG_SAMPLING_FUNCTIONS_H

void update_local_shrink(arma::vec& local_shrink,
                         arma::vec& local_shrink_inv,
                         const arma::vec& param_vec2,
                         arma::vec& global_shrink,
                         arma::vec& a);

double update_global_shrink(const arma::vec& prior_var,
                            double a,
                            double hyper1,
                            double hyper2,
                            int d);

#endif
