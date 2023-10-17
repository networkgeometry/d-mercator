#ifndef MERCATOR_INTEGRATE_EXPECTED_DEGREE_H
#define MERCATOR_INTEGRATE_EXPECTED_DEGREE_H

#include <cmath>

extern "C" {
  /**
   * Integral from 0 to PI: (Sin[theta]^(D - 1)) / (1 + (cx)^beta)
   *
   * @param result - results of the integral
   * @param d - dimension
   * @param beta
   * @param c = R / (mu * kappa_i * kappa_j) ** (1 / d)
   */
  void pkk_(double *result, int *d, double *beta, double *c, double *upper_bound);
}

extern "C" {
  /**
   * Integral from 0 to PI: (theta * Sin[theta]^(D - 1)) / (1 + (cx)^beta)
   *
   * @param result - results of the integral
   * @param d - dimension
   * @param beta
   * @param c = R / (mu * kappa_i * kappa_j) ** (1 / d)
   */
  void pkk_expected_(double *result, int *d, double *beta, double *c);
}

double compute_integral_expected_degree_dimensions(int dim, double radius, double mu, double beta,
                                                   double kappa1, double kappa2, double upper_bound=M_PI) {
    double c = radius / std::pow(mu * kappa1 * kappa2, 1.0 / dim);
    double result{0};
    pkk_(&result, &dim, &beta, &c, &upper_bound);
    return result;
}


double compute_integral_expected_theta(int dim, double radius, double mu, double beta, 
                                       double kappa1, double kappa2) {
    double c = radius / std::pow(mu * kappa1 * kappa2, 1.0 / dim);
    double result{0};
    pkk_expected_(&result, &dim, &beta, &c);
    return result;
}


#endif //MERCATOR_INTEGRATE_EXPECTED_DEGREE_H
