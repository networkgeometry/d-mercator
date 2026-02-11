#ifndef DMERCATOR_GEOMETRY_SD_GEOMETRY_HPP
#define DMERCATOR_GEOMETRY_SD_GEOMETRY_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace dmercator::geometry {

constexpr double PI = 3.141592653589793238462643383279502884197;

inline double wrap_angle(double angle)
{
  const double two_pi = 2.0 * PI;
  while(angle >= two_pi) {
    angle -= two_pi;
  }
  while(angle < 0.0) {
    angle += two_pi;
  }
  return angle;
}

inline double circular_distance(double a, double b)
{
  return PI - std::fabs(PI - std::fabs(a - b));
}

inline double vector_angle(const std::vector<double> &v1,
                           const std::vector<double> &v2,
                           double numerical_zero = 1e-12)
{
  if(v1.size() != v2.size()) {
    throw std::runtime_error("vector_angle expects vectors of identical length");
  }

  double dot = 0.0;
  double n1 = 0.0;
  double n2 = 0.0;
  for(std::size_t i = 0; i < v1.size(); ++i) {
    dot += v1[i] * v2[i];
    n1 += v1[i] * v1[i];
    n2 += v2[i] * v2[i];
  }

  const double denom = std::sqrt(n1) * std::sqrt(n2);
  if(denom <= numerical_zero) {
    return 0.0;
  }

  double cosine = dot / denom;
  cosine = std::max(-1.0, std::min(1.0, cosine));
  if(std::fabs(cosine - 1.0) < numerical_zero) {
    return 0.0;
  }

  return std::acos(cosine);
}

inline double sphere_radius(int dimension, int n_vertices)
{
  if(dimension < 1) {
    throw std::runtime_error("dimension must be >= 1");
  }
  const double inside = n_vertices /
    (2.0 * std::pow(PI, (dimension + 1) / 2.0)) *
    std::tgamma((dimension + 1) / 2.0);
  return std::pow(inside, 1.0 / static_cast<double>(dimension));
}

} // namespace dmercator::geometry

#endif // DMERCATOR_GEOMETRY_SD_GEOMETRY_HPP
