#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace jointguard {
// Row-major samples. U-statistic correction assumes iid ensemble members.
inline std::vector<double> energy(const double *x, std::size_t m, const double *y,
                                  std::size_t n, std::size_t d, bool unbiased) {
  if (d == 0 || n == 0 || m == 0 || (unbiased && m < 2))
    throw std::invalid_argument("Need nonempty arrays and >=2 members for unbiased scoring");
  for (std::size_t i = 0; i < m * d; ++i)
    if (!std::isfinite(x[i]))
      throw std::invalid_argument("Ensemble must be finite");
  for (std::size_t i = 0; i < n * d; ++i)
    if (!std::isfinite(y[i]))
      throw std::invalid_argument("Observations must be finite");
  auto distance = [d](const double *a, const double *b) {
    double norm = 0;
    for (std::size_t k = 0; k < d; ++k)
      norm = std::hypot(norm, a[k] - b[k]);
    return norm;
  };
  double correction = 0;
  const double denominator = static_cast<double>(m) * static_cast<double>(unbiased ? m - 1 : m);
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < i; ++j)
      correction += distance(x + i * d, x + j * d) / denominator;
  std::vector<double> scores(n, -correction);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < m; ++j)
      scores[i] += distance(y + i * d, x + j * d) / static_cast<double>(m);
    if (!std::isfinite(scores[i]))
      throw std::overflow_error("Energy score overflow");
  }
  return scores;
}
} // namespace jointguard
