#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

// The observation is taken BEFORE the interval load is applied.
class Recurrence {
  std::array<double, 4> a;
  std::array<double, 2> b;
  std::size_t n, tail;

  static void finite(const std::vector<double> &v) {
    for (double x : v)
      if (!std::isfinite(x))
        throw std::invalid_argument("input must be finite");
  }

public:
  Recurrence(std::array<double, 4> transition, std::array<double, 2> input, std::size_t loads,
             std::size_t extra)
      : a(transition), b(input), n(loads), tail(extra) {
    if (n == 0 || n > 10000000 || tail > 10000000 || n + tail > 10000000)
      throw std::invalid_argument("require 1 <= loads and loads + tail <= 10000000");
    finite(std::vector<double>(a.begin(), a.end()));
    finite(std::vector<double>(b.begin(), b.end()));
  }

  std::vector<double> forward(const std::vector<double> &f) const {
    if (f.size() != n)
      throw std::invalid_argument("wrong load length");
    finite(f);
    std::vector<double> y(n + tail);
    double q = 0, v = 0;
    for (std::size_t k = 0; k < y.size(); ++k) {
      y[k] = q;
      const double load = k < n ? f[k] : 0;
      const double qnew = a[0] * q + a[1] * v + b[0] * load;
      v = a[2] * q + a[3] * v + b[1] * load;
      q = qnew;
    }
    finite(y);
    return y;
  }

  std::vector<double> adjoint(const std::vector<double> &r) const {
    if (r.size() != n + tail)
      throw std::invalid_argument("wrong residual length");
    finite(r);
    std::vector<double> g(n);
    double p = 0, s = 0;
    for (std::size_t k = r.size(); k-- > 0;) {
      if (k < n)
        g[k] = b[0] * p + b[1] * s;
      const double pnew = a[0] * p + a[2] * s + r[k];
      s = a[1] * p + a[3] * s;
      p = pnew;
    }
    finite(g);
    return g;
  }
};
