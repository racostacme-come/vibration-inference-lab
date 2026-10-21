#include "energy.hpp"
#include <iostream>
int main() {
  const double x[] = {0, 2};
  const double y[] = {1};
  const auto u = jointguard::energy(x, 2, y, 1, 1, true);
  const auto e = jointguard::energy(x, 2, y, 1, 1, false);
  if (std::abs(u[0]) > 1e-14 || std::abs(e[0] - 0.5) > 1e-14)
    return 1;
  try {
    jointguard::energy(x, 1, y, 1, 1, true);
    return 2;
  } catch (const std::invalid_argument &) {
  }
  std::cout << "Native analytical energy scores and invalid-size guard passed\n";
  return 0;
}
