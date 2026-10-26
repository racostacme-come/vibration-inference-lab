// Exact rectangular-load convolution independently checks ForceWindow's kernel.
#include "../modules/forcewindow/cpp/state.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

int main() {
  const double omega = 2 * std::acos(-1.0), zeta = 0.05, dt = 0.013;
  const double alpha = zeta * omega, beta = omega * std::sqrt(1-zeta*zeta);
  const double decay = std::exp(-alpha*dt), s = std::sin(beta*dt)/beta;
  const double c = std::cos(beta*dt);
  const std::array<double,4> f{decay*(c+alpha*s), decay*s,
                              -decay*omega*omega*s, decay*(c-alpha*s)};
  const std::array<double,2> b{(1-f[0])/(omega*omega), -f[2]/(omega*omega)};
  const std::vector<double> loads{1.0, -0.3, 2.0, 0.0, -0.7};
  const auto response = Recurrence(f, b, loads.size(), 73).forward(loads);
  auto step = [=](double t) {
    if(t <= 0) return 0.0;
    return (1-std::exp(-alpha*t)*(std::cos(beta*t)+alpha*std::sin(beta*t)/beta))/(omega*omega);
  };
  double maximum = 0;
  for(std::size_t k=0; k<response.size(); ++k) {
    double exact = 0;
    for(std::size_t j=0; j<loads.size(); ++j)
      exact += loads[j]*(step((static_cast<double>(k)-static_cast<double>(j))*dt)
                        -step((static_cast<double>(k)-static_cast<double>(j)-1)*dt));
    maximum = std::max(maximum, std::abs(response[k]-exact));
  }
  std::cout << std::setprecision(17) << "Continuous rectangular-load response maximum error: " << maximum << '\n';
  return maximum < 2e-13 ? 0 : 1;
}
