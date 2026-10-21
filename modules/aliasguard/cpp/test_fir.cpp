#include "fir.hpp"
#include <iostream>
#include <limits>

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("streaming invariant failed");
    }
}
int main() {
    using aliasguard::Decimator;
    Decimator fir({1.0, 2.0, 3.0}, 2);
    require(fir.process({1.0, 2.0}) == std::vector<double>{1.0});
    require(fir.process({}) == std::vector<double>{});
    try {
        fir.process({8.0, std::numeric_limits<double>::quiet_NaN()});
        require(false);
    } catch (const std::invalid_argument &) {
    }
    require(fir.process({3.0, 4.0, 5.0}) == (std::vector<double>{10.0, 22.0}));
    fir.reset();
    require(fir.process({1.0, 2.0, 3.0, 4.0, 5.0}) == (std::vector<double>{1.0, 10.0, 22.0}));
    Decimator one({2.0}, 1);
    require(one.process({-3.0, 4.0}) == (std::vector<double>{-6.0, 8.0}));
    try {
        Decimator invalid({1.0}, 0);
        require(false);
    } catch (const std::invalid_argument &) {
    }
    std::cout << "Native streaming, reset, phase and transactional-input checks passed\n";
}
