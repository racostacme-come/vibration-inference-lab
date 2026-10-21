#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace aliasguard {
// Causal FIR with zero prehistory. Emit n=0,R,2R,... across all process calls.
class Decimator {
  public:
    Decimator(std::vector<double> coefficients, std::size_t factor)
        : h_(std::move(coefficients)), history_(h_.size(), 0.0), factor_(factor) {
        if (h_.empty() || factor_ == 0) {
            throw std::invalid_argument("nonempty coefficients and positive factor required");
        }
        validate(h_);
    }

    std::vector<double> process(const std::vector<double> &input) {
        // Reject the entire chunk before changing the history or sampling phase.
        validate(input);
        std::vector<double> output;
        output.reserve(input.size() / factor_ + (input.size() % factor_ != 0 ? 1 : 0));
        for (const double value : input) {
            history_[head_] = value;
            if (phase_ == 0) {
                double sum = 0.0;
                auto index = head_;
                for (const double coefficient : h_) {
                    sum += coefficient * history_[index];
                    index = index == 0 ? history_.size() - 1 : index - 1;
                }
                output.push_back(sum);
            }
            head_ = head_ + 1 == history_.size() ? 0 : head_ + 1;
            phase_ = phase_ + 1 == factor_ ? 0 : phase_ + 1;
        }
        return output;
    }

    void reset() {
        std::fill(history_.begin(), history_.end(), 0.0);
        head_ = 0;
        phase_ = 0;
    }

  private:
    static void validate(const std::vector<double> &values) {
        for (const double value : values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("all values must be finite");
            }
        }
    }
    std::vector<double> h_;
    std::vector<double> history_;
    std::size_t factor_;
    std::size_t head_ = 0;
    std::size_t phase_ = 0;
};
} // namespace aliasguard
