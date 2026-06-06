#pragma once

#include <algorithm>
#include <cmath>

namespace core::fv {

// Two-argument minmod limiter used by MUSCL-Hancock reconstructions.
inline double minmod(double a, double b)
{
    if (a * b <= 0.0) {
        return 0.0;
    }
    return std::abs(a) < std::abs(b) ? a : b;
}

inline double monotonized_central(double a, double b)
{
    if (a * b <= 0.0) {
        return 0.0;
    }
    const double sign = a > 0.0 ? 1.0 : -1.0;
    return sign * std::min({2.0 * std::abs(a), 2.0 * std::abs(b), 0.5 * std::abs(a + b)});
}

} // namespace core::fv
