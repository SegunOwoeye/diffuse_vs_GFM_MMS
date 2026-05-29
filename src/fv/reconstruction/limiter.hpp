#pragma once

#include <algorithm>
#include <cmath>


// [1] Minmod limiter
inline double minmod(double a, double b)
{
    if (a * b <= 0.0) {
        return 0.0;
    }

    return (std::abs(a) < std::abs(b)) ? a : b;
}

// [2] Monotonized central limiter
inline double mc_limiter(double a, double b)
{
    if (a * b <= 0.0) {
        return 0.0;
    }

    double s = (a > 0.0) ? 1.0 : -1.0;

    double abs_a = std::abs(a);
    double abs_b = std::abs(b);
    double abs_c = 0.5 * std::abs(a + b);

    return s * std::min({2.0 * abs_a, 2.0 * abs_b, abs_c});
}

