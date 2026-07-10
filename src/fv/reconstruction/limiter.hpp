#pragma once

#include "src/core/fv/limiters.hpp"

inline double minmod(double a, double b)
{
    return core::fv::minmod(a, b);
}

inline double mc_limiter(double a, double b)
{
    return core::fv::monotonized_central(a, b);
}

