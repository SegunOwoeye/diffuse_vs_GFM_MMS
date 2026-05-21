#pragma once
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

/*
Program provides safe tooling for numerical computations
*/

// [1] Creating a double that is bound to a tolerance
inline double clamp_min(double x, double tol = 1e-10)
{
    return std::max(x, tol);
}

// [2] Safe division of numerical values to avoid division by 0
inline double safe_denom(double denom, double tol = 1e-10
)
{
    if (std::abs(denom) < tol) {
        return (denom < 0.0) ? -tol : tol;
    }

    return denom;
}

inline double safe_div(double num, double denom, double tol = 1e-10
)
{
    return num / safe_denom(denom, tol);
}

// [2.1] Roundoff-scale tolerance for geometry/level-set comparisons
inline double geometry_tolerance(
    double length_scale,
    double rel_tol = 1e-10,
    double abs_tol = 1e-12
)
{
    return std::max(abs_tol, rel_tol * std::max(std::abs(length_scale), abs_tol));
}

// [3] Ensures the value being returned is a finite number
inline double require_finite(double x, const std::string& msg)
{
    if (!std::isfinite(x)) {
        throw std::runtime_error(msg);
    }
    return x;
}

// [4] Require positivity for physics calculations
inline double require_positive(double x, const std::string& msg, double tol = 1e-10)
{
    if (!std::isfinite(x)) {
        throw std::runtime_error(msg + " (non-finite)");
    }

    if (x <= 0.0) {
        return tol; 
    }

    return x;
}

