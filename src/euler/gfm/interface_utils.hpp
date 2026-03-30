#pragma once

#include <array>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "src/euler/level_set/level_set_core.hpp"


// [0] Check if a face crosses the interface
inline bool is_interface_face(
    double phiL,
    double phiR
)
{
    return (phiL * phiR <= 0.0);
}


/*
[1] Compute linearised interface position fraction

    returns alpha in [0,1]
    alpha = 0 -> at left cell
    alpha = 1 -> at right cell
*/
inline double interface_fraction(
    double phiL,
    double phiR
)
{
    const double denom = phiL - phiR;

    if (std::abs(denom) < 1e-14) {
        return 0.5;
    }

    const double alpha = phiL / denom;

    return std::clamp(alpha, 0.0, 1.0);
}


// [2] Access precomputed interface normal
template<int DIM>
inline std::array<double, DIM> interface_normal(
    const std::vector<std::array<double, DIM>>& normals,
    int id
)
{
    if (id < 0 || id >= static_cast<int>(normals.size())) {
        throw std::runtime_error("interface_normal: invalid index");
    }

    return normals[id];
}


// [3] Signed distance to interface (linear along face)
inline double interface_distance(
    double phiL,
    double phiR,
    double dx
)
{
    const double alpha = interface_fraction(phiL, phiR);

    return alpha * dx;
}