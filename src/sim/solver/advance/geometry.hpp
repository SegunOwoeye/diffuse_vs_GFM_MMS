#pragma once

#include <array>
#include <vector>
#include <cmath>

#include "src/math/vector_ops.hpp"

template<int DIM>
inline std::array<double, DIM> axis_normal(int dir)
{
    std::array<double, DIM> n{};
    n[dir] = 1.0;
    return n;
}

template<int DIM>
inline std::array<double, DIM> build_face_normal(
    const std::vector<std::array<double, DIM>>& normals,
    int idL,
    int idR,
    int dir,
    double tol = 1e-14
)
{
    std::array<double, DIM> n_face{};

    for (int d = 0; d < DIM; ++d) {
        n_face[d] = 0.5 * (normals[idL][d] + normals[idR][d]);
    }

    if (norm<DIM>(n_face) < tol) {
        return axis_normal<DIM>(dir);
    }

    return normalize<DIM>(n_face, tol);
}