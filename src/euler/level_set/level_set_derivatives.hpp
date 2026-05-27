#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "src/euler/level_set/level_set_core.hpp"
#include "src/math/numerical_safety.hpp"


namespace level_set_detail {

// [0] Small parameter for WENO weights
inline double weno_eps()
{
    return 1e-6;
}


// [1] Check that phi size matches grid
template<int DIM>
inline void validate_phi_size(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    const char* where
)
{
    const int Ntot = static_cast<int>(total_cells(grid));

    if (static_cast<int>(phi.size()) != Ntot) {
        throw std::runtime_error(std::string(where) + ": phi size mismatch");
    }
}

// Standard (INDEX-BASED)


// [2] First-order one-sided minus derivative
template<int DIM>
inline double dminus_first_order(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    const std::array<int, DIM>& idx,
    int dir
)
{
    const int id = flatten_index<DIM>(idx, grid);
    const auto idx_m = offset_index_checked<DIM>(idx, dir, -1, grid);
    const int id_m = flatten_index<DIM>(idx_m, grid);

    return (phi[id] - phi[id_m]) / grid.dx[dir];
}


// [3] First-order one-sided plus derivative
template<int DIM>
inline double dplus_first_order(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    const std::array<int, DIM>& idx,
    int dir
)
{
    const int id = flatten_index<DIM>(idx, grid);
    const auto idx_p = offset_index_checked<DIM>(idx, dir, +1, grid);
    const int id_p = flatten_index<DIM>(idx_p, grid);

    return (phi[id_p] - phi[id]) / grid.dx[dir];
}


// [4] Second-order one-sided minus derivative
template<int DIM>
inline double dminus_second_order(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    const std::array<int, DIM>& idx,
    int dir
)
{
    const int id = flatten_index<DIM>(idx, grid);

    const auto idx_m1 = offset_index_checked<DIM>(idx, dir, -1, grid);
    const auto idx_m2 = offset_index_checked<DIM>(idx, dir, -2, grid);

    const int id_m1 = flatten_index<DIM>(idx_m1, grid);
    const int id_m2 = flatten_index<DIM>(idx_m2, grid);

    return (3.0 * phi[id] - 4.0 * phi[id_m1] + phi[id_m2]) / (2.0 * grid.dx[dir]);
}


// [5] Second-order one-sided plus derivative
template<int DIM>
inline double dplus_second_order(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    const std::array<int, DIM>& idx,
    int dir
)
{
    const int id = flatten_index<DIM>(idx, grid);

    const auto idx_p1 = offset_index_checked<DIM>(idx, dir, +1, grid);
    const auto idx_p2 = offset_index_checked<DIM>(idx, dir, +2, grid);

    const int id_p1 = flatten_index<DIM>(idx_p1, grid);
    const int id_p2 = flatten_index<DIM>(idx_p2, grid);

    return (-3.0 * phi[id] + 4.0 * phi[id_p1] - phi[id_p2]) / (2.0 * grid.dx[dir]);
}

// FLAT-INDEX 


// [6] First-order minus derivative (flat)
inline double dminus_first_order_flat(
    const std::vector<double>& phi,
    int id,
    int stride,
    double dx
)
{
    return (phi[id] - phi[id - stride]) / dx;
}


// [7] First-order plus derivative (flat)
inline double dplus_first_order_flat(
    const std::vector<double>& phi,
    int id,
    int stride,
    double dx
)
{
    return (phi[id + stride] - phi[id]) / dx;
}


// [8] Second-order minus derivative (flat)
inline double dminus_second_order_flat(
    const std::vector<double>& phi,
    int id,
    int stride,
    double dx
)
{
    return (3.0 * phi[id] - 4.0 * phi[id - stride] + phi[id - 2 * stride]) / (2.0 * dx);
}


// [9] Second-order plus derivative (flat)
inline double dplus_second_order_flat(
    const std::vector<double>& phi,
    int id,
    int stride,
    double dx
)
{
    return (-3.0 * phi[id] + 4.0 * phi[id + stride] - phi[id + 2 * stride]) / (2.0 * dx);
}


// [10] WENO5 left-biased derivative from stencil
inline double weno5_left_from_stencil(
    double v0, double v1, double v2,
    double v3, double v4,
    double dx
)
{
    const double q0 = (1.0/3.0*v0 - 7.0/6.0*v1 + 11.0/6.0*v2) / dx;
    const double q1 = (-1.0/6.0*v1 + 5.0/6.0*v2 + 1.0/3.0*v3) / dx;
    const double q2 = (1.0/3.0*v2 + 5.0/6.0*v3 - 1.0/6.0*v4) / dx;

    const double b0 =
        13.0/12.0 * (v0 - 2*v1 + v2)*(v0 - 2*v1 + v2) +
        0.25 * (v0 - 4*v1 + 3*v2)*(v0 - 4*v1 + 3*v2);

    const double b1 =
        13.0/12.0 * (v1 - 2*v2 + v3)*(v1 - 2*v2 + v3) +
        0.25 * (v1 - v3)*(v1 - v3);

    const double b2 =
        13.0/12.0 * (v2 - 2*v3 + v4)*(v2 - 2*v3 + v4) +
        0.25 * (3*v2 - 4*v3 + v4)*(3*v2 - 4*v3 + v4);

    const double eps = weno_eps();

    const double a0 = 0.1 / ((eps + b0)*(eps + b0));
    const double a1 = 0.6 / ((eps + b1)*(eps + b1));
    const double a2 = 0.3 / ((eps + b2)*(eps + b2));

    const double asum = a0 + a1 + a2;

    return (a0/asum)*q0 + (a1/asum)*q1 + (a2/asum)*q2;
}


// [11] WENO5 right-biased derivative
inline double weno5_right_from_stencil(
    double v0, double v1, double v2,
    double v3, double v4,
    double dx
)
{
    return -weno5_left_from_stencil(v4, v3, v2, v1, v0, dx);
}


// [12] WENO5 left (flat)
inline double weno5_left_flat(
    const std::vector<double>& phi,
    int id,
    int stride,
    double dx
)
{
    return weno5_left_from_stencil(
        phi[id - 2*stride],
        phi[id - stride],
        phi[id],
        phi[id + stride],
        phi[id + 2*stride],
        dx
    );
}


// [13] WENO5 right (flat)
inline double weno5_right_flat(
    const std::vector<double>& phi,
    int id,
    int stride,
    double dx
)
{
    return weno5_right_from_stencil(
        phi[id - 2*stride],
        phi[id - stride],
        phi[id],
        phi[id + stride],
        phi[id + 2*stride],
        dx
    );
}

// Public Flat API 

/*
    [14] Public one-sided minus derivative (flat)

    The WENO helpers above reconstruct point values, not phi_x directly, so
    they do not preserve the derivative of a linear signed-distance field.
*/
template<int DIM>
inline double dminus_flat(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    int id,
    int coord,
    int dir
)
{
    const int stride = grid.stride[dir];
    const double dx = grid.dx[dir];

    if (coord >= 2) {
        return dminus_second_order_flat(phi, id, stride, dx);
    }

    return dminus_first_order_flat(phi, id, stride, dx);
}


// [15] Public one-sided plus derivative (flat)
template<int DIM>
inline double dplus_flat(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    int id,
    int coord,
    int dir
)
{
    const int stride = grid.stride[dir];
    const double dx = grid.dx[dir];

    if (coord <= grid.N[dir] - 3) {
        return dplus_second_order_flat(phi, id, stride, dx);
    }

    return dplus_first_order_flat(phi, id, stride, dx);
}


// [16] Public central derivative (flat)
template<int DIM>
inline double dcenter_flat(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    int id,
    int dir
)
{
    const int stride = grid.stride[dir];
    return (phi[id + stride] - phi[id - stride]) / (2.0 * grid.dx[dir]);
}
}


