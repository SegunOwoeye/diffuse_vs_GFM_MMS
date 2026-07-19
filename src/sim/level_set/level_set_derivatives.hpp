#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "src/sim/level_set/level_set_core.hpp"
#include "src/math/numerical_safety.hpp"
namespace level_set_detail {

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


// [4] First-order minus derivative (flat)
inline double dminus_first_order_flat(
    const std::vector<double>& phi,
    int id,
    int stride,
    double dx
)
{
    return (phi[id] - phi[id - stride]) / dx;
}


// [5] First-order plus derivative (flat)
inline double dplus_first_order_flat(
    const std::vector<double>& phi,
    int id,
    int stride,
    double dx
)
{
    return (phi[id + stride] - phi[id]) / dx;
}


// [6] WENO2 nonlinear slope
inline double weno2_slope_flat(
    const std::vector<double>& phi,
    int id,
    int stride
)
{
    const double delta_minus = phi[id] - phi[id - stride];
    const double delta_plus = phi[id + stride] - phi[id];
    const double epsilon = 1.0e-12;
    const double beta_minus = delta_minus * delta_minus;
    const double beta_plus = delta_plus * delta_plus;
    const double alpha_minus =
        0.5 / ((epsilon + beta_minus) * (epsilon + beta_minus));
    const double alpha_plus =
        0.5 / ((epsilon + beta_plus) * (epsilon + beta_plus));
    const double weight_sum = alpha_minus + alpha_plus;

    return
        (alpha_minus * delta_minus + alpha_plus * delta_plus) /
        weight_sum;
}


// [7] Second-order WENO upwind derivative for positive transport speed
inline double dminus_weno2_flat(
    const std::vector<double>& phi,
    int id,
    int stride,
    double dx
)
{
    const double slope_i = weno2_slope_flat(phi, id, stride);
    const double slope_im1 = weno2_slope_flat(phi, id - stride, stride);
    const double right_face_left_state = phi[id] + 0.5 * slope_i;
    const double left_face_left_state = phi[id - stride] + 0.5 * slope_im1;

    return (right_face_left_state - left_face_left_state) / dx;
}


// [8] Second-order WENO upwind derivative for negative transport speed
inline double dplus_weno2_flat(
    const std::vector<double>& phi,
    int id,
    int stride,
    double dx
)
{
    const double slope_i = weno2_slope_flat(phi, id, stride);
    const double slope_ip1 = weno2_slope_flat(phi, id + stride, stride);
    const double right_face_right_state = phi[id + stride] - 0.5 * slope_ip1;
    const double left_face_right_state = phi[id] - 0.5 * slope_i;

    return (right_face_right_state - left_face_right_state) / dx;
}


// Public Flat API

/*
    [9] Public one-sided minus derivative (flat)

    Uses the selected second-order reconstruction in the interior and falls
    back to first-order one-sided differences near boundaries.
*/
template<int DIM>
inline double dminus_flat(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    int id,
    int coord,
    int dir,
    LevelSetDerivativeScheme scheme = LevelSetDerivativeScheme::Weno2
)
{
    const int stride = grid.stride[dir];
    const double dx = grid.dx[dir];

    if (scheme == LevelSetDerivativeScheme::Weno2 &&
        coord >= 2 && coord <= grid.N[dir] - 2) {
        return dminus_weno2_flat(phi, id, stride, dx);
    }

    return dminus_first_order_flat(phi, id, stride, dx);
}


// [10] Public one-sided plus derivative (flat)
template<int DIM>
inline double dplus_flat(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    int id,
    int coord,
    int dir,
    LevelSetDerivativeScheme scheme = LevelSetDerivativeScheme::Weno2
)
{
    const int stride = grid.stride[dir];
    const double dx = grid.dx[dir];

    if (scheme == LevelSetDerivativeScheme::Weno2 &&
        coord >= 1 && coord <= grid.N[dir] - 3) {
        return dplus_weno2_flat(phi, id, stride, dx);
    }

    return dplus_first_order_flat(phi, id, stride, dx);
}


// [11] Public central derivative (flat)
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


