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


/* [6] WENO5 left-biased derivative from 5-point stencil
    Stencil values are:
        v0 = phi_{i-2}, v1 = phi_{i-1}, v2 = phi_i, v3 = phi_{i+1}, v4 = phi_{i+2}
*/
inline double weno5_left_from_stencil(
    double v0,
    double v1,
    double v2,
    double v3,
    double v4,
    double dx
)
{
    const double q0 = ( 1.0 / 3.0 * v0 - 7.0 / 6.0 * v1 + 11.0 / 6.0 * v2) / dx;
    const double q1 = (-1.0 / 6.0 * v1 + 5.0 / 6.0 * v2 + 1.0 / 3.0 * v3) / dx;
    const double q2 = ( 1.0 / 3.0 * v2 + 5.0 / 6.0 * v3-1.0 / 6.0 * v4) / dx;

    const double b0 =
        13.0 / 12.0 * (v0 - 2.0 * v1 + v2) * (v0 - 2.0 * v1 + v2) +
        0.25 * (v0 - 4.0 * v1 + 3.0 * v2) * (v0 - 4.0 * v1 + 3.0 * v2);

    const double b1 =
        13.0 / 12.0 * (v1 - 2.0 * v2 + v3) * (v1 - 2.0 * v2 + v3) +
        0.25 * (v1 - v3) * (v1 - v3);

    const double b2 =
        13.0 / 12.0 * (v2 - 2.0 * v3 + v4) * (v2 - 2.0 * v3 + v4) +
        0.25 * (3.0 * v2 - 4.0 * v3 + v4) * (3.0 * v2 - 4.0 * v3 + v4);

    const double eps = weno_eps();

    const double a0 = 0.1 / ((eps + b0) * (eps + b0));
    const double a1 = 0.6 / ((eps + b1) * (eps + b1));
    const double a2 = 0.3 / ((eps + b2) * (eps + b2));

    const double asum = a0 + a1 + a2;

    const double w0 = a0 / asum;
    const double w1 = a1 / asum;
    const double w2 = a2 / asum;

    return w0 * q0 + w1 * q1 + w2 * q2;
}


// [7] WENO5 right-biased derivative from 5-point stencil
// Uses mirrored stencil and symmetry.
inline double weno5_right_from_stencil(
    double v0,
    double v1,
    double v2,
    double v3,
    double v4,
    double dx
)
{
    return -weno5_left_from_stencil(v4, v3, v2, v1, v0, dx);
}


// [8] Check if WENO5 stencil is available
template<int DIM>
inline bool can_use_weno5(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid,
    int dir
)
{
    return idx[dir] >= 2 && idx[dir] <= grid.N[dir] - 3;
}


// [9] Check if second-order minus stencil is available
template<int DIM>
inline bool can_use_second_order_minus(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>&,
    int dir
)
{
    return idx[dir] >= 2;
}


// [10] Check if second-order plus stencil is available
template<int DIM>
inline bool can_use_second_order_plus(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid,
    int dir
)
{
    return idx[dir] <= grid.N[dir] - 3;
}

} // namespace level_set_detail


/* [11] Public one-sided minus derivative
    Uses WENO5 when available, then second-order, then first-order.
*/
 template<int DIM>
inline double dminus(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    const std::array<int, DIM>& idx,
    int dir
)
{
    level_set_detail::validate_phi_size<DIM>(phi, grid, "dminus");

    if (!is_valid_index<DIM>(idx, grid)) {
        throw std::runtime_error("dminus: invalid index");
    }

    if (dir < 0 || dir >= DIM) {
        throw std::runtime_error("dminus: bad dir");
    }

    if (level_set_detail::can_use_weno5<DIM>(idx, grid, dir)) {
        const int id_m2 = flatten_index<DIM>(offset_index_checked<DIM>(idx, dir, -2, grid), grid);
        const int id_m1 = flatten_index<DIM>(offset_index_checked<DIM>(idx, dir, -1, grid), grid);
        const int id = flatten_index<DIM>(idx, grid);
        const int id_p1 = flatten_index<DIM>(offset_index_checked<DIM>(idx, dir, +1, grid), grid);
        const int id_p2 = flatten_index<DIM>(offset_index_checked<DIM>(idx, dir, +2, grid), grid);

        return level_set_detail::weno5_left_from_stencil(
            phi[id_m2],
            phi[id_m1],
            phi[id],
            phi[id_p1],
            phi[id_p2],
            grid.dx[dir]
        );
    }

    if (level_set_detail::can_use_second_order_minus<DIM>(idx, grid, dir)) {
        return level_set_detail::dminus_second_order<DIM>(phi, grid, idx, dir);
    }

    return level_set_detail::dminus_first_order<DIM>(phi, grid, idx, dir);
}


/* [12] Public one-sided plus derivative
    Uses WENO5 when available, then second-order, then first-order.
*/
template<int DIM>
inline double dplus(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    const std::array<int, DIM>& idx,
    int dir
)
{
    level_set_detail::validate_phi_size<DIM>(phi, grid, "dplus");

    if (!is_valid_index<DIM>(idx, grid)) {
        throw std::runtime_error("dplus: invalid index");
    }

    if (dir < 0 || dir >= DIM) {
        throw std::runtime_error("dplus: bad dir");
    }

    if (level_set_detail::can_use_weno5<DIM>(idx, grid, dir)) {
        const int id_m2 = flatten_index<DIM>(offset_index_checked<DIM>(idx, dir, -2, grid), grid);
        const int id_m1 = flatten_index<DIM>(offset_index_checked<DIM>(idx, dir, -1, grid), grid);
        const int id = flatten_index<DIM>(idx, grid);
        const int id_p1 = flatten_index<DIM>(offset_index_checked<DIM>(idx, dir, +1, grid), grid);
        const int id_p2 = flatten_index<DIM>(offset_index_checked<DIM>(idx, dir, +2, grid), grid);

        return level_set_detail::weno5_right_from_stencil(
            phi[id_m2], phi[id_m1], phi[id], phi[id_p1],
            phi[id_p2], grid.dx[dir]
        );
    }

    if (level_set_detail::can_use_second_order_plus<DIM>(idx, grid, dir)) {
        return level_set_detail::dplus_second_order<DIM>(phi, grid, idx, dir);
    }

    return level_set_detail::dplus_first_order<DIM>(phi, grid, idx, dir);
}


/* [13] Public central derivative
    Keeps standard second-order central difference.
    This is mainly for normals and diagnostics, not upwind advection.
*/
template<int DIM>
inline double dcenter(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    const std::array<int, DIM>& idx,
    int dir
)
{
    level_set_detail::validate_phi_size<DIM>(phi, grid, "dcenter");

    if (!is_valid_index<DIM>(idx, grid)) {
        throw std::runtime_error("dcenter: invalid index");
    }

    if (dir < 0 || dir >= DIM) {
        throw std::runtime_error("dcenter: bad dir");
    }

    const auto idx_m = offset_index_checked<DIM>(idx, dir, -1, grid);
    const auto idx_p = offset_index_checked<DIM>(idx, dir, +1, grid);

    const int id_m = flatten_index<DIM>(idx_m, grid);
    const int id_p = flatten_index<DIM>(idx_p, grid);

    return (phi[id_p] - phi[id_m]) / (2.0 * grid.dx[dir]);
}

