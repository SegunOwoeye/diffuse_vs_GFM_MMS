#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include "src/euler/level_set/level_set_core.hpp"


// [0] Validate interior access
template<int DIM>
inline void assert_interior(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid,
    int dir
)
{
    if (idx[dir] <= 0 || idx[dir] >= grid.N[dir] - 1) {
        throw std::runtime_error("Derivative requires interior cell");
    }
}


// [1] Backward Difference
template<int DIM>
inline double dminus(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    const std::array<int, DIM>& idx,
    int dir
)
{
    assert_interior<DIM>(idx, grid, dir);

    const auto stride = compute_strides<DIM>(grid);

    const int id  = flatten_index(idx, grid);
    const int idm = id - stride[dir];

    return (phi[id] - phi[idm]) / grid.dx[dir];
}


// [2] Forward Difference
template<int DIM>
inline double dplus(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    const std::array<int, DIM>& idx,
    int dir
)
{
    assert_interior<DIM>(idx, grid, dir);

    const auto stride = compute_strides<DIM>(grid);

    const int id  = flatten_index(idx, grid);
    const int idp = id + stride[dir];

    return (phi[idp] - phi[id]) / grid.dx[dir];
}


// [3] Central Difference
template<int DIM>
inline double dcenter(
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    const std::array<int, DIM>& idx,
    int dir
)
{
    assert_interior<DIM>(idx, grid, dir);

    const auto stride = compute_strides<DIM>(grid);

    const int id  = flatten_index(idx, grid);
    const int idm = id - stride[dir];
    const int idp = id + stride[dir];

    return (phi[idp] - phi[idm]) / (2.0 * grid.dx[dir]);
}

