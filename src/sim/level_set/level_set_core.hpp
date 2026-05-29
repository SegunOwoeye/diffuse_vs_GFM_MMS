#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>

template<int DIM>
struct LevelSetGrid {
    std::array<int, DIM> N{};
    std::array<double, DIM> dx{};
    std::array<int, DIM> stride{};
};


// [1] Compute strides
template<int DIM>
inline std::array<int, DIM> compute_strides(
    const LevelSetGrid<DIM>& grid
)
{
    std::array<int, DIM> stride{};
    stride[0] = 1;

    for (int d = 1; d < DIM; ++d) {
        if (grid.N[d - 1] <= 0) {
            throw std::runtime_error("compute_strides: invalid grid");
        }

        stride[d] = stride[d - 1] * grid.N[d - 1];
    }

    return stride;
}


// [2] Build grid
template<int DIM>
inline LevelSetGrid<DIM> make_level_set_grid(
    const std::array<int, DIM>& N,
    const std::array<double, DIM>& dx
)
{
    LevelSetGrid<DIM> grid{};
    grid.N = N;
    grid.dx = dx;
    grid.stride = compute_strides<DIM>(grid);
    return grid;
}


// [3] Total cells
template<int DIM>
inline std::size_t total_cells(
    const LevelSetGrid<DIM>& grid
)
{
    std::size_t total = 1;

    for (int d = 0; d < DIM; ++d) {
        if (grid.N[d] <= 0) {
            throw std::runtime_error("total_cells: invalid grid");
        }

        total *= static_cast<std::size_t>(grid.N[d]);
    }

    return total;
}


// [4] Valid index check
template<int DIM>
inline bool is_valid_index(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid
)
{
    for (int d = 0; d < DIM; ++d) {
        if (idx[d] < 0 || idx[d] >= grid.N[d]) {
            return false;
        }
    }
    return true;
}


// [5] Boundary check
template<int DIM>
inline bool is_boundary_cell(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid
)
{
    if (!is_valid_index<DIM>(idx, grid)) {
        throw std::runtime_error("is_boundary_cell: invalid index");
    }

    for (int d = 0; d < DIM; ++d) {
        if (idx[d] == 0 || idx[d] == grid.N[d] - 1) {
            return true;
        }
    }

    return false;
}


// [6] Interior check
template<int DIM>
inline bool is_interior_cell(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid
)
{
    if (!is_valid_index<DIM>(idx, grid)) {
        return false;
    }

    for (int d = 0; d < DIM; ++d) {
        if (idx[d] <= 0 || idx[d] >= grid.N[d] - 1) {
            return false;
        }
    }

    return true;
}


// [7] Flatten (grid-aware)
template<int DIM>
inline int flatten_index(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid
)
{
    if (!is_valid_index<DIM>(idx, grid)) {
        throw std::runtime_error("flatten_index: OOB");
    }

    int id = 0;

    for (int d = 0; d < DIM; ++d) {
        id += idx[d] * grid.stride[d];
    }

    return id;
}


// [8] Unflatten
template<int DIM>
inline std::array<int, DIM> unflatten_index(
    int id,
    const LevelSetGrid<DIM>& grid
)
{
    const int total = static_cast<int>(total_cells(grid));

    if (id < 0 || id >= total) {
        throw std::runtime_error("unflatten_index: OOB");
    }

    std::array<int, DIM> idx{};

    for (int d = DIM - 1; d >= 0; --d) {
        idx[d] = id / grid.stride[d];
        id %= grid.stride[d];
    }

    return idx;
}


// [9] Offset (no bounds check)
template<int DIM>
inline std::array<int, DIM> offset_index(
    const std::array<int, DIM>& idx,
    int dir,
    int step
)
{
    if (dir < 0 || dir >= DIM) {
        throw std::runtime_error("offset_index: bad dir");
    }

    auto out = idx;
    out[dir] += step;
    return out;
}


// [10] Safe offset
template<int DIM>
inline std::array<int, DIM> offset_index_checked(
    const std::array<int, DIM>& idx,
    int dir,
    int step,
    const LevelSetGrid<DIM>& grid
)
{
    const auto out = offset_index<DIM>(idx, dir, step);

    if (!is_valid_index<DIM>(out, grid)) {
        throw std::runtime_error("offset_index_checked: OOB");
    }

    return out;
}


// [11] Try offset
template<int DIM>
inline bool try_offset_index(
    const std::array<int, DIM>& idx,
    int dir,
    int step,
    const LevelSetGrid<DIM>& grid,
    std::array<int, DIM>& out
)
{
    out = offset_index<DIM>(idx, dir, step);
    return is_valid_index<DIM>(out, grid);
}


// [12] Clamp to interior (for derivatives / normals)
template<int DIM>
inline std::array<int, DIM> clamp_to_interior(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid
)
{
    if (!is_valid_index<DIM>(idx, grid)) {
        throw std::runtime_error("clamp_to_interior: invalid index");
    }

    std::array<int, DIM> out = idx;

    for (int d = 0; d < DIM; ++d) {
        if (grid.N[d] < 3) {
            throw std::runtime_error("clamp_to_interior: grid too small");
        }

        if (out[d] == 0) {
            out[d] = 1;
        }
        else if (out[d] == grid.N[d] - 1) {
            out[d] = grid.N[d] - 2;
        }
    }

    return out;
}



