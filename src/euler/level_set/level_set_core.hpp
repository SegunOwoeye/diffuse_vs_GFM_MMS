#pragma once

#include <array>
#include <stdexcept>


// [0] Level Set Grid Descriptor
template<int DIM>
struct LevelSetGrid {
    std::array<int, DIM> N;
    std::array<double, DIM> dx;
};


/* [1] Compute strides for row-major layout
    stride[d] = product of N[0] ... N[d-1]
*/
template<int DIM>
inline std::array<int, DIM> compute_strides(
    const LevelSetGrid<DIM>& grid
)
{
    std::array<int, DIM> stride{};

    stride[0] = 1;

    for (int d = 1; d < DIM; ++d) {
        stride[d] = stride[d - 1] * grid.N[d - 1];
    }

    return stride;
}


// [2] Total Number of Cells

template<int DIM>
inline int total_cells(
    const LevelSetGrid<DIM>& grid
)
{
    int total = 1;

    for (int d = 0; d < DIM; ++d) {
        if (grid.N[d] <= 0) {
            throw std::runtime_error("total_cells: invalid grid dimension");
        }

        total *= grid.N[d];
    }

    return total;
}


// [3] Flatten Multi-Index (row-major)
template<int DIM>
inline int flatten_index(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid
)
{
    const auto stride = compute_strides<DIM>(grid);

    int id = 0;

    for (int d = 0; d < DIM; ++d) {

        if (idx[d] < 0 || idx[d] >= grid.N[d]) {
            throw std::runtime_error("flatten_index: index out of bounds");
        }

        id += idx[d] * stride[d];
    }

    return id;
}


// [4] Check Interior Cell
template<int DIM>
inline bool is_interior_cell(
    const std::array<int, DIM>& idx,
    const LevelSetGrid<DIM>& grid
)
{
    for (int d = 0; d < DIM; ++d) {
        if (idx[d] <= 0 || idx[d] >= grid.N[d] - 1) {
            return false;
        }
    }

    return true;
}


// [5] Offset Index by One Cell in Direction DIR

template<int DIM>
inline std::array<int, DIM> offset_index(
    const std::array<int, DIM>& idx,
    int dir,
    int step
)
{
    if (dir < 0 || dir >= DIM) {
        throw std::runtime_error("offset_index: invalid direction");
    }

    std::array<int, DIM> out = idx;
    out[dir] += step;
    return out;
}

