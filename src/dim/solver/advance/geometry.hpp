#pragma once

#include <array>
#include <stdexcept>

namespace dim {

template<int DIM>
inline int total_cells(const std::array<int, DIM>& N)
{
    int total = 1;

    for (int d = 0; d < DIM; ++d) {
        if (N[d] <= 0) {
            throw std::runtime_error("dim::total_cells: grid size must be positive");
        }
        total *= N[d];
    }

    return total;
}


// [0] Compute strides (row-major)
template<int DIM>
inline std::array<int, DIM> compute_strides(
    const std::array<int, DIM>& N
)
{
    std::array<int, DIM> stride{};
    stride[0] = 1;

    for (int d = 1; d < DIM; ++d) {
        stride[d] = stride[d - 1] * N[d - 1];
    }

    return stride;
}


// [1] Flatten multi-index -> linear index
template<int DIM>
inline int flatten_index(
    const std::array<int, DIM>& idx,
    const std::array<int, DIM>& stride
)
{
    int linear = 0;

    for (int d = 0; d < DIM; ++d) {
        linear += idx[d] * stride[d];
    }

    return linear;
}

template<int DIM>
inline std::array<int, DIM> unflatten_index(
    int linear,
    const std::array<int, DIM>& N
)
{
    std::array<int, DIM> idx{};
    int stride = 1;

    for (int d = 0; d < DIM; ++d) {
        idx[d] = (linear / stride) % N[d];
        stride *= N[d];
    }

    return idx;
}

template<int DIM>
inline std::array<double, DIM> compute_cell_center(
    const std::array<int, DIM>& idx,
    const std::array<double, DIM>& domain_min,
    const std::array<double, DIM>& dx
)
{
    std::array<double, DIM> x{};

    for (int d = 0; d < DIM; ++d) {
        x[d] = domain_min[d] + (idx[d] + 0.5) * dx[d];
    }

    return x;
}

} 


