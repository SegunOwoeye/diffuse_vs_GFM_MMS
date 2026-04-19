#pragma once

#include <array>


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


// [1] Flatten multi-index → linear index
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

