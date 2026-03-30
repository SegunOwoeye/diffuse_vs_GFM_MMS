#pragma once

#include <array>
#include <stdexcept>


// [0] Convert flat index -> multi-index
template<int DIM>
inline std::array<int, DIM> unflatten_index(
    int id,
    const std::array<int, DIM>& N
)
{
    std::array<int, DIM> idx;

    for (int d = 0; d < DIM; ++d) {
        idx[d] = id % N[d];
        id /= N[d];
    }

    return idx;
}


// [1] Convert multi-index -> flat index
template<int DIM>
inline int flatten_index(
    const std::array<int, DIM>& idx,
    const std::array<int, DIM>& N
)
{
    int id = idx[0];
    int stride = 1;

    for (int d = 1; d < DIM; ++d) {
        stride *= N[d - 1];
        id += idx[d] * stride;
    }

    return id;
}


// [2] Check interior cell
template<int DIM>
inline bool is_interior(
    const std::array<int, DIM>& idx,
    const std::array<int, DIM>& N
)
{
    for (int d = 0; d < DIM; ++d) {
        if (idx[d] <= 0 || idx[d] >= N[d] - 1) {
            return false;
        }
    }

    return true;
}


// [3] Offset index in direction dir
template<int DIM>
inline std::array<int, DIM> offset(
    const std::array<int, DIM>& idx,
    int dir,
    int step
)
{
    std::array<int, DIM> out = idx;
    out[dir] += step;
    return out;
}


// [4] Check valid index
template<int DIM>
inline bool is_valid(
    const std::array<int, DIM>& idx,
    const std::array<int, DIM>& N
)
{
    for (int d = 0; d < DIM; ++d) {
        if (idx[d] < 0 || idx[d] >= N[d]) {
            return false;
        }
    }

    return true;
}