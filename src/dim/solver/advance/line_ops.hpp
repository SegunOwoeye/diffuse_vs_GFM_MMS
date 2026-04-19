#pragma once

#include <vector>
#include <array>

#include "src/dim/state.hpp"


// [0] Extract line along direction
template<int DIM, int NMAT>
inline void extract_line(
    const std::vector<Conserved<DIM, NMAT>>& U,
    const std::array<int, DIM>& N,
    const std::array<int, DIM>& stride,
    int dir,
    const std::array<int, DIM>& base_idx,
    std::vector<Conserved<DIM, NMAT>>& line
)
{
    const int length = N[dir];

    line.resize(length);

    std::array<int, DIM> idx = base_idx;

    for (int i = 0; i < length; ++i) {
        idx[dir] = i;

        const int linear = flatten_index<DIM>(idx, stride);
        line[i] = U[linear];
    }
}


// [1] Write line back
template<int DIM, int NMAT>
inline void write_line(
    std::vector<Conserved<DIM, NMAT>>& U,
    const std::array<int, DIM>& N,
    const std::array<int, DIM>& stride,
    int dir,
    const std::array<int, DIM>& base_idx,
    const std::vector<Conserved<DIM, NMAT>>& line
)
{
    std::array<int, DIM> idx = base_idx;

    for (int i = 0; i < static_cast<int>(line.size()); ++i) {
        idx[dir] = i;

        const int linear = flatten_index<DIM>(idx, stride);
        U[linear] = line[i];
    }
}

