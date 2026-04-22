#pragma once

#include <array>
#include <vector>

#include "src/dim/state.hpp"
#include "src/dim/solver/advance/geometry.hpp"

namespace dim {

    // [0] Extract line along direction
    template<int DIM>
    inline void extract_line(
        const std::vector<State<DIM>>& U,
        const std::array<int, DIM>& N,
        const std::array<int, DIM>& stride,
        int dir,
        const std::array<int, DIM>& base_idx,
        std::vector<State<DIM>>& line
    )
    {
        const int length = N[dir];
        line.resize(length);

        std::array<int, DIM> idx = base_idx;
        for (int i = 0; i < length; ++i) {
            idx[dir] = i;
            line[i] = U[flatten_index<DIM>(idx, stride)];
        }
    }

    // [1] Write line back
    template<int DIM>
    inline void write_line(
        std::vector<State<DIM>>& U,
        const std::array<int, DIM>& N,
        const std::array<int, DIM>& stride,
        int dir,
        const std::array<int, DIM>& base_idx,
        const std::vector<State<DIM>>& line
    )
    {
        std::array<int, DIM> idx = base_idx;

        for (int i = 0; i < static_cast<int>(line.size()); ++i) {
            idx[dir] = i;
            U[flatten_index<DIM>(idx, stride)] = line[i];
        }
    }

}

