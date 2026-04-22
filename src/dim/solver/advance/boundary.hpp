#pragma once

#include <array>
#include <vector>

#include "src/dim/state.hpp"

namespace dim {

    template<int DIM>
    inline void apply_boundary_conditions(
        std::vector<State<DIM>>&,
        const std::array<int, DIM>&
    )
    {
        // Zero-gradient boundaries are handled implicitly by the line solver.
    }

} 
