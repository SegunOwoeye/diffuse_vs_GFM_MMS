#pragma once

#include <vector>
#include <array>

#include "src/dim/state.hpp"


// [0] Zero-gradient boundary (placeholder)
template<int DIM, int NMAT>
inline void apply_boundary_conditions(
    std::vector<Conserved<DIM, NMAT>>& U,
    const std::array<int, DIM>& N
)
{
    // intentionally empty for now
    // handled implicitly in flux via copying
}