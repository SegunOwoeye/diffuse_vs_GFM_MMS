#pragma once

#include <vector>

#include "src/euler/state.hpp"
#include "src/euler/solver/solver_context.hpp"
#include "src/euler/grid/grid_utils.hpp"

template<int DIM>
inline void apply_boundary_conditions(
    std::vector<Conserved<DIM>>& U,
    const SolverContext<DIM>& ctx
)
{
    const int Ntot = static_cast<int>(U.size());
    const std::vector<Conserved<DIM>> U_old = U;

    for (int id = 0; id < Ntot; ++id) {
        const auto idx = unflatten_index<DIM>(id, ctx.level_set_grid);
        std::array<int, DIM> src_idx = idx;
        bool is_boundary = false;

        for (int d = 0; d < DIM; ++d) {
            if (idx[d] == 0) {
                src_idx[d] = 1;
                is_boundary = true;
            }
            else if (idx[d] == ctx.N[d] - 1) {
                src_idx[d] = ctx.N[d] - 2;
                is_boundary = true;
            }
        }

        if (!is_boundary) {
            continue;
        }

        const int src_id = flatten_index<DIM>(src_idx, ctx.N);
        U[id] = U_old[src_id];

        for (int d = 0; d < DIM; ++d) {
            if (idx[d] == 0 && ctx.bc_lo[d] == BoundaryConditionType::reflective) {
                U[id].mom[d] *= -1.0;
            }
            else if (idx[d] == ctx.N[d] - 1 && ctx.bc_hi[d] == BoundaryConditionType::reflective) {
                U[id].mom[d] *= -1.0;
            }
        }
    }
}
